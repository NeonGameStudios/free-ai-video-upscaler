/**
 * Video processing worker using Real-ESRGAN and Real-CUGAN for upscaling.
 *
 * This worker handles video frame extraction, AI upscaling,
 * and video encoding with multiple format support.
 */

import {
  BlobSource,
  BufferTarget,
  CanvasSource,
  Input,
  MP4,
  Mp4OutputFormat,
  Output,
  QUALITY_HIGH,
  StreamTarget,
  VideoSample,
  VideoSampleSink,
  WebMOutputFormat,
  AudioSampleSink,
  AudioSampleSource,
  EncodedAudioPacketSource,
  EncodedPacketSink,
  EncodedPacket,
  type AudioCodec,
} from 'mediabunny';

import { Upscaler } from './upscaler';

import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
  InitData,
  SwitchModelData,
  ProcessSettings,
  Resolution,
  OutputFormat,
} from './types/worker-messages';

// Worker state
let upscaler: Upscaler | null = null;
let upscaled_canvas: OffscreenCanvas;
let original_canvas: OffscreenCanvas;
let resolution: Resolution;
let ctx: ImageBitmapRenderingContext | null;
let currentScale: number = 4;
let isModelSwitching: boolean = false;
let pendingModelSwitch: SwitchModelData | null = null;
let cancelRequested: boolean = false;

class CancelledError extends Error {
  constructor() {
    super('Processing cancelled');
    this.name = 'CancelledError';
  }
}

function throwIfCancelled(): void {
  if (cancelRequested) {
    throw new CancelledError();
  }
}

function cancelCurrentJob(): void {
  cancelRequested = true;
}

/**
 * Check if WebGPU is supported in this environment.
 */
async function isSupported(): Promise<void> {
  const supported = await Upscaler.isWebGPUSupported();

  postMessage({
    cmd: 'isSupported',
    data: supported
  } satisfies WorkerResponseMessage);
}

/**
 * Initialize the worker with canvases and create upscaler instance.
 */
async function init(config: InitData): Promise<void> {
  try {
    if (upscaler) {
      await upscaler.dispose();
      upscaler = null;
    }

    // Store canvases
    upscaled_canvas = config.upscaled;
    original_canvas = config.original;
    resolution = config.resolution;

    // Get the scale from config
    currentScale = config.modelConfig.scale;

    const previewOutputResolution = getOutputResolution(config.targetHeight);

    // Set up output canvas dimensions
    upscaled_canvas.width = previewOutputResolution.width;
    upscaled_canvas.height = previewOutputResolution.height;
    original_canvas.width = previewOutputResolution.width;
    original_canvas.height = previewOutputResolution.height;

    // Set up original canvas context for "before" preview
    ctx = original_canvas.getContext('bitmaprenderer');

    // Create upscaler
    upscaler = new Upscaler({
      modelId: config.modelConfig.modelId,
      scale: config.modelConfig.scale,
      tileSize: config.modelConfig.tileSize || 256,
      tilePadding: config.modelConfig.tilePadding || 16,
      inputWidth: config.modelConfig.inputWidth,
      inputHeight: config.modelConfig.inputHeight,
      inputMultiple: config.modelConfig.inputMultiple,
      denoiseLevel: config.modelConfig.denoiseLevel,
    });

    // Progress callback for model loading
    const onProgress = (progress: number, message: string) => {
      postMessage({ cmd: 'modelLoading', data: progress } satisfies WorkerResponseMessage);
    };

    // Initialize the model with progress tracking
    await upscaler.init(upscaled_canvas, onProgress);

    postMessage({ cmd: 'status', data: 'Rendering preview frame...' } satisfies WorkerResponseMessage);
    await renderPreviewFrame(config.bitmap, config.targetHeight);

    postMessage({ cmd: 'modelLoaded' } satisfies WorkerResponseMessage);
  } catch (e) {
    await upscaler?.dispose();
    upscaler = null;
    console.error('Failed to initialize upscaler:', e);
    postMessage({
      cmd: 'error',
      data: `Failed to initialize AI upscaler: ${e}`
    } satisfies WorkerResponseMessage);
  }
}

/**
 * Switch to a different model.
 */
async function switchModel(data: SwitchModelData): Promise<void> {
  if (!upscaler) {
    postMessage({
      cmd: 'error',
      data: 'Upscaler not initialized'
    } satisfies WorkerResponseMessage);
    return;
  }

  // Prevent concurrent model switches
  if (isModelSwitching) {
    pendingModelSwitch = data;
    console.log('Model switch already in progress, queueing latest request');
    return;
  }

  isModelSwitching = true;

  try {
    let nextSwitch: SwitchModelData | null = data;

    while (nextSwitch) {
      const activeSwitch = nextSwitch;
      pendingModelSwitch = null;

      // Update scale
      currentScale = activeSwitch.modelConfig.scale;

      const previewOutputResolution = getOutputResolution(activeSwitch.targetHeight);

      // Update canvas dimensions for new scale and target output size
      upscaled_canvas.width = previewOutputResolution.width;
      upscaled_canvas.height = previewOutputResolution.height;
      original_canvas.width = previewOutputResolution.width;
      original_canvas.height = previewOutputResolution.height;

      // Progress callback for model loading
      const onProgress = (progress: number, message: string) => {
        postMessage({ cmd: 'modelLoading', data: progress } satisfies WorkerResponseMessage);
      };

      // Switch model with progress tracking
      await upscaler.switchModel({
        modelId: activeSwitch.modelConfig.modelId,
        scale: activeSwitch.modelConfig.scale,
        tileSize: activeSwitch.modelConfig.tileSize,
        tilePadding: activeSwitch.modelConfig.tilePadding,
        inputWidth: activeSwitch.modelConfig.inputWidth,
        inputHeight: activeSwitch.modelConfig.inputHeight,
        inputMultiple: activeSwitch.modelConfig.inputMultiple,
        denoiseLevel: activeSwitch.modelConfig.denoiseLevel,
      }, onProgress);

      if (!pendingModelSwitch) {
        postMessage({ cmd: 'status', data: 'Rendering preview frame...' } satisfies WorkerResponseMessage);
        await renderPreviewFrame(activeSwitch.bitmap, activeSwitch.targetHeight);
      }

      nextSwitch = pendingModelSwitch;
    }

    postMessage({ cmd: 'modelLoaded' } satisfies WorkerResponseMessage);
  } catch (e) {
    pendingModelSwitch = null;
    console.error('Failed to switch model:', e);
    postMessage({
      cmd: 'error',
      data: `Failed to switch model: ${e}`
    } satisfies WorkerResponseMessage);
  } finally {
    isModelSwitching = false;
  }
}

/**
 * Get the output format handler.
 */
function getOutputFormat(format: OutputFormat) {
  switch (format) {
    case 'webm':
      return new WebMOutputFormat();
    case 'mp4':
    default:
      return new Mp4OutputFormat();
  }
}

/**
 * Get codec for format.
 */
function getCodec(format: OutputFormat): 'avc' | 'vp9' | 'hevc' | 'av1' | 'vp8' {
  switch (format) {
    case 'webm':
      return 'vp9';
    case 'mp4':
    default:
      return 'avc';
  }
}

/**
 * Get MIME type for format.
 */
function getMimeType(format: OutputFormat): string {
  switch (format) {
    case 'webm':
      return 'video/webm';
    case 'mp4':
    default:
      return 'video/mp4';
  }
}

/**
 * Calculate the actual output canvas dimensions.
 * Resolution presets cap native model output instead of upscaling beyond it.
 */
function getOutputResolution(targetHeight?: number): Resolution {
  const nativeWidth = resolution.width * currentScale;
  const nativeHeight = resolution.height * currentScale;

  if (!targetHeight || nativeHeight <= targetHeight) {
    return { width: nativeWidth, height: nativeHeight };
  }

  const aspectRatio = nativeWidth / nativeHeight;
  const height = makeEven(targetHeight);
  const width = makeEven(Math.round(height * aspectRatio));

  return { width, height };
}

/**
 * Calculate the actual canvas dimensions used for encoding.
 */
function getEncodeResolution(settings: ProcessSettings): Resolution {
  return getOutputResolution(settings.targetHeight);
}

/**
 * Calculate the frame size sent into the model.
 * When the output preset caps native model output, resize before inference so
 * we do not spend time generating pixels that will be downscaled afterward.
 */
function getInferenceResolution(encodeResolution: Resolution): Resolution {
  const inputForEncodeWidth = makeEven(Math.round(encodeResolution.width / currentScale));
  const inputForEncodeHeight = makeEven(Math.round(encodeResolution.height / currentScale));

  if (inputForEncodeHeight >= resolution.height) {
    return { width: resolution.width, height: resolution.height };
  }

  return {
    width: Math.min(resolution.width, inputForEncodeWidth),
    height: Math.min(resolution.height, inputForEncodeHeight),
  };
}

function makeEven(value: number): number {
  const rounded = Math.max(2, Math.round(value));
  return rounded % 2 === 0 ? rounded : rounded - 1;
}

function normalizeAudioPacketTimestamp(packet: EncodedPacket): EncodedPacket | null {
  if (packet.timestamp >= 0) {
    return packet;
  }

  const endTimestamp = packet.timestamp + packet.duration;
  if (endTimestamp <= 0) {
    return null;
  }

  return new EncodedPacket(
    packet.data,
    packet.type,
    0,
    endTimestamp,
    packet.sequenceNumber,
    packet.byteLength,
    packet.sideData
  );
}

async function renderPreviewFrame(source: ImageBitmap, targetHeight?: number): Promise<void> {
  if (!upscaler) return;

  const previewOutputResolution = getOutputResolution(targetHeight);
  const inferenceResolution = getInferenceResolution(previewOutputResolution);
  const needsInferenceResize =
    inferenceResolution.width !== resolution.width ||
    inferenceResolution.height !== resolution.height;

  upscaled_canvas.width = previewOutputResolution.width;
  upscaled_canvas.height = previewOutputResolution.height;
  original_canvas.width = previewOutputResolution.width;
  original_canvas.height = previewOutputResolution.height;

  let beforeBitmap: ImageBitmap | null = null;
  let inferenceBitmap: ImageBitmap | null = null;

  try {
    beforeBitmap = await createImageBitmap(source, {
      resizeWidth: previewOutputResolution.width,
      resizeHeight: previewOutputResolution.height,
    });

    if (needsInferenceResize) {
      inferenceBitmap = await createImageBitmap(source, {
        resizeWidth: inferenceResolution.width,
        resizeHeight: inferenceResolution.height,
      });
      await upscaler.render(inferenceBitmap);
    } else {
      await upscaler.render(source);
    }

    if (ctx) {
      ctx.transferFromImageBitmap(beforeBitmap);
      beforeBitmap = null;
    }
  } finally {
    beforeBitmap?.close();
    inferenceBitmap?.close();
    upscaler.clearFrameResources();
  }
}

/**
 * Main video processing function.
 * Accepts either inputHandle (FileSystemFileHandle) or inputFile (File) for remuxed MKV files.
 */
async function initRecording(
  inputHandle: FileSystemFileHandle | undefined,
  inputFile: File | undefined,
  outputHandle: FileSystemFileHandle | undefined,
  settings: ProcessSettings
): Promise<void> {
  cancelRequested = false;

  if (!upscaler || !upscaler.isReady()) {
    postMessage({
      cmd: 'error',
      data: 'Upscaler model not loaded'
    } satisfies WorkerResponseMessage);
    return;
  }

  let input: Input | null = null;
  let inferenceCanvas: OffscreenCanvas | null = null;
  let encodeCanvasForCleanup: OffscreenCanvas | null = null;
  let writable: WritableStream | null = null;
  let audioSource: AudioSampleSource | null = null;
  let audioPacketSource: EncodedAudioPacketSource | null = null;
  let audioPacketSourceClosed = false;
  let videoSource: CanvasSource | null = null;
  let outputStarted = false;
  let outputFinalized = false;
  let wasCancelled = false;

  try {
    // Get the file from handle or use provided file directly (for remuxed MKV)
    let file: File;
    if (inputFile) {
      file = inputFile;
    } else if (inputHandle) {
      file = await inputHandle.getFile();
    } else {
      postMessage({
        cmd: 'error',
        data: 'No input file provided'
      } satisfies WorkerResponseMessage);
      return;
    }

    // MediaBunny handles streaming from the blob for large files
    const source = new BlobSource(file);

    input = new Input({
      formats: [MP4],
      source
    });

    let target: BufferTarget | StreamTarget;
    if (outputHandle) {
      writable = await outputHandle.createWritable();
      target = new StreamTarget(writable);
    } else {
      target = new BufferTarget();
    }

    const outputFormat = getOutputFormat(settings.outputFormat);
    const output = new Output({
      format: outputFormat,
      target: target,
    });

    const encodeResolution = getEncodeResolution(settings);
    const inferenceResolution = getInferenceResolution(encodeResolution);
    const renderOutputWidth = inferenceResolution.width * currentScale;
    const renderOutputHeight = inferenceResolution.height * currentScale;
    const needsEncodeResize =
      encodeResolution.width !== renderOutputWidth ||
      encodeResolution.height !== renderOutputHeight;
    const encodeCanvas = needsEncodeResize
      ? new OffscreenCanvas(encodeResolution.width, encodeResolution.height)
      : upscaled_canvas;
    encodeCanvasForCleanup = needsEncodeResize ? encodeCanvas : null;
    const encodeCtx = needsEncodeResize
      ? encodeCanvas.getContext('2d', { alpha: false })
      : null;
    const needsInferenceResize =
      inferenceResolution.width !== resolution.width ||
      inferenceResolution.height !== resolution.height;
    inferenceCanvas = needsInferenceResize
      ? new OffscreenCanvas(inferenceResolution.width, inferenceResolution.height)
      : null;
    const inferenceCtx = inferenceCanvas?.getContext('2d', { alpha: false }) || null;

    videoSource = new CanvasSource(encodeCanvas, {
      codec: getCodec(settings.outputFormat),
      bitrate: QUALITY_HIGH,
      keyFrameInterval: 60,
    });

    output.addVideoTrack(videoSource, { frameRate: 30 });

    // Set up audio passthrough
    const audioTrack = await input.getPrimaryAudioTrack();
    let audioSink: AudioSampleSink | null = null;
    let audioPacketIterator: AsyncIterator<EncodedPacket> | null = null;
    let pendingAudioPacket: EncodedPacket | null = null;
    let audioPacketMeta: EncodedAudioChunkMetadata | undefined;

    if (audioTrack) {
      const sourceCodec = audioTrack.codec;
      const supportedAudioCodecs = outputFormat.getSupportedAudioCodecs();

      if (sourceCodec && supportedAudioCodecs.includes(sourceCodec)) {
        audioPacketSource = new EncodedAudioPacketSource(sourceCodec);
        output.addAudioTrack(audioPacketSource);

        const audioPacketSink = new EncodedPacketSink(audioTrack);
        audioPacketIterator = audioPacketSink.packets()[Symbol.asyncIterator]();

        const decoderConfig = await audioTrack.getDecoderConfig();
        audioPacketMeta = { decoderConfig: decoderConfig ?? undefined };
        console.log(`Copying ${sourceCodec} audio without re-encoding`);
      } else {
        // Fall back to re-encoding only when the source codec cannot live in the output container.
        const audioCodec = (settings.outputFormat === 'webm' ? 'opus' : 'aac') as AudioCodec;
        audioSource = new AudioSampleSource({
          codec: audioCodec,
          bitrate: 128000,
        });
        output.addAudioTrack(audioSource);
        audioSink = new AudioSampleSink(audioTrack);
        console.log(`Re-encoding audio to ${audioCodec}`);
      }
    }

    await output.start();
    outputStarted = true;

    const videoTrack = await input.getPrimaryVideoTrack();

    if (!videoTrack) {
      postMessage({
        cmd: 'error',
        data: 'No video track found in input file'
      } satisfies WorkerResponseMessage);
      return;
    }

    const decodable = await videoTrack.canDecode();
    if (!decodable) {
      postMessage({
        cmd: 'error',
        data: 'Video codec not supported for decoding'
      } satisfies WorkerResponseMessage);
      return;
    }

    const videoSink = new VideoSampleSink(videoTrack);
    const duration = await input.computeDuration();
    const start_time = performance.now();
    let lastProgress = -1;

    // Track audio progress separately
    let lastAudioTimestamp = 0;

    function reportProgress(sample: VideoSample) {
      const time_elapsed = performance.now() - start_time;
      const completedTimestamp = Math.min(duration, sample.timestamp + sample.duration);
      const percentComplete = duration > 0 ? (completedTimestamp / duration) * 100 : 100;
      const progress = Math.min(99, Math.floor(percentComplete));

      if (progress !== lastProgress) {
        postMessage({ cmd: 'progress', data: progress });
        lastProgress = progress;
      }

      if (time_elapsed > 1000 && percentComplete > 0) {
        const processing_rate = percentComplete / time_elapsed;
        const eta = Math.max(0, Math.round(((100 - percentComplete) / processing_rate) / 1000));
        postMessage({ cmd: 'eta', data: prettyTime(eta) });
      } else {
        postMessage({ cmd: 'eta', data: 'calculating...' });
      }
    }

    // Process audio up to a given timestamp
    async function processAudioUpTo(timestamp: number) {
      throwIfCancelled();

      if (audioPacketSource && audioPacketIterator) {
        while (true) {
          throwIfCancelled();

          if (!pendingAudioPacket) {
            const next = await audioPacketIterator.next();
            if (next.done) {
              if (!audioPacketSourceClosed) {
                audioPacketSource.close();
                audioPacketSourceClosed = true;
              }
              audioPacketIterator = null;
              return;
            }
            pendingAudioPacket = next.value;
          }

          if (pendingAudioPacket.timestamp >= timestamp) {
            return;
          }

          const packet = normalizeAudioPacketTimestamp(pendingAudioPacket);
          if (packet) {
            await audioPacketSource.add(packet, audioPacketMeta);
            throwIfCancelled();
          }
          pendingAudioPacket = null;
        }
      }

      if (!audioSink || !audioSource) return;

      // Get audio samples up to this video frame's timestamp
      for await (const audioSample of audioSink.samples(lastAudioTimestamp, timestamp)) {
        throwIfCancelled();
        await audioSource.add(audioSample);
        lastAudioTimestamp = audioSample.timestamp + audioSample.duration;
        audioSample.close();
      }
    }

    // Loop over all frames
    postMessage({ cmd: 'progress', data: 0 });
    for await (const sample of videoSink.samples()) {
      let videoFrame: VideoFrame | null = null;
      try {
        throwIfCancelled();

        // Process audio up to this frame's timestamp
        await processAudioUpTo(sample.timestamp + sample.duration);

        videoFrame = sample.toVideoFrame();
        throwIfCancelled();

        // Render through upscaler (skip "before" preview during processing for speed)
        if (inferenceCanvas && inferenceCtx) {
          inferenceCtx.drawImage(
            videoFrame,
            0,
            0,
            inferenceResolution.width,
            inferenceResolution.height
          );
          await upscaler.render(inferenceCanvas);
        } else {
          await upscaler.render(videoFrame);
        }
        throwIfCancelled();

        if (encodeCtx) {
          encodeCtx.drawImage(
            upscaled_canvas,
            0,
            0,
            encodeResolution.width,
            encodeResolution.height
          );
        }

        // Add frame to output video
        await videoSource.add(sample.timestamp, sample.duration);
        throwIfCancelled();

        reportProgress(sample);
      } finally {
        // Cleanup - always close resources even on error
        videoFrame?.close();
        sample.close();
      }
    }

    // Process any remaining audio
    await processAudioUpTo(Number.POSITIVE_INFINITY);
    throwIfCancelled();

    videoSource.close();
    audioSource?.close();

    if (audioPacketSource && !audioPacketSourceClosed) {
      audioPacketSource.close();
      audioPacketSourceClosed = true;
    }

    await output.finalize();
    outputFinalized = true;
    postMessage({ cmd: 'progress', data: 100 });
    postMessage({ cmd: 'eta', data: prettyTime(0) });

    if (writable) {
      postMessage({ cmd: 'finished', data: null }, []);
    } else {
      const buffer = (output.target as BufferTarget).buffer;
      postMessage({ cmd: 'finished', data: buffer }, [buffer]);
    }
  } catch (e) {
    if (e instanceof CancelledError) {
      console.log('Video processing cancelled');
      wasCancelled = true;
      return;
    }

    console.error('Video processing error:', e);
    postMessage({
      cmd: 'error',
      data: `Video processing failed: ${e}`
    } satisfies WorkerResponseMessage);
  } finally {
    if (cancelRequested) {
      try {
        videoSource?.close();
      } catch {
        // Ignore close errors after cancellation.
      }

      try {
        audioSource?.close();
      } catch {
        // Ignore close errors after cancellation.
      }

      if (audioPacketSource && !audioPacketSourceClosed) {
        try {
          audioPacketSource.close();
          audioPacketSourceClosed = true;
        } catch {
          // Ignore close errors after cancellation.
        }
      }

      if (outputStarted && !outputFinalized) {
        try {
          await writable?.abort();
        } catch {
          // Ignore writer abort errors after cancellation.
        }
      }
    }

    cancelRequested = false;
    upscaler?.clearFrameResources();
    input?.dispose();

    if (inferenceCanvas) {
      inferenceCanvas.width = 0;
      inferenceCanvas.height = 0;
    }

    if (encodeCanvasForCleanup) {
      encodeCanvasForCleanup.width = 0;
      encodeCanvasForCleanup.height = 0;
    }

    if (wasCancelled) {
      postMessage({ cmd: 'cancelled' } satisfies WorkerResponseMessage);
    }
  }
}

/**
 * Format seconds into HH:MM:SS or MM:SS.
 */
function prettyTime(secs: number): string {
  const sec_num = parseInt(secs.toString(), 10);
  const hours = Math.floor(sec_num / 3600);
  const minutes = Math.floor(sec_num / 60) % 60;
  const seconds = sec_num % 60;

  return [hours, minutes, seconds]
    .map(v => v < 10 ? "0" + v : v)
    .filter((v, i) => v !== "00" || i > 0)
    .join(":");
}

/**
 * Worker message handler with type-safe message routing.
 */
self.onmessage = async function (event: MessageEvent<WorkerRequestMessage>) {
  if (!event.data.cmd) return;

  switch (event.data.cmd) {
    case 'init':
      await init(event.data.data);
      break;

    case 'isSupported':
      await isSupported();
      break;

    case 'switchModel':
      await switchModel(event.data.data);
      break;

    case 'process':
      await initRecording(
        event.data.inputHandle,
        event.data.inputFile,
        event.data.outputHandle,
        event.data.settings
      );
      break;

    case 'cancel':
      cancelCurrentJob();
      break;
  }
};
