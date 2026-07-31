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
  MATROSKA,
  MP4,
  Mp4OutputFormat,
  Output,
  QUALITY_HIGH,
  StreamTarget,
  VideoSample,
  VideoSampleSink,
  WEBM,
  WebMOutputFormat,
  AudioSampleSink,
  AudioSampleSource,
  type AudioSample,
  EncodedAudioPacketSource,
  EncodedPacketSink,
  EncodedPacket,
  type AudioCodec,
  type StreamTargetChunk,
  canEncodeVideo,
} from 'mediabunny';

import { Upscaler } from './upscaler';
import {
  resolveInferenceResolution,
  resolveOutputResolution,
} from './types/worker-messages';

import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
  InitData,
  SwitchModelData,
  RenderPreviewData,
  ProcessSettings,
  Resolution,
  OutputFormat,
  FrameTiming,
  PipelineTelemetry,
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
let modelSwitchScheduled = false;
let pendingPreviewRender: RenderPreviewData | null = null;
let previewRenderScheduled = false;
let previewOperationTail: Promise<void> = Promise.resolve();
let cancelRequested: boolean = false;
let finalizationInProgress: boolean = false;
const DEFAULT_AUTO_TARGET_HEIGHT = 1080;

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
  if (finalizationInProgress) {
    postMessage({
      cmd: 'status',
      data: 'Finalizing output; cancellation is no longer available.'
    } satisfies WorkerResponseMessage);
    return;
  }
  cancelRequested = true;
}

function canSnapshotCanvas(canvas: OffscreenCanvas): boolean {
  try {
    const probe = new VideoFrame(canvas, { timestamp: 0, duration: 1 });
    probe.close();
    return true;
  } catch (error) {
    console.debug('Direct WebGPU canvas snapshots unavailable; using 2D mirror:', error);
    return false;
  }
}

/**
 * ONNX sessions and the shared preview canvases are not re-entrant. Keep model
 * switches and preview renders on one serial queue even though worker message
 * handlers themselves may overlap while awaiting async work.
 */
function enqueuePreviewOperation(operation: () => Promise<void>): Promise<void> {
  const result = previewOperationTail.then(operation, operation);
  previewOperationTail = result.catch(() => undefined);
  return result;
}

/** Coalesce rapid resolution changes to the latest target while preserving order. */
function queuePreviewRender(data: RenderPreviewData): Promise<void> {
  pendingPreviewRender = data;
  if (previewRenderScheduled) return previewOperationTail;

  previewRenderScheduled = true;
  return enqueuePreviewOperation(async () => {
    try {
      let renderSucceeded = true;
      while (pendingPreviewRender) {
        const nextRender = pendingPreviewRender;
        pendingPreviewRender = null;
        renderSucceeded = await rerenderPreview(nextRender);
        if (!renderSucceeded) {
          pendingPreviewRender = null;
          break;
        }
      }
      if (renderSucceeded) {
        postMessage({ cmd: 'modelLoaded' } satisfies WorkerResponseMessage);
      }
    } finally {
      previewRenderScheduled = false;
    }
  });
}

/** Coalesce rapid model choices while serializing them with preview renders. */
function queueModelSwitch(data: SwitchModelData): Promise<void> {
  pendingModelSwitch = data;
  if (modelSwitchScheduled) return previewOperationTail;

  modelSwitchScheduled = true;
  return enqueuePreviewOperation(async () => {
    try {
      const nextSwitch = pendingModelSwitch;
      pendingModelSwitch = null;
      if (nextSwitch) {
        await switchModel(nextSwitch);
      }
    } finally {
      modelSwitchScheduled = false;
    }
  });
}

/**
 * Check that the bundled ONNX Runtime fallback can initialize. WebGPU is
 * detected later as an optional acceleration path for the selected model.
 */
async function isSupported(): Promise<void> {
  // WebGPU is an acceleration path, not a startup requirement. Initializing
  // ORT here verifies the bundled WASM fallback used on non-WebGPU browsers.
  const supported = await Upscaler.initORT();

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
 * Re-render the retained preview frame for a new output cap without rebuilding
 * or reloading the current ONNX session.
 */
async function rerenderPreview(data: RenderPreviewData): Promise<boolean> {
  if (!upscaler || !upscaler.isReady()) {
    postMessage({
      cmd: 'error',
      data: 'Upscaler model is not ready for preview rendering'
    } satisfies WorkerResponseMessage);
    return false;
  }

  try {
    postMessage({ cmd: 'status', data: 'Rendering preview frame...' } satisfies WorkerResponseMessage);
    await renderPreviewFrame(data.bitmap, data.targetHeight);
    return true;
  } catch (error) {
    console.error('Failed to update preview resolution:', error);
    postMessage({
      cmd: 'error',
      data: `Failed to update preview: ${error}`
    } satisfies WorkerResponseMessage);
    return false;
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
      // BufferTarget otherwise defaults to in-memory fast start, retaining
      // encoded media packets until finalization. Explicitly write a normal MP4
      // progressively; StreamTarget already uses this behavior by default.
      return new Mp4OutputFormat({ fastStart: false });
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
 * Prefer the hardware video encoder on Macs when the browser advertises a
 * compatible configuration, but retain MediaBunny's normal software path as
 * a fallback.  Calling canEncodeVideo here avoids making an unsupported
 * `hardwareAcceleration: 'prefer-hardware'` configuration fail the whole job
 * during Output.start().
 */
async function getHardwareAcceleration(
  codec: 'avc' | 'vp9' | 'hevc' | 'av1' | 'vp8',
  width: number,
  height: number,
): Promise<'prefer-hardware' | 'no-preference'> {
  try {
    const supported = await canEncodeVideo(codec, {
      width,
      height,
      bitrate: QUALITY_HIGH,
      hardwareAcceleration: 'prefer-hardware',
    });

    if (supported) return 'prefer-hardware';
  } catch (error) {
    // Encoding support is still checked by MediaBunny when the first frame is
    // submitted.  Treat a probe failure as a normal software fallback rather
    // than making a platform-specific capability check fatal.
    console.debug('Hardware encoder probe failed; using browser default:', error);
  }

  return 'no-preference';
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
  // Current callers resolve Auto on the main thread. Keep a deterministic,
  // conservative fallback here so an omitted/legacy target never implies 8K.
  return resolveOutputResolution(
    resolution,
    currentScale,
    targetHeight,
    DEFAULT_AUTO_TARGET_HEIGHT,
  );
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
  return resolveInferenceResolution(resolution, currentScale, encodeResolution);
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
  let writable: WritableStream<StreamTargetChunk> | null = null;
  let fileWritable: FileSystemWritableFileStream | null = null;
  let fileTargetSettled = false;
  let commitFileTarget = false;
  let audioSource: AudioSampleSource | null = null;
  let audioPacketSource: EncodedAudioPacketSource | null = null;
  let audioPacketSourceClosed = false;
  let videoSource: CanvasSource | null = null;
  let output: Output | null = null;
  let outputFinalized = false;
  let wasCancelled = false;
  let usesDirectGpuEncodeCanvas = false;
  let videoIterator: AsyncIterator<VideoSample> | null = null;
  let audioSampleIterator: AsyncIterator<AudioSample> | null = null;
  let audioPacketIterator: AsyncIterator<EncodedPacket> | null = null;
  let pendingAudioSample: AudioSample | null = null;

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
      // Prefer the native demuxers for MP4, Matroska, and WebM. The main
      // thread probes MKV support and only invokes FFmpeg when the browser's
      // WebCodecs path cannot decode it.
      formats: [MP4, MATROSKA, WEBM],
      source
    });

    let target: BufferTarget | StreamTarget;
    if (outputHandle) {
      fileWritable = await outputHandle.createWritable();
      // MediaBunny locks the stream it receives. A small forwarding stream
      // keeps the underlying file handle abortable and makes Output.cancel()
      // discard the browser's temporary file instead of committing a partial
      // result through WritableStream.close().
      writable = new WritableStream<StreamTargetChunk>({
        write: chunk => fileWritable!.write(chunk),
        close: async () => {
          if (fileTargetSettled) return;
          if (commitFileTarget) {
            await fileWritable!.close();
          } else {
            await fileWritable!.abort();
          }
          fileTargetSettled = true;
        },
        abort: async reason => {
          if (fileTargetSettled) return;
          await fileWritable!.abort(reason);
          fileTargetSettled = true;
        },
      });
      // Coalesce writes into large chunks so the File System Access stream
      // does not receive one tiny write per muxer packet.
      target = new StreamTarget(writable, { chunked: true });
    } else {
      target = new BufferTarget();
    }

    const outputFormat = getOutputFormat(settings.outputFormat);
    output = new Output({
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
    // When inference already produces the requested encode dimensions, bind
    // CanvasSource directly to the renderer-owned WebGPU surface. This avoids
    // WebGPU -> public 2D canvas -> encoder copies on every frame.
    const candidateGpuEncodeCanvas = !needsEncodeResize
      ? upscaler.getGpuOutputCanvas(renderOutputWidth, renderOutputHeight)
      : null;
    const directGpuEncodeCanvas = candidateGpuEncodeCanvas && canSnapshotCanvas(candidateGpuEncodeCanvas)
      ? candidateGpuEncodeCanvas
      : null;
    if (candidateGpuEncodeCanvas && !directGpuEncodeCanvas) {
      upscaler.releaseGpuOutputCanvas();
    }
    usesDirectGpuEncodeCanvas = directGpuEncodeCanvas !== null;
    const encodeCanvas = directGpuEncodeCanvas ?? (
      needsEncodeResize
        ? new OffscreenCanvas(encodeResolution.width, encodeResolution.height)
        : upscaled_canvas
    );
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

    const codec = getCodec(settings.outputFormat);
    const hardwareAcceleration = await getHardwareAcceleration(
      codec,
      encodeResolution.width,
      encodeResolution.height,
    );
    let encoderConfig: VideoEncoderConfig | undefined;
    let lastPipelineKey = '';
    const getRenderPath = (): PipelineTelemetry['renderPath'] => {
      if (usesDirectGpuEncodeCanvas) return 'webgpu-direct';
      if (upscaler?.isUsingGPUPath()) {
        return needsEncodeResize ? 'webgpu-2d-resize' : 'webgpu-2d-mirror';
      }
      return needsEncodeResize ? 'cpu-tensor-2d-resize' : 'cpu-tensor-direct-2d';
    };
    const reportPipeline = (force = false): void => {
      const executionProvider = upscaler?.getExecutionProvider() ?? 'wasm';
      const renderPath = getRenderPath();
      const pipelineKey = `${executionProvider}:${renderPath}:${encoderConfig ? 'configured' : 'pending'}`;
      if (!force && pipelineKey === lastPipelineKey) return;
      lastPipelineKey = pipelineKey;
      postMessage({
        cmd: 'pipeline',
        data: {
          executionProvider,
          renderPath,
          ...(encoderConfig ? { encoderConfig } : {}),
        },
      } satisfies WorkerResponseMessage);
    };

    videoSource = new CanvasSource(encodeCanvas, {
      codec,
      bitrate: QUALITY_HIGH,
      keyFrameInterval: 60,
      hardwareAcceleration,
      onEncoderConfig: config => {
        encoderConfig = { ...config };
        // Forward every encoder reconfiguration, not only the first one.
        reportPipeline(true);
      },
    });
    reportPipeline();

    // Preserve the source sample timestamps/durations instead of snapping all
    // output to a hard-coded 30 fps.  This avoids silently dropping temporal
    // detail from 60 fps input and lets MediaBunny derive the encoder cadence
    // from the frames we actually submit.
    output.addVideoTrack(videoSource);

    // Set up audio passthrough
    const audioTrack = await input.getPrimaryAudioTrack();
    let audioSink: AudioSampleSink | null = null;
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
        // Keep one decoder-backed iterator alive for the whole job. Creating
        // a new samples() generator for every video frame repeatedly seeks or
        // reinitializes the audio decoder on incompatible-container jobs.
        audioSampleIterator = audioSink.samples(0, Number.POSITIVE_INFINITY)[Symbol.asyncIterator]();
        console.log(`Re-encoding audio to ${audioCodec}`);
      }
    }

    await output.start();

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
    let lastFrameCompletedAt = start_time;
    let lastProgress = -1;
    let lastEta = '';
    let lastEtaAt = -Infinity;

    // Track audio progress separately
    const timingTotals: FrameTiming = {
      decodeMs: 0,
      decodeWaitMs: 0,
      frameConversionMs: 0,
      audioMs: 0,
      preprocessMs: 0,
      inferenceMs: 0,
      postprocessMs: 0,
      gpuWaitMs: 0,
      gpuTimestampMs: 0,
      canvasMs: 0,
      encodeMs: 0,
      finalizeMs: 0,
      wallFps: 0,
      pipelineFps: 0,
      totalMs: 0,
      tileCount: 0,
      inputPixels: 0,
      inferredPixels: 0,
      frames: 0,
    };

    function buildTimingReport(finalizeMs = 0, pipelineComplete = false): FrameTiming {
      const frames = timingTotals.frames;
      const frameLoopSeconds = Math.max((lastFrameCompletedAt - start_time) / 1000, 0.001);
      const pipelineSeconds = Math.max((performance.now() - start_time) / 1000, 0.001);
      return {
        ...timingTotals,
        decodeMs: timingTotals.decodeMs / frames,
        decodeWaitMs: timingTotals.decodeWaitMs / frames,
        frameConversionMs: timingTotals.frameConversionMs / frames,
        audioMs: timingTotals.audioMs / frames,
        preprocessMs: timingTotals.preprocessMs / frames,
        inferenceMs: timingTotals.inferenceMs / frames,
        postprocessMs: timingTotals.postprocessMs / frames,
        gpuWaitMs: timingTotals.gpuWaitMs / frames,
        gpuTimestampMs: timingTotals.gpuTimestampMs / frames,
        canvasMs: timingTotals.canvasMs / frames,
        encodeMs: timingTotals.encodeMs / frames,
        finalizeMs,
        wallFps: frames / frameLoopSeconds,
        pipelineFps: pipelineComplete ? frames / pipelineSeconds : 0,
        totalMs: timingTotals.totalMs / frames,
        tileCount: timingTotals.tileCount / frames,
        inputPixels: timingTotals.inputPixels / frames,
        inferredPixels: timingTotals.inferredPixels / frames,
      };
    }

    function reportTiming(frameTiming: FrameTiming): void {
      for (const key of Object.keys(timingTotals) as Array<keyof FrameTiming>) {
        timingTotals[key] += frameTiming[key];
      }

      if (timingTotals.frames % 30 === 0) {
        postMessage({
          cmd: 'timing',
          data: buildTimingReport(),
        } satisfies WorkerResponseMessage);
      }
    }

    function reportProgress(sample: VideoSample) {
      const time_elapsed = performance.now() - start_time;
      const completedTimestamp = Math.min(duration, sample.timestamp + sample.duration);
      const percentComplete = duration > 0 ? (completedTimestamp / duration) * 100 : 100;
      const progress = Math.min(99, Math.floor(percentComplete));

      if (progress !== lastProgress) {
        postMessage({ cmd: 'progress', data: progress });
        lastProgress = progress;
      }

      let etaMessage: string;
      if (time_elapsed > 1000 && percentComplete > 0) {
        const processing_rate = percentComplete / time_elapsed;
        const eta = Math.max(0, Math.round(((100 - percentComplete) / processing_rate) / 1000));
        etaMessage = prettyTime(eta);
      } else {
        etaMessage = 'calculating...';
      }

      // ETA is informational and can be updated at a much lower cadence than
      // frame processing. Avoid posting one message for every decoded frame,
      // especially on high-FPS sources.
      const etaNow = performance.now();
      if (etaMessage !== lastEta || etaNow - lastEtaAt >= 250) {
        postMessage({ cmd: 'eta', data: etaMessage });
        lastEta = etaMessage;
        lastEtaAt = etaNow;
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

      // Consume the persistent audio iterator up to this video frame's timestamp.
      while (audioSampleIterator) {
        throwIfCancelled();
        if (!pendingAudioSample) {
          const nextAudio = await audioSampleIterator.next();
          if (nextAudio.done) {
            audioSampleIterator = null;
            break;
          }
          pendingAudioSample = nextAudio.value;
        }

        if (pendingAudioSample.timestamp >= timestamp) return;

        const audioSample = pendingAudioSample;
        pendingAudioSample = null;
        try {
          await audioSource.add(audioSample);
        } finally {
          audioSample.close();
        }
      }
    }

    // Loop over all frames. Measure the async iterator wait itself: this is
    // where MediaBunny performs demux/decode work. Timing only toVideoFrame()
    // substantially underreported decoding cost.
    postMessage({ cmd: 'progress', data: 0 });
    videoIterator = videoSink.samples()[Symbol.asyncIterator]();
    while (true) {
      const decodeWaitStarted = performance.now();
      const nextVideo = await videoIterator.next();
      const decodeWaitMs = performance.now() - decodeWaitStarted;
      if (nextVideo.done || !nextVideo.value) break;

      const sample = nextVideo.value;
      let videoFrame: VideoFrame | null = null;
      const frameStarted = decodeWaitStarted;
      try {
        throwIfCancelled();

        // Process audio up to this frame's timestamp
        const audioStarted = performance.now();
        await processAudioUpTo(sample.timestamp + sample.duration);
        const audioMs = performance.now() - audioStarted;

        const conversionStarted = performance.now();
        videoFrame = sample.toVideoFrame();
        const frameConversionMs = performance.now() - conversionStarted;
        const decodeMs = decodeWaitMs + frameConversionMs;
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
          await upscaler.render(inferenceCanvas, {
            mirrorOutput: !usesDirectGpuEncodeCanvas,
            requireGpuOutput: usesDirectGpuEncodeCanvas,
          });
        } else {
          await upscaler.render(videoFrame, {
            mirrorOutput: !usesDirectGpuEncodeCanvas,
            requireGpuOutput: usesDirectGpuEncodeCanvas,
          });
        }
        throwIfCancelled();
        reportPipeline();

        const encodeStarted = performance.now();
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
        lastFrameCompletedAt = performance.now();

        const upscaleTiming = upscaler.getLastTiming();
        if (upscaleTiming) {
          reportTiming({
            decodeMs,
            decodeWaitMs,
            frameConversionMs,
            audioMs,
            preprocessMs: upscaleTiming.preprocessMs,
            inferenceMs: upscaleTiming.inferenceMs,
            postprocessMs: upscaleTiming.postprocessMs,
            gpuWaitMs: upscaleTiming.gpuWaitMs,
            gpuTimestampMs: upscaleTiming.gpuTimestampMs,
            canvasMs: upscaleTiming.canvasMs,
            encodeMs: performance.now() - encodeStarted,
            finalizeMs: 0,
            wallFps: 0,
            pipelineFps: 0,
            totalMs: performance.now() - frameStarted,
            tileCount: upscaleTiming.tileCount,
            inputPixels: upscaleTiming.inputPixels,
            inferredPixels: upscaleTiming.inferredPixels,
            frames: 1,
          });
        }

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

    const finalizeStarted = performance.now();
    finalizationInProgress = true;
    postMessage({ cmd: 'status', data: 'Finalizing output...' } satisfies WorkerResponseMessage);
    commitFileTarget = true;
    await output.finalize();
    const finalizeMs = performance.now() - finalizeStarted;
    outputFinalized = true;
    postMessage({ cmd: 'progress', data: 100 });
    postMessage({ cmd: 'eta', data: prettyTime(0) });

    if (timingTotals.frames > 0) {
      postMessage({
        cmd: 'timing',
        data: buildTimingReport(finalizeMs, true),
      } satisfies WorkerResponseMessage);
    }

    if (writable) {
      postMessage({ cmd: 'finished', data: null }, []);
    } else {
      const buffer = (output.target as BufferTarget).buffer;
      postMessage({ cmd: 'finished', data: buffer }, [buffer]);
    }
  } catch (e) {
    commitFileTarget = false;
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
    // Always release encoder/source resources on both cancellation and errors.
    // Previously this branch only ran for explicit cancellation, leaving a
    // failed encode (and, for file-backed output, its writable stream) alive
    // until the worker was torn down.
    if (!outputFinalized) {
      commitFileTarget = false;
      // Let MediaBunny force-close encoders before closing its target writer.
      // The forwarding stream converts that close into an underlying file
      // abort while commitFileTarget is false.
      if (output && output.state !== 'canceled' && output.state !== 'finalized') {
        try {
          await output.cancel();
        } catch {
          // Ignore output cancellation errors while unwinding.
        }
      }

      if (fileWritable && !fileTargetSettled) {
        try {
          await fileWritable.abort();
          fileTargetSettled = true;
        } catch {
          // Ignore file abort errors after cancellation/error.
        }
      }

      // MediaBunny currently leaves an Output in its non-cancelable
      // `finalizing` state if finalize() itself rejects. File-backed output is
      // still safely aborted above; discard any partial in-memory target here
      // so a failed mux does not retain a potentially large ArrayBuffer until
      // the dependency-owned Output is garbage-collected.
      if (output?.target instanceof BufferTarget) {
        output.target.buffer = null;
      }
    }

    cancelRequested = false;
    await Promise.allSettled([
      (async () => videoIterator?.return?.())(),
      (async () => audioSampleIterator?.return?.())(),
      (async () => audioPacketIterator?.return?.())(),
    ]);
    finalizationInProgress = false;
    videoIterator = null;
    audioSampleIterator = null;
    audioPacketIterator = null;
    pendingAudioSample?.close();
    pendingAudioSample = null;
    upscaler?.releaseGpuOutputCanvas();
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

    case 'switchModel': {
      const switchData = event.data.data;
      await queueModelSwitch(switchData);
      break;
    }

    case 'renderPreview':
      await queuePreviewRender(event.data.data);
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
