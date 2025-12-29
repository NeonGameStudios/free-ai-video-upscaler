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
    // Store canvases
    upscaled_canvas = config.upscaled;
    original_canvas = config.original;
    resolution = config.resolution;

    // Get the scale from config
    currentScale = config.modelConfig.scale;

    // Set up output canvas dimensions
    upscaled_canvas.width = resolution.width * currentScale;
    upscaled_canvas.height = resolution.height * currentScale;

    // Set up original canvas context for "before" preview
    ctx = original_canvas.getContext('bitmaprenderer');

    // Create upscaler
    upscaler = new Upscaler({
      modelPath: config.modelConfig.modelPath,
      scale: config.modelConfig.scale,
      tileSize: config.modelConfig.tileSize || 256,
      tilePadding: config.modelConfig.tilePadding || 16,
      denoiseLevel: config.modelConfig.denoiseLevel,
    });

    // Initialize the model
    postMessage({ cmd: 'modelLoading', data: 0 } satisfies WorkerResponseMessage);

    await upscaler.init(upscaled_canvas);

    postMessage({ cmd: 'modelLoaded' } satisfies WorkerResponseMessage);

    // Render preview frame
    const bitmap = await createImageBitmap(config.bitmap, {
      resizeHeight: resolution.height * currentScale,
      resizeWidth: resolution.width * currentScale,
    });

    // Render the upscaled preview
    await upscaler.render(config.bitmap);

    // Render the "before" preview (bilinear upscale)
    if (ctx) {
      ctx.transferFromImageBitmap(bitmap);
    }

    bitmap.close();
  } catch (e) {
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

  try {
    postMessage({ cmd: 'modelLoading', data: 0 } satisfies WorkerResponseMessage);

    // Update scale
    currentScale = data.modelConfig.scale;

    // Update canvas dimensions for new scale
    upscaled_canvas.width = resolution.width * currentScale;
    upscaled_canvas.height = resolution.height * currentScale;
    original_canvas.width = resolution.width * currentScale;
    original_canvas.height = resolution.height * currentScale;

    // Switch model
    await upscaler.switchModel({
      modelPath: data.modelConfig.modelPath,
      scale: data.modelConfig.scale,
      tileSize: data.modelConfig.tileSize,
      tilePadding: data.modelConfig.tilePadding,
      denoiseLevel: data.modelConfig.denoiseLevel,
    });

    postMessage({ cmd: 'modelLoaded' } satisfies WorkerResponseMessage);

    // Render preview with new model
    const bitmap = await createImageBitmap(data.bitmap, {
      resizeHeight: resolution.height * currentScale,
      resizeWidth: resolution.width * currentScale,
    });

    await upscaler.render(data.bitmap);

    if (ctx) {
      ctx.transferFromImageBitmap(bitmap);
    }

    bitmap.close();
  } catch (e) {
    console.error('Failed to switch model:', e);
    postMessage({
      cmd: 'error',
      data: `Failed to switch model: ${e}`
    } satisfies WorkerResponseMessage);
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
function getCodec(format: OutputFormat): string {
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
 * Main video processing function.
 */
async function initRecording(
  inputHandle: FileSystemFileHandle,
  outputHandle: FileSystemFileHandle | undefined,
  settings: ProcessSettings
): Promise<void> {
  if (!upscaler || !upscaler.isReady()) {
    postMessage({
      cmd: 'error',
      data: 'Upscaler model not loaded'
    } satisfies WorkerResponseMessage);
    return;
  }

  try {
    // Get the file from the handle
    const file = await inputHandle.getFile();

    // MediaBunny handles streaming from the blob for large files
    const source = new BlobSource(file);

    const input = new Input({
      formats: [MP4],
      source
    });

    let target: BufferTarget | StreamTarget;
    let writer: WritableStream | undefined;

    if (outputHandle) {
      writer = await outputHandle.createWritable();
      target = new StreamTarget(writer);
    } else {
      target = new BufferTarget();
    }

    const output = new Output({
      format: getOutputFormat(settings.outputFormat),
      target: target,
    });

    const videoSource = new CanvasSource(upscaled_canvas, {
      codec: getCodec(settings.outputFormat),
      bitrate: QUALITY_HIGH,
      keyFrameInterval: 60,
    });

    output.addVideoTrack(videoSource, { frameRate: 30 });
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

    const sink = new VideoSampleSink(videoTrack);
    const duration = await input.computeDuration();
    const start_time = performance.now();

    function reportProgress(sample: VideoSample) {
      const time_elapsed = performance.now() - start_time;
      const progress = Math.floor((sample.timestamp) / duration * 100);

      postMessage({ cmd: 'progress', data: progress });

      if (time_elapsed > 1000) {
        const processing_rate = ((sample.timestamp) / duration * 100) / time_elapsed;
        const eta = Math.round(((100 - progress) / processing_rate) / 1000);
        postMessage({ cmd: 'eta', data: prettyTime(eta) });
      } else {
        postMessage({ cmd: 'eta', data: 'calculating...' });
      }
    }

    // Loop over all frames
    for await (const sample of sink.samples()) {
      const videoFrame = sample.toVideoFrame();

      // Create "before" preview (bilinear upscale)
      const bitmap = await createImageBitmap(videoFrame, {
        resizeHeight: videoFrame.codedHeight * currentScale,
        resizeWidth: videoFrame.codedWidth * currentScale
      });

      // Render through upscaler
      await upscaler.render(videoFrame);

      // Render the "Before" preview
      if (ctx) {
        ctx.transferFromImageBitmap(bitmap);
      }

      // Add frame to output video
      videoSource.add(sample.timestamp, sample.duration);

      reportProgress(sample);

      // Cleanup
      videoFrame.close();
      sample.close();
    }

    await output.finalize();

    if (writer) {
      postMessage({ cmd: 'finished', data: null }, []);
    } else {
      const buffer = (output.target as BufferTarget).buffer;
      postMessage({ cmd: 'finished', data: buffer }, [buffer]);
    }
  } catch (e) {
    console.error('Video processing error:', e);
    postMessage({
      cmd: 'error',
      data: `Video processing failed: ${e}`
    } satisfies WorkerResponseMessage);
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
        event.data.outputHandle,
        event.data.settings
      );
      break;
  }
};
