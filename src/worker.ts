/**
 * Video processing worker using Real-ESRGAN for upscaling.
 *
 * This worker handles video frame extraction, AI upscaling with
 * Real-ESRGAN (realesr-animevideov3), and video encoding.
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
} from 'mediabunny';

import { RealESRGAN } from './realesrgan';

import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
  InitData,
  Resolution
} from './types/worker-messages';

// Worker state
let upscaler: RealESRGAN | null = null;
let upscaled_canvas: OffscreenCanvas;
let original_canvas: OffscreenCanvas;
let resolution: Resolution;
let ctx: ImageBitmapRenderingContext | null;
let scale: number = 4; // Real-ESRGAN animevideov3 is 4x by default

/**
 * Check if WebGPU is supported in this environment.
 */
async function isSupported(): Promise<void> {
  const supported = await RealESRGAN.isWebGPUSupported();

  postMessage({
    cmd: 'isSupported',
    data: supported
  } satisfies WorkerResponseMessage);
}

/**
 * Initialize the worker with canvases and create Real-ESRGAN instance.
 */
async function init(config: InitData): Promise<void> {
  try {
    // Store canvases
    upscaled_canvas = config.upscaled;
    original_canvas = config.original;
    resolution = config.resolution;

    // Get the scale from config or use default
    scale = config.modelConfig?.scale || 4;

    // Set up output canvas dimensions
    upscaled_canvas.width = resolution.width * scale;
    upscaled_canvas.height = resolution.height * scale;

    // Set up original canvas context for "before" preview
    ctx = original_canvas.getContext('bitmaprenderer');

    // Create Real-ESRGAN upscaler
    upscaler = new RealESRGAN({
      modelPath: config.modelConfig?.modelPath || '/models/realesr-animevideov3.onnx',
      scale: scale,
      tileSize: config.modelConfig?.tileSize || 256,
      tilePadding: config.modelConfig?.tilePadding || 16,
    });

    // Initialize the model
    postMessage({ cmd: 'modelLoading', data: 0 } satisfies WorkerResponseMessage);

    await upscaler.init(upscaled_canvas);

    postMessage({ cmd: 'modelLoaded' } satisfies WorkerResponseMessage);

    // Render preview frame
    const bitmap = await createImageBitmap(config.bitmap, {
      resizeHeight: resolution.height * scale,
      resizeWidth: resolution.width * scale,
    });

    // Render the upscaled preview
    await upscaler.render(config.bitmap);

    // Render the "before" preview (bilinear upscale)
    if (ctx) {
      ctx.transferFromImageBitmap(bitmap);
    }

    bitmap.close();
  } catch (e) {
    console.error('Failed to initialize Real-ESRGAN:', e);
    postMessage({
      cmd: 'error',
      data: `Failed to initialize AI upscaler: ${e}`
    } satisfies WorkerResponseMessage);
  }
}

/**
 * Main video processing function using MediaBunny.
 */
async function initRecording(
  inputHandle: FileSystemFileHandle,
  outputHandle?: FileSystemFileHandle
): Promise<void> {
  if (!upscaler || !upscaler.isReady()) {
    postMessage({
      cmd: 'error',
      data: 'Real-ESRGAN model not loaded'
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
      format: new Mp4OutputFormat(),
      target: target,
    });

    const videoSource = new CanvasSource(upscaled_canvas, {
      codec: 'avc',
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
        resizeHeight: videoFrame.codedHeight * scale,
        resizeWidth: videoFrame.codedWidth * scale
      });

      // Render through Real-ESRGAN
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

    case 'process':
      await initRecording(event.data.inputHandle, event.data.outputHandle);
      break;
  }
};
