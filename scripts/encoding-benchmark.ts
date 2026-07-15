import {
  BlobSource,
  BufferTarget,
  CanvasSource,
  Input,
  MP4,
  Mp4OutputFormat,
  Output,
  QUALITY_HIGH,
  VideoSampleSink,
  canEncodeVideo,
} from 'mediabunny';
import { Upscaler } from '../src/upscaler';
import type { ModelType } from '../src/types/worker-messages';
import shortClipData from '../test-clips/BotsMaster-15sec.mp4';
import longClipData from '../test-clips/BotsMaster-1min21sec.mp4';

const output = document.createElement('pre');
document.body.append(output);

const CLIPS = {
  short: shortClipData,
  long: longClipData,
} as const;

const DEFAULT_MODEL: ModelType = 'realesr-animevideov3';
const DEFAULT_TARGET_HEIGHTS = [480, 720, 1080];

interface Resolution {
  width: number;
  height: number;
}

interface FrameStats {
  decodeMs: number;
  resizeMs: number;
  renderMs: number;
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  gpuWaitMs: number;
  gpuTimestampMs: number;
  canvasMs: number;
  encodeMs: number;
  totalMs: number;
  tileCount: number;
  inputPixels: number;
  inferredPixels: number;
}

function write(label: string, value: unknown): void {
  output.textContent += `${label} ${typeof value === 'string' ? value : JSON.stringify(value)}\n`;
}

function dataUrlToBlob(dataUrl: string): Blob {
  const comma = dataUrl.indexOf(',');
  if (!dataUrl.startsWith('data:') || comma < 0) {
    throw new Error('Encoding benchmark clip asset is not an inline data URL');
  }

  const metadata = dataUrl.slice(5, comma);
  const base64 = dataUrl.slice(comma + 1);
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) {
    bytes[index] = binary.charCodeAt(index);
  }

  return new Blob([bytes], { type: metadata.split(';')[0] || 'video/mp4' });
}

function even(value: number): number {
  const rounded = Math.max(2, Math.round(value));
  return rounded % 2 === 0 ? rounded : rounded - 1;
}

function getOutputResolution(source: Resolution, scale: number, targetHeight: number): Resolution {
  const nativeHeight = source.height * scale;
  const nativeWidth = source.width * scale;
  if (nativeHeight <= targetHeight) {
    return { width: even(nativeWidth), height: even(nativeHeight) };
  }

  return {
    width: even((nativeWidth / nativeHeight) * targetHeight),
    height: even(targetHeight),
  };
}

function getTargetHeights(): number[] {
  const raw = new URLSearchParams(location.search).get('targets');
  if (!raw) return DEFAULT_TARGET_HEIGHTS;

  const values = raw
    .split(',')
    .map(value => Number(value.trim()))
    .filter(value => Number.isFinite(value) && value >= 2)
    .map(value => even(value));

  return values.length > 0 ? [...new Set(values)] : DEFAULT_TARGET_HEIGHTS;
}

function getModelScale(modelId: ModelType): number {
  if (modelId.includes('2x') || modelId.startsWith('animejanai')) return 2;
  if (
    modelId.startsWith('realplksr') ||
    modelId.startsWith('scunet') ||
    modelId.startsWith('swinir')
  ) {
    return 1;
  }
  return 4;
}

function getFrameLimit(): number | null {
  const raw = new URLSearchParams(location.search).get('frames');
  if (!raw || raw === 'all') return null;

  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 30;
}

function getTileSize(scale: number): number {
  const raw = new URLSearchParams(location.search).get('tileSize');
  if (raw) {
    const value = Number(raw);
    if (Number.isFinite(value) && value >= 32) return Math.floor(value);
  }
  return scale === 1 ? 256 : 512;
}

function addStats(total: FrameStats, frame: FrameStats): void {
  total.decodeMs += frame.decodeMs;
  total.resizeMs += frame.resizeMs;
  total.renderMs += frame.renderMs;
  total.preprocessMs += frame.preprocessMs;
  total.inferenceMs += frame.inferenceMs;
  total.postprocessMs += frame.postprocessMs;
  total.gpuWaitMs += frame.gpuWaitMs;
  total.gpuTimestampMs += frame.gpuTimestampMs;
  total.canvasMs += frame.canvasMs;
  total.encodeMs += frame.encodeMs;
  total.totalMs += frame.totalMs;
  total.tileCount += frame.tileCount;
  total.inputPixels += frame.inputPixels;
  total.inferredPixels += frame.inferredPixels;
}

function averageStats(total: FrameStats, frames: number): FrameStats {
  const average = { ...total };
  for (const key of Object.keys(average) as Array<keyof FrameStats>) {
    average[key] /= frames;
  }
  return average;
}

async function runCase(
  clipName: keyof typeof CLIPS,
  targetHeight: number,
  modelId: ModelType,
  frameLimit: number | null,
): Promise<void> {
  const input = new Input({
    formats: [MP4],
    source: new BlobSource(dataUrlToBlob(CLIPS[clipName])),
  });
  const track = await input.getPrimaryVideoTrack();
  if (!track) {
    input.dispose();
    throw new Error(`${clipName} has no video track`);
  }
  if (!(await track.canDecode())) {
    input.dispose();
    throw new Error(`${clipName} cannot be decoded in this browser`);
  }

  const sourceResolution = { width: track.displayWidth, height: track.displayHeight };
  const scale = getModelScale(modelId);
  const outputResolution = getOutputResolution(sourceResolution, scale, targetHeight);
  const tileSize = getTileSize(scale);
  const gpuTiming = new URLSearchParams(location.search).get('gpuTiming') !== '0';
  const inferenceResolution = {
    width: Math.min(sourceResolution.width, even(outputResolution.width / scale)),
    height: Math.min(sourceResolution.height, even(outputResolution.height / scale)),
  };

  const outputCanvas = new OffscreenCanvas(outputResolution.width, outputResolution.height);
  const outputContext = outputCanvas.getContext('2d', { alpha: false });
  if (!outputContext) throw new Error('Benchmark output canvas unavailable');

  const upscaler = new Upscaler({
    modelId,
    scale,
    tileSize,
    tilePadding: 32,
    enableGpuTimestamps: gpuTiming,
  });
  await upscaler.init(outputCanvas, (progress, message) => write('model', `${progress}% ${message}`));

  const codec = 'avc' as const;
  const hardwareAcceleration = await canEncodeVideo(codec, {
    width: outputResolution.width,
    height: outputResolution.height,
    bitrate: QUALITY_HIGH,
    hardwareAcceleration: 'prefer-hardware',
  }).then(supported => supported ? 'prefer-hardware' as const : 'no-preference' as const)
    .catch(() => 'no-preference' as const);

  const encodeCanvas = new OffscreenCanvas(outputResolution.width, outputResolution.height);
  const encodeContext = encodeCanvas.getContext('2d', { alpha: false });
  if (!encodeContext) throw new Error('Benchmark encode canvas unavailable');

  const videoSource = new CanvasSource(encodeCanvas, {
    codec,
    bitrate: QUALITY_HIGH,
    keyFrameInterval: 60,
    hardwareAcceleration,
  });
  const target = new BufferTarget();
  const encoded = new Output({
    format: new Mp4OutputFormat(),
    target,
  });
  encoded.addVideoTrack(videoSource);
  await encoded.start();

  const total: FrameStats = {
    decodeMs: 0,
    resizeMs: 0,
    renderMs: 0,
    preprocessMs: 0,
    inferenceMs: 0,
    postprocessMs: 0,
    gpuWaitMs: 0,
    gpuTimestampMs: 0,
    canvasMs: 0,
    encodeMs: 0,
    totalMs: 0,
    tileCount: 0,
    inputPixels: 0,
    inferredPixels: 0,
  };
  const sink = new VideoSampleSink(track);
  let frames = 0;
  let checksum = 0;
  let provider: 'webgpu' | 'wasm' = 'wasm';
  let gpuRenderer = false;
  const started = performance.now();

  try {
    for await (const sample of sink.samples()) {
      const frameStarted = performance.now();
      const decodeStarted = performance.now();
      const frame = sample.toVideoFrame();
      const decodeMs = performance.now() - decodeStarted;
      let inferenceSource: ImageBitmap | VideoFrame = frame;
      let resized: ImageBitmap | null = null;

      try {
        const resizeStarted = performance.now();
        if (inferenceResolution.width !== sourceResolution.width || inferenceResolution.height !== sourceResolution.height) {
          resized = await createImageBitmap(frame, {
            resizeWidth: inferenceResolution.width,
            resizeHeight: inferenceResolution.height,
          });
          inferenceSource = resized;
        }
        const resizeMs = performance.now() - resizeStarted;

        const renderStarted = performance.now();
        await upscaler.render(inferenceSource);
        const renderMs = performance.now() - renderStarted;
        const timing = upscaler.getLastTiming();

        const encodeStarted = performance.now();
        encodeContext.clearRect(0, 0, outputResolution.width, outputResolution.height);
        encodeContext.drawImage(outputCanvas, 0, 0, outputResolution.width, outputResolution.height);
        await videoSource.add(sample.timestamp, sample.duration);
        const encodeMs = performance.now() - encodeStarted;

        const probe = outputContext.getImageData(0, 0, 1, 1).data;
        checksum = (checksum + probe[0] + probe[1] * 3 + probe[2] * 7) >>> 0;
        addStats(total, {
          decodeMs,
          resizeMs,
          renderMs,
          preprocessMs: timing?.preprocessMs || 0,
          inferenceMs: timing?.inferenceMs || 0,
          postprocessMs: timing?.postprocessMs || 0,
          gpuWaitMs: timing?.gpuWaitMs || 0,
          gpuTimestampMs: timing?.gpuTimestampMs || 0,
          canvasMs: timing?.canvasMs || 0,
          encodeMs,
          totalMs: performance.now() - frameStarted,
          tileCount: timing?.tileCount || 0,
          inputPixels: timing?.inputPixels || 0,
          inferredPixels: timing?.inferredPixels || 0,
        });
      } finally {
        resized?.close();
        frame.close();
        sample.close();
      }

      frames++;
      if (frameLimit !== null && frames >= frameLimit) break;
    }

    videoSource.close();
    await encoded.finalize();
    provider = upscaler.getExecutionProvider();
    gpuRenderer = upscaler.isUsingGPUPath();
  } finally {
    await upscaler.dispose();
    input.dispose();
  }

  if (frames === 0 || checksum === 0) {
    throw new Error(`No usable output for ${clipName}/${targetHeight}p (frames=${frames}, checksum=${checksum})`);
  }

  write('case', {
    clip: clipName,
    sourceResolution,
    targetHeight,
    outputResolution,
    inferenceResolution,
    tileSize,
    gpuTiming,
    model: modelId,
    provider,
    gpuRenderer,
    hardwareAcceleration,
    frames,
    wallMs: performance.now() - started,
    outputBytes: target.buffer?.byteLength || 0,
    checksum,
    average: averageStats(total, frames),
  });
}

async function run(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const requestedClip = params.get('clip') as keyof typeof CLIPS | null;
  const clips = requestedClip && requestedClip in CLIPS ? [requestedClip] : (Object.keys(CLIPS) as Array<keyof typeof CLIPS>);
  const model = (params.get('model') || DEFAULT_MODEL) as ModelType;
  const frameLimit = getFrameLimit();

  write('benchmark', 'encoding');
  write('audio', 'video-only (intentional; isolates decode/upscale/encode)');
  write('targets', getTargetHeights());
  write('frames', frameLimit === null ? 'all' : frameLimit);

  for (const clip of clips) {
    for (const targetHeight of getTargetHeights()) {
      try {
        await runCase(clip, targetHeight, model, frameLimit);
      } catch (error) {
        write('caseError', {
          clip,
          targetHeight,
          error: error && error.stack ? error.stack : String(error),
        });
      }
    }
  }
}

run().catch(error => write('error', error && error.stack ? error.stack : String(error)));
