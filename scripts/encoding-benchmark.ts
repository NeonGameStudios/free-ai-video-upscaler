import {
  BlobSource,
  BufferTarget,
  CanvasSource,
  Input,
  MP4,
  Mp4OutputFormat,
  Output,
  QUALITY_HIGH,
  type VideoSample,
  VideoSampleSink,
  canEncodeVideo,
} from 'mediabunny';
import { Upscaler } from '../src/upscaler';
import { calculateTilePlan } from '../src/tiling';
import {
  ceilToEven,
  resolveInferenceResolution,
  resolveOutputResolution,
  type ModelType,
} from '../src/types/worker-messages';
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
const DEFAULT_WARMUP_FRAMES = 3;

interface Resolution {
  width: number;
  height: number;
}

interface FrameStats {
  decodeWaitMs: number;
  frameWrapMs: number;
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

interface ValidationProbe {
  label: string;
  kind: 'border' | 'seam' | 'interior';
  x: number;
  y: number;
}

interface PixelSample extends ValidationProbe {
  rgba: [number, number, number, number];
}

interface ValidationSummary {
  checksum: number;
  sampleCount: number;
  uniqueColors: number;
  nonBlackSamples: number;
  transparentSamples: number;
  border: ValidationCategorySummary;
  seam: ValidationCategorySummary;
  interior: ValidationCategorySummary;
  seamDelta: {
    pairs: number;
    p50: number;
    p95: number;
    max: number;
  };
}

interface ValidationCategorySummary {
  samples: number;
  checksum: number;
  nonBlackSamples: number;
  transparentSamples: number;
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

function getOutputResolution(source: Resolution, scale: number, targetHeight: number): Resolution {
  return resolveOutputResolution(source, scale, targetHeight);
}

function getTargetHeights(): number[] {
  const raw = new URLSearchParams(location.search).get('targets');
  if (!raw) return DEFAULT_TARGET_HEIGHTS;

  const values = raw
    .split(',')
    .map(value => Number(value.trim()))
    .filter(value => Number.isFinite(value) && value >= 2)
    .map(value => ceilToEven(value));

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

function getWarmupFrames(): number {
  const raw = new URLSearchParams(location.search).get('warmup');
  if (raw === null) return DEFAULT_WARMUP_FRAMES;

  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : DEFAULT_WARMUP_FRAMES;
}

function getTileSize(scale: number): number {
  const raw = new URLSearchParams(location.search).get('tileSize');
  if (raw) {
    const value = Number(raw);
    if (Number.isFinite(value) && value >= 32) return Math.floor(value);
  }
  return scale === 1 ? 256 : 512;
}

function directGpuCanvasEnabled(): boolean {
  return new URLSearchParams(location.search).get('directGpu') !== '0';
}

function addStats(total: FrameStats, frame: FrameStats): void {
  total.decodeWaitMs += frame.decodeWaitMs;
  total.frameWrapMs += frame.frameWrapMs;
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

function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) return 0;

  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(Math.max(quantile, 0), 1) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) return sorted[lower];

  const weight = rank - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function summarizeEncoderConfig(config: VideoEncoderConfig | null) {
  if (!config) return null;

  return {
    codec: config.codec,
    width: config.width,
    height: config.height,
    bitrate: config.bitrate,
    framerate: config.framerate,
    bitrateMode: config.bitrateMode,
    latencyMode: config.latencyMode,
    hardwareAcceleration: config.hardwareAcceleration,
  };
}

function describeRenderPath(
  provider: 'webgpu' | 'wasm',
  gpuRenderer: boolean,
  directGpuCanvas: boolean,
  needsEncodeResize: boolean,
): string {
  if (directGpuCanvas) return 'webgpu-direct';
  if (gpuRenderer) return needsEncodeResize ? 'webgpu-2d-resize' : 'webgpu-2d-mirror';
  if (provider === 'webgpu') {
    return needsEncodeResize ? 'webgpu-cpu-tensor-2d-resize' : 'webgpu-cpu-tensor-2d';
  }
  return needsEncodeResize ? 'wasm-2d-resize' : 'wasm-2d';
}

function canSnapshotCanvas(canvas: OffscreenCanvas): boolean {
  try {
    const probe = new VideoFrame(canvas, { timestamp: 0, duration: 1 });
    probe.close();
    return true;
  } catch (error) {
    console.debug('Direct WebGPU benchmark snapshots unavailable; using the 2D path:', error);
    return false;
  }
}

function checksumSamples(samples: readonly PixelSample[]): number {
  let hash = 2166136261;
  for (const sample of samples) {
    for (const channel of sample.rgba) {
      hash ^= channel;
      hash = Math.imul(hash, 16777619) >>> 0;
    }
  }
  return hash >>> 0;
}

function summarizeValidationCategory(
  samples: readonly PixelSample[],
  kind: ValidationProbe['kind'],
): ValidationCategorySummary {
  const categorySamples = samples.filter(sample => sample.kind === kind);
  return {
    samples: categorySamples.length,
    checksum: checksumSamples(categorySamples),
    nonBlackSamples: categorySamples.filter(sample => sample.rgba[0] + sample.rgba[1] + sample.rgba[2] > 0).length,
    transparentSamples: categorySamples.filter(sample => sample.rgba[3] === 0).length,
  };
}

function validateOutputFrame(
  context: OffscreenCanvasRenderingContext2D,
  outputResolution: Resolution,
  inferenceResolution: Resolution,
  tileSize: number,
  tilePadding: number,
): ValidationSummary {
  const probes: ValidationProbe[] = [];
  const seamPairs: Array<[number, number]> = [];
  const clampX = (value: number) => Math.min(Math.max(Math.round(value), 0), outputResolution.width - 1);
  const clampY = (value: number) => Math.min(Math.max(Math.round(value), 0), outputResolution.height - 1);
  const addProbe = (
    label: string,
    kind: ValidationProbe['kind'],
    x: number,
    y: number,
  ): number => {
    probes.push({ label, kind, x: clampX(x), y: clampY(y) });
    return probes.length - 1;
  };

  const xFractions = [0.25, 0.5, 0.75];
  const yFractions = [0.25, 0.5, 0.75];
  for (const fraction of yFractions) {
    const y = fraction * (outputResolution.height - 1);
    addProbe(`left-${fraction}`, 'border', 0, y);
    addProbe(`right-${fraction}`, 'border', outputResolution.width - 1, y);
  }
  for (const fraction of xFractions) {
    const x = fraction * (outputResolution.width - 1);
    addProbe(`top-${fraction}`, 'border', x, 0);
    addProbe(`bottom-${fraction}`, 'border', x, outputResolution.height - 1);
  }
  addProbe('top-left', 'border', 0, 0);
  addProbe('top-right', 'border', outputResolution.width - 1, 0);
  addProbe('bottom-left', 'border', 0, outputResolution.height - 1);
  addProbe('bottom-right', 'border', outputResolution.width - 1, outputResolution.height - 1);
  addProbe('center', 'interior', outputResolution.width / 2, outputResolution.height / 2);

  const tilePlan = calculateTilePlan(
    inferenceResolution.width,
    inferenceResolution.height,
    tileSize,
    tilePadding,
  );
  const effectivePadding = Math.min(tilePadding, Math.floor(tilePlan.overlap / 2));

  for (let tileIndex = 1; tileIndex < tilePlan.x.count; tileIndex++) {
    const sourceX = Math.min(
      tileIndex * tilePlan.x.step,
      Math.max(0, inferenceResolution.width - tilePlan.x.tileSize),
    );
    const seamX = clampX(
      (sourceX + effectivePadding) * outputResolution.width / inferenceResolution.width,
    );
    for (const fraction of yFractions) {
      const y = fraction * (outputResolution.height - 1);
      const before = addProbe(`vertical-${tileIndex}-${fraction}-before`, 'seam', seamX - 1, y);
      const after = addProbe(`vertical-${tileIndex}-${fraction}-after`, 'seam', seamX, y);
      seamPairs.push([before, after]);
    }
  }

  for (let tileIndex = 1; tileIndex < tilePlan.y.count; tileIndex++) {
    const sourceY = Math.min(
      tileIndex * tilePlan.y.step,
      Math.max(0, inferenceResolution.height - tilePlan.y.tileSize),
    );
    const seamY = clampY(
      (sourceY + effectivePadding) * outputResolution.height / inferenceResolution.height,
    );
    for (const fraction of xFractions) {
      const x = fraction * (outputResolution.width - 1);
      const before = addProbe(`horizontal-${tileIndex}-${fraction}-before`, 'seam', x, seamY - 1);
      const after = addProbe(`horizontal-${tileIndex}-${fraction}-after`, 'seam', x, seamY);
      seamPairs.push([before, after]);
    }
  }

  const samples: PixelSample[] = probes.map(probe => {
    const pixel = context.getImageData(probe.x, probe.y, 1, 1).data;
    return {
      ...probe,
      rgba: [pixel[0], pixel[1], pixel[2], pixel[3]],
    };
  });
  const seamDeltas = seamPairs.map(([before, after]) => {
    const first = samples[before].rgba;
    const second = samples[after].rgba;
    return Math.abs(first[0] - second[0])
      + Math.abs(first[1] - second[1])
      + Math.abs(first[2] - second[2]);
  });
  const summary: ValidationSummary = {
    checksum: checksumSamples(samples),
    sampleCount: samples.length,
    uniqueColors: new Set(samples.map(sample => sample.rgba.join(','))).size,
    nonBlackSamples: samples.filter(sample => sample.rgba[0] + sample.rgba[1] + sample.rgba[2] > 0).length,
    transparentSamples: samples.filter(sample => sample.rgba[3] === 0).length,
    border: summarizeValidationCategory(samples, 'border'),
    seam: summarizeValidationCategory(samples, 'seam'),
    interior: summarizeValidationCategory(samples, 'interior'),
    seamDelta: {
      pairs: seamDeltas.length,
      p50: percentile(seamDeltas, 0.5),
      p95: percentile(seamDeltas, 0.95),
      max: seamDeltas.length > 0 ? Math.max(...seamDeltas) : 0,
    },
  };

  const failures: string[] = [];
  if (summary.nonBlackSamples === 0) failures.push('all validation probes are black');
  if (summary.uniqueColors < 2) failures.push('validation probes contain no color variation');
  if (summary.transparentSamples > 0) failures.push(`${summary.transparentSamples} probes are transparent`);
  if (summary.border.nonBlackSamples === 0) failures.push('all border probes are black');
  if (summary.seam.samples > 0 && summary.seam.nonBlackSamples === 0) failures.push('all seam probes are black');
  if (failures.length > 0) {
    throw new Error(`Output validation failed: ${failures.join('; ')}`);
  }

  return summary;
}

async function runCase(
  clipName: keyof typeof CLIPS,
  targetHeight: number,
  modelId: ModelType,
  frameLimit: number | null,
  warmupFrames: number,
): Promise<void> {
  const input = new Input({
    formats: [MP4],
    source: new BlobSource(dataUrlToBlob(CLIPS[clipName])),
  });
  let upscaler: Upscaler | null = null;
  let outputCanvas: OffscreenCanvas | null = null;
  let encodeCanvas: OffscreenCanvas | null = null;
  let validationCanvas: OffscreenCanvas | null = null;
  let videoSource: CanvasSource | null = null;
  let encoded: Output | null = null;
  let target: BufferTarget | null = null;
  let videoSourceClosed = false;
  let outputFinalized = false;
  let caseResult: Record<string, unknown> | null = null;

  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error(`${clipName} has no video track`);
    if (!(await track.canDecode())) throw new Error(`${clipName} cannot be decoded in this browser`);

    const sourceResolution = { width: track.displayWidth, height: track.displayHeight };
    const scale = getModelScale(modelId);
    const outputResolution = getOutputResolution(sourceResolution, scale, targetHeight);
    const tileSize = getTileSize(scale);
    const tilePadding = 32;
    const gpuTiming = new URLSearchParams(location.search).get('gpuTiming') !== '0';
    const useDirectGpuCanvas = directGpuCanvasEnabled();
    const inferenceResolution = resolveInferenceResolution(
      sourceResolution,
      scale,
      outputResolution,
    );
    const renderResolution = {
      width: inferenceResolution.width * scale,
      height: inferenceResolution.height * scale,
    };

    outputCanvas = new OffscreenCanvas(renderResolution.width, renderResolution.height);
    const outputContext = outputCanvas.getContext('2d', { alpha: false });
    if (!outputContext) throw new Error('Benchmark output canvas unavailable');

    upscaler = new Upscaler({
      modelId,
      scale,
      tileSize,
      tilePadding,
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

    const directEncodeCanvas = renderResolution.width === outputResolution.width
      && renderResolution.height === outputResolution.height;
    const candidateGpuEncodeCanvas = directEncodeCanvas && useDirectGpuCanvas
      ? upscaler.getGpuOutputCanvas(renderResolution.width, renderResolution.height)
      : null;
    const directGpuEncodeCanvas = candidateGpuEncodeCanvas
      && canSnapshotCanvas(candidateGpuEncodeCanvas)
      ? candidateGpuEncodeCanvas
      : null;
    if (candidateGpuEncodeCanvas && !directGpuEncodeCanvas) {
      upscaler.releaseGpuOutputCanvas();
    }
    encodeCanvas = directGpuEncodeCanvas ?? (
      directEncodeCanvas
        ? outputCanvas
        : new OffscreenCanvas(outputResolution.width, outputResolution.height)
    );
    const encodeContext = directEncodeCanvas
      ? null
      : encodeCanvas.getContext('2d', { alpha: false });
    if (!directEncodeCanvas && !encodeContext) {
      throw new Error('Benchmark encode canvas unavailable');
    }

    let encoderConfig: VideoEncoderConfig | null = null;
    videoSource = new CanvasSource(encodeCanvas, {
      codec,
      bitrate: QUALITY_HIGH,
      keyFrameInterval: 60,
      hardwareAcceleration,
      onEncoderConfig: config => {
        encoderConfig = { ...config };
      },
    });
    target = new BufferTarget();
    encoded = new Output({
      format: new Mp4OutputFormat({ fastStart: false }),
      target,
    });
    encoded.addVideoTrack(videoSource);
    await encoded.start();

    const total: FrameStats = {
      decodeWaitMs: 0,
      frameWrapMs: 0,
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
    const frameTotals: number[] = [];
    const decodeWaits: number[] = [];
    const sink = new VideoSampleSink(track);
    const iterator = sink.samples()[Symbol.asyncIterator]();
    let encodedFrames = 0;
    let warmupFramesProcessed = 0;
    let measuredFrames = 0;
    let measuredWallStarted: number | null = null;
    let measuredWallEnded: number | null = null;
    const processingStarted = performance.now();

    try {
      while (frameLimit === null || measuredFrames < frameLimit) {
        const shouldMeasure = encodedFrames >= warmupFrames;
        const frameStarted = performance.now();
        if (shouldMeasure && measuredWallStarted === null) {
          measuredWallStarted = frameStarted;
        }

        const next = await iterator.next();
        const decodeWaitMs = performance.now() - frameStarted;
        if (next.done) break;

        const sample = next.value as VideoSample;
        let frame: VideoFrame | null = null;
        let resized: ImageBitmap | null = null;
        try {
          const frameWrapStarted = performance.now();
          frame = sample.toVideoFrame();
          const frameWrapMs = performance.now() - frameWrapStarted;
          let inferenceSource: ImageBitmap | VideoFrame = frame;

          const resizeStarted = performance.now();
          if (
            inferenceResolution.width !== sourceResolution.width
            || inferenceResolution.height !== sourceResolution.height
          ) {
            resized = await createImageBitmap(frame, {
              resizeWidth: inferenceResolution.width,
              resizeHeight: inferenceResolution.height,
            });
            inferenceSource = resized;
          }
          const resizeMs = performance.now() - resizeStarted;

          const renderStarted = performance.now();
          await upscaler.render(inferenceSource, {
            mirrorOutput: directGpuEncodeCanvas === null,
            requireGpuOutput: directGpuEncodeCanvas !== null,
          });
          const renderMs = performance.now() - renderStarted;
          const timing = upscaler.getLastTiming();

          const encodeStarted = performance.now();
          if (encodeContext) {
            encodeContext.drawImage(
              outputCanvas,
              0,
              0,
              outputResolution.width,
              outputResolution.height,
            );
          }
          await videoSource.add(sample.timestamp, sample.duration);
          const encodeMs = performance.now() - encodeStarted;
          const totalMs = performance.now() - frameStarted;

          if (shouldMeasure) {
            addStats(total, {
              decodeWaitMs,
              frameWrapMs,
              resizeMs,
              renderMs,
              preprocessMs: timing?.preprocessMs || 0,
              inferenceMs: timing?.inferenceMs || 0,
              postprocessMs: timing?.postprocessMs || 0,
              gpuWaitMs: timing?.gpuWaitMs || 0,
              gpuTimestampMs: timing?.gpuTimestampMs || 0,
              canvasMs: timing?.canvasMs || 0,
              encodeMs,
              totalMs,
              tileCount: timing?.tileCount || 0,
              inputPixels: timing?.inputPixels || 0,
              inferredPixels: timing?.inferredPixels || 0,
            });
            frameTotals.push(totalMs);
            decodeWaits.push(decodeWaitMs);
            measuredFrames++;
            measuredWallEnded = performance.now();
          } else {
            warmupFramesProcessed++;
          }
          encodedFrames++;
        } finally {
          resized?.close();
          frame?.close();
          sample.close();
        }
      }
    } finally {
      await iterator.return?.();
    }

    const frameProcessingWallMs = performance.now() - processingStarted;
    if (measuredFrames === 0 || measuredWallStarted === null || measuredWallEnded === null) {
      throw new Error(
        `No measured frames for ${clipName}/${targetHeight}p after ${warmupFramesProcessed} warmup frames`,
      );
    }

    const measuredWallMs = measuredWallEnded - measuredWallStarted;
    const provider = upscaler.getExecutionProvider();
    const gpuRenderer = upscaler.isUsingGPUPath();
    validationCanvas = new OffscreenCanvas(outputResolution.width, outputResolution.height);
    const validationContext = validationCanvas.getContext('2d', {
      alpha: false,
      willReadFrequently: true,
    });
    if (!validationContext) throw new Error('Benchmark validation canvas unavailable');
    const validationStarted = performance.now();
    validationContext.drawImage(
      encodeCanvas,
      0,
      0,
      outputResolution.width,
      outputResolution.height,
    );
    const validation = validateOutputFrame(
      validationContext,
      outputResolution,
      inferenceResolution,
      tileSize,
      tilePadding,
    );
    const validationMs = performance.now() - validationStarted;

    const sourceCloseStarted = performance.now();
    videoSource.close();
    videoSourceClosed = true;
    const sourceCloseCallMs = performance.now() - sourceCloseStarted;
    const finalizeStarted = performance.now();
    await encoded.finalize();
    const finalizeMs = performance.now() - finalizeStarted;
    outputFinalized = true;
    const caseWallMs = performance.now() - processingStarted;
    const pipelineWallMs = frameProcessingWallMs + sourceCloseCallMs + finalizeMs;
    const outputBytes = target.buffer?.byteLength || 0;

    caseResult = {
      clip: clipName,
      sourceResolution,
      targetHeight,
      outputResolution,
      inferenceResolution,
      renderResolution,
      tileSize,
      tilePadding,
      gpuTiming,
      directGpuCanvasRequested: useDirectGpuCanvas,
      model: modelId,
      provider,
      gpuRenderer,
      renderPath: describeRenderPath(
        provider,
        gpuRenderer,
        directGpuEncodeCanvas !== null,
        !directEncodeCanvas,
      ),
      inferenceResizePath: inferenceResolution.width === sourceResolution.width
        && inferenceResolution.height === sourceResolution.height
        ? 'none'
        : 'create-image-bitmap',
      encodeCanvasPath: directGpuEncodeCanvas
        ? 'direct-webgpu-canvas'
        : directEncodeCanvas
          ? 'direct-output-2d-canvas'
          : '2d-resize-copy',
      hardwareAcceleration,
      encoderConfig: summarizeEncoderConfig(encoderConfig),
      warmupFramesRequested: warmupFrames,
      warmupFramesProcessed,
      frames: measuredFrames,
      encodedFrames,
      measuredWallMs,
      wallFps: measuredFrames / (measuredWallMs / 1000),
      frameProcessingWallMs,
      pipelineWallMs,
      pipelineFpsIncludingFinalize: encodedFrames / (pipelineWallMs / 1000),
      caseWallMs,
      caseWallFps: encodedFrames / (caseWallMs / 1000),
      validationMs,
      output: {
        bytes: outputBytes,
        bytesPerEncodedFrame: outputBytes / encodedFrames,
        sourceCloseCallMs,
        finalizeMs,
      },
      frameTotalMs: {
        p50: percentile(frameTotals, 0.5),
        p95: percentile(frameTotals, 0.95),
        max: Math.max(...frameTotals),
      },
      decodeWaitMs: {
        p50: percentile(decodeWaits, 0.5),
        p95: percentile(decodeWaits, 0.95),
        max: Math.max(...decodeWaits),
      },
      validation,
      average: averageStats(total, measuredFrames),
    };
  } finally {
    if (videoSource && !videoSourceClosed) {
      try {
        videoSource.close();
      } catch {
        // Ignore cleanup errors after a failed benchmark case.
      }
    }
    if (encoded && !outputFinalized && encoded.state !== 'canceled' && encoded.state !== 'finalized') {
      try {
        await encoded.cancel();
      } catch {
        // Ignore cleanup errors after a failed benchmark case.
      }
    }
    try {
      await upscaler?.dispose();
    } finally {
      input.dispose();
      if (encodeCanvas && encodeCanvas !== outputCanvas) {
        encodeCanvas.width = 0;
        encodeCanvas.height = 0;
      }
      if (validationCanvas) {
        validationCanvas.width = 0;
        validationCanvas.height = 0;
      }
      if (outputCanvas) {
        outputCanvas.width = 0;
        outputCanvas.height = 0;
      }
      if (target) target.buffer = null;
    }
  }

  if (!caseResult) throw new Error(`Benchmark case ${clipName}/${targetHeight}p produced no result`);
  write('case', caseResult);
}

async function run(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const requestedClip = params.get('clip') as keyof typeof CLIPS | null;
  const clips = requestedClip && requestedClip in CLIPS ? [requestedClip] : (Object.keys(CLIPS) as Array<keyof typeof CLIPS>);
  const model = (params.get('model') || DEFAULT_MODEL) as ModelType;
  const frameLimit = getFrameLimit();
  const warmupFrames = getWarmupFrames();

  write('benchmark', 'encoding');
  write('audio', 'video-only (intentional; isolates decode/upscale/encode)');
  write('targets', getTargetHeights());
  write('frames', frameLimit === null ? 'all' : frameLimit);
  write('warmupFrames', warmupFrames);

  for (const clip of clips) {
    for (const targetHeight of getTargetHeights()) {
      try {
        await runCase(clip, targetHeight, model, frameLimit, warmupFrames);
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
