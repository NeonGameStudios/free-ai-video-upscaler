/**
 * Type-safe worker message definitions for communication between
 * the main thread and the video processing worker.
 *
 * Updated for Real-ESRGAN and Real-CUGAN model support.
 */

export interface Resolution {
  width: number;
  height: number;
}

/**
 * Round a display dimension to the next codec-friendly even integer. Rounding
 * upward avoids silently dropping a source row/column for odd-sized videos.
 */
export function ceilToEven(value: number): number {
  if (!Number.isFinite(value)) return 2;
  const rounded = Math.max(2, Math.ceil(value));
  return rounded % 2 === 0 ? rounded : rounded + 1;
}

/**
 * Resolve the model's native output against a height cap and normalize the
 * final encode/display dimensions in one shared place for main and worker.
 */
export function resolveOutputResolution(
  source: Resolution,
  scale: number,
  targetHeight?: number,
  defaultTargetHeight = 1080,
): Resolution {
  const safeScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
  const nativeWidth = source.width * safeScale;
  const nativeHeight = source.height * safeScale;
  const fallbackTarget = Math.max(source.height, defaultTargetHeight);
  const resolvedTarget = typeof targetHeight === 'number'
    && Number.isFinite(targetHeight)
    && targetHeight > 0
    ? targetHeight
    : fallbackTarget;
  const outputHeight = Math.min(nativeHeight, resolvedTarget);
  const outputWidth = nativeHeight > 0
    ? outputHeight * (nativeWidth / nativeHeight)
    : nativeWidth;

  return {
    width: ceilToEven(outputWidth),
    height: ceilToEven(outputHeight),
  };
}

/**
 * Resolve the source dimensions sent into a fixed-scale model. Inference
 * surfaces do not need codec-even dimensions; rounding upward ensures the
 * model output never undershoots the requested encode surface.
 */
export function resolveInferenceResolution(
  source: Resolution,
  scale: number,
  encode: Resolution,
): Resolution {
  const safeScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
  const width = Math.max(1, Math.ceil(encode.width / safeScale));
  const height = Math.max(1, Math.ceil(encode.height / safeScale));

  if (width >= source.width && height >= source.height) {
    return { width: source.width, height: source.height };
  }

  return {
    width: Math.min(source.width, width),
    height: Math.min(source.height, height),
  };
}

export interface FrameTiming {
  /** Decoder/demux iterator wait plus VideoSample -> VideoFrame conversion. */
  decodeMs: number;
  decodeWaitMs: number;
  frameConversionMs: number;
  audioMs: number;
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  gpuWaitMs: number;
  gpuTimestampMs: number;
  canvasMs: number;
  encodeMs: number;
  /** Output mux/target finalization time (populated by the final report). */
  finalizeMs: number;
  /** Frame-loop throughput through the latest encoded frame. */
  wallFps: number;
  /** Final pipeline throughput; zero until mux/target finalization completes. */
  pipelineFps: number;
  totalMs: number;
  tileCount: number;
  inputPixels: number;
  inferredPixels: number;
  frames: number;
}

// Available upscaling models
export type ModelType =
  | 'realesr-animevideov3'
  | 'animejanai-v3-sd'
  | 'animejanai-v3-hd'
  | 'animejanai-v3-hd-fast'
  | 'animejanai-v3-hd-superfast'
  | 'realesrgan-anime-fast'
  | 'realesrgan-anime-plus'
  | 'realesrgan-general-fast'
  | 'realesrgan-general-plus'
  | 'realcugan-2x'
  | 'realcugan-4x'
  | 'realplksr-deh264-1x'
  | 'realplksr-dejpg-1x'
  | 'realplksr-denoise-1x'
  | 'scunet-psnr'
  | 'scunet-gan'
  | 'swinir-jpeg40-1x';

export type ModelCategory = 'upscale' | 'cleanup';

// Model metadata
export interface ModelInfo {
  id: ModelType;
  name: string;
  description: string;
  scale: number;
  supportsDenoising: boolean;
  category: ModelCategory;
}

// All available models
export const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: 'realesr-animevideov3',
    name: 'RealESR AnimeVideo v3',
    description: 'Optimized for anime videos (4x) - Recommended',
    scale: 4,
    supportsDenoising: false,
    category: 'upscale'
  },
  {
    id: 'animejanai-v3-sd',
    name: 'AnimeJaNai V3 - SD',
    description: 'Soft upscaling, faithful to source (2x)',
    scale: 2,
    supportsDenoising: false,
    category: 'upscale'
  },
  {
    id: 'animejanai-v3-hd',
    name: 'AnimeJaNai V3 - HD',
    description: 'Sharp upscaling for high quality sources (2x)',
    scale: 2,
    supportsDenoising: false,
    category: 'upscale'
  },
  {
    id: 'animejanai-v3-hd-fast',
    name: 'AnimeJaNai V3 - HD Fast',
    description: 'Fast HD upscaling, good balance of speed/quality (2x)',
    scale: 2,
    supportsDenoising: false,
    category: 'upscale'
  },
  {
    id: 'animejanai-v3-hd-superfast',
    name: 'AnimeJaNai V3 - HD Superfast',
    description: 'Fastest HD upscaling, lower quality (2x)',
    scale: 2,
    supportsDenoising: false,
    category: 'upscale'
  },
  {
    id: 'realesrgan-anime-fast',
    name: 'Real-ESRGAN Anime Fast',
    description: 'Fast anime upscaling (4x)',
    scale: 4,
    supportsDenoising: false,
    category: 'upscale'
  },
  {
    id: 'realesrgan-anime-plus',
    name: 'Real-ESRGAN Anime Plus',
    description: 'High quality anime upscaling (4x)',
    scale: 4,
    supportsDenoising: false,
    category: 'upscale'
  },
  {
    id: 'realesrgan-general-fast',
    name: 'Real-ESRGAN General Fast',
    description: 'Fast general content upscaling (4x)',
    scale: 4,
    supportsDenoising: false,
    category: 'upscale'
  },
  {
    id: 'realesrgan-general-plus',
    name: 'Real-ESRGAN General Plus',
    description: 'High quality general content upscaling (4x)',
    scale: 4,
    supportsDenoising: false,
    category: 'upscale'
  },
  {
    id: 'realcugan-2x',
    name: 'Real-CUGAN 2x',
    description: 'Conservative anime upscaling with denoising (2x)',
    scale: 2,
    supportsDenoising: true,
    category: 'upscale'
  },
  {
    id: 'realcugan-4x',
    name: 'Real-CUGAN 4x',
    description: 'High quality anime upscaling with denoising (4x)',
    scale: 4,
    supportsDenoising: true,
    category: 'upscale'
  },
  {
    id: 'realplksr-deh264-1x',
    name: 'RealPLKSR DeH264 1x',
    description: 'Same-resolution H.264 compression artifact cleanup',
    scale: 1,
    supportsDenoising: false,
    category: 'cleanup'
  },
  {
    id: 'realplksr-dejpg-1x',
    name: 'RealPLKSR DeJPG 1x',
    description: 'Same-resolution JPEG/block artifact cleanup',
    scale: 1,
    supportsDenoising: false,
    category: 'cleanup'
  },
  {
    id: 'realplksr-denoise-1x',
    name: 'RealPLKSR Denoise 1x',
    description: 'Same-resolution general denoising without upscaling',
    scale: 1,
    supportsDenoising: false,
    category: 'cleanup'
  },
  {
    id: 'scunet-psnr',
    name: 'SCUNet PSNR 1x',
    description: 'Same-resolution blind denoising with conservative PSNR output',
    scale: 1,
    supportsDenoising: false,
    category: 'cleanup'
  },
  {
    id: 'scunet-gan',
    name: 'SCUNet GAN 1x',
    description: 'Same-resolution blind denoising with stronger GAN restoration',
    scale: 1,
    supportsDenoising: false,
    category: 'cleanup'
  },
  {
    id: 'swinir-jpeg40-1x',
    name: 'SwinIR JPEG40 1x',
    description: 'Same-resolution JPEG artifact and ringing cleanup',
    scale: 1,
    supportsDenoising: false,
    category: 'cleanup'
  }
];

// Output format options
export type OutputFormat = 'mp4' | 'webm';

export interface OutputFormatInfo {
  id: OutputFormat;
  name: string;
  mimeType: string;
  extension: string;
  codec: string;
}

export const OUTPUT_FORMATS: OutputFormatInfo[] = [
  {
    id: 'mp4',
    name: 'MP4 (H.264)',
    mimeType: 'video/mp4',
    extension: '.mp4',
    codec: 'avc'
  },
  {
    id: 'webm',
    name: 'WebM (VP9)',
    mimeType: 'video/webm',
    extension: '.webm',
    codec: 'vp9'
  }
];

// Output resolution presets
export type OutputResolution = 'source' | 'auto' | '720p' | '1080p' | '1440p' | '4k';

export interface ResolutionPreset {
  id: OutputResolution;
  name: string;
  maxHeight: number | null; // null means resolved dynamically by the caller
}

export const RESOLUTION_PRESETS: ResolutionPreset[] = [
  { id: 'source', name: 'Keep Existing Resolution', maxHeight: null },
  { id: 'auto', name: 'Auto (Native Scale)', maxHeight: null },
  { id: '720p', name: '720p (1280×720)', maxHeight: 720 },
  { id: '1080p', name: '1080p (1920×1080)', maxHeight: 1080 },
  { id: '1440p', name: '1440p (2560×1440)', maxHeight: 1440 },
  { id: '4k', name: '4K (3840×2160)', maxHeight: 2160 }
];

// Denoise levels for Real-CUGAN
export type DenoiseLevel = 0 | 1 | 2 | 3;

export interface UpscaleSettings {
  model: ModelType;
  denoiseLevel: DenoiseLevel;
  outputFormat: OutputFormat;
  outputResolution: OutputResolution;
}

export interface ModelConfig {
  modelId: ModelType;
  scale: number;
  tileSize: number;
  tilePadding: number;
  inputWidth?: number;
  inputHeight?: number;
  inputMultiple?: number;
  denoiseLevel?: DenoiseLevel;
}

// Messages sent FROM main thread TO worker
export type WorkerRequestMessage =
  | { cmd: 'isSupported' }
  | { cmd: 'init'; data: InitData }
  | { cmd: 'switchModel'; data: SwitchModelData }
  | { cmd: 'renderPreview'; data: RenderPreviewData }
  | { cmd: 'cancel' }
  | { cmd: 'process'; inputHandle?: FileSystemFileHandle; inputFile?: File; outputHandle?: FileSystemFileHandle; settings: ProcessSettings };

export interface InitData {
  bitmap: ImageBitmap;
  upscaled: OffscreenCanvas;
  original: OffscreenCanvas;
  resolution: Resolution;
  modelConfig: ModelConfig;
  targetHeight?: number;
}

export interface SwitchModelData {
  bitmap: ImageBitmap;
  modelConfig: ModelConfig;
  targetHeight?: number;
}

export interface RenderPreviewData {
  bitmap: ImageBitmap;
  targetHeight?: number;
}

export interface ProcessSettings {
  outputFormat: OutputFormat;
  outputResolution: OutputResolution;
  targetHeight?: number;
}

export interface PipelineTelemetry {
  executionProvider: 'webgpu' | 'wasm';
  renderPath:
    | 'webgpu-direct'
    | 'webgpu-2d-mirror'
    | 'webgpu-2d-resize'
    | 'cpu-tensor-direct-2d'
    | 'cpu-tensor-2d-resize';
  encoderConfig?: VideoEncoderConfig;
}

// Messages sent FROM worker TO main thread
export type WorkerResponseMessage =
  | { cmd: 'isSupported'; data: boolean }
  | { cmd: 'modelLoading'; data: number }
  | { cmd: 'status'; data: string }
  | { cmd: 'modelLoaded' }
  | { cmd: 'progress'; data: number }
  | { cmd: 'timing'; data: FrameTiming }
  | { cmd: 'pipeline'; data: PipelineTelemetry }
  | { cmd: 'eta'; data: string }
  | { cmd: 'process' }
  | { cmd: 'cancelled' }
  | { cmd: 'error'; data: string }
  | { cmd: 'finished'; data: ArrayBuffer | null };

// Type guard helpers
export function isWorkerRequestMessage(msg: any): msg is WorkerRequestMessage {
  return msg && typeof msg.cmd === 'string';
}

export function isWorkerResponseMessage(msg: any): msg is WorkerResponseMessage {
  return msg && typeof msg.cmd === 'string';
}

// Helper to get model info by ID
export function getModelInfo(modelId: ModelType): ModelInfo | undefined {
  return AVAILABLE_MODELS.find(m => m.id === modelId);
}

// Helper to get format info by ID
export function getFormatInfo(formatId: OutputFormat): OutputFormatInfo | undefined {
  return OUTPUT_FORMATS.find(f => f.id === formatId);
}

// Helper to get resolution preset by ID
export function getResolutionPreset(resolutionId: OutputResolution): ResolutionPreset | undefined {
  return RESOLUTION_PRESETS.find(r => r.id === resolutionId);
}
