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

// Available upscaling models
export type ModelType =
  | 'realesrgan-anime-fast'
  | 'realesrgan-anime-plus'
  | 'realesrgan-general-fast'
  | 'realesrgan-general-plus'
  | 'realcugan-2x'
  | 'realcugan-4x';

// Model metadata
export interface ModelInfo {
  id: ModelType;
  name: string;
  description: string;
  scale: number;
  supportsDenoising: boolean;
  modelFile: string;
}

// All available models
export const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: 'realesrgan-anime-fast',
    name: 'Real-ESRGAN Anime Fast',
    description: 'Fast anime upscaling (4x)',
    scale: 4,
    supportsDenoising: false,
    modelFile: 'realesrgan-anime-fast.onnx'
  },
  {
    id: 'realesrgan-anime-plus',
    name: 'Real-ESRGAN Anime Plus',
    description: 'High quality anime upscaling (4x)',
    scale: 4,
    supportsDenoising: false,
    modelFile: 'realesrgan-anime-plus.onnx'
  },
  {
    id: 'realesrgan-general-fast',
    name: 'Real-ESRGAN General Fast',
    description: 'Fast general content upscaling (4x)',
    scale: 4,
    supportsDenoising: false,
    modelFile: 'realesrgan-general-fast.onnx'
  },
  {
    id: 'realesrgan-general-plus',
    name: 'Real-ESRGAN General Plus',
    description: 'High quality general content upscaling (4x)',
    scale: 4,
    supportsDenoising: false,
    modelFile: 'realesrgan-general-plus.onnx'
  },
  {
    id: 'realcugan-2x',
    name: 'Real-CUGAN 2x',
    description: 'Conservative anime upscaling with denoising (2x)',
    scale: 2,
    supportsDenoising: true,
    modelFile: 'realcugan-2x.onnx'
  },
  {
    id: 'realcugan-4x',
    name: 'Real-CUGAN 4x',
    description: 'High quality anime upscaling with denoising (4x)',
    scale: 4,
    supportsDenoising: true,
    modelFile: 'realcugan-4x.onnx'
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
export type OutputResolution = 'auto' | '720p' | '1080p' | '1440p' | '4k';

export interface ResolutionPreset {
  id: OutputResolution;
  name: string;
  maxHeight: number | null; // null means use model's native scale
}

export const RESOLUTION_PRESETS: ResolutionPreset[] = [
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
  modelPath: string;
  scale: number;
  tileSize: number;
  tilePadding: number;
  denoiseLevel?: DenoiseLevel;
}

// Messages sent FROM main thread TO worker
export type WorkerRequestMessage =
  | { cmd: 'isSupported' }
  | { cmd: 'init'; data: InitData }
  | { cmd: 'switchModel'; data: SwitchModelData }
  | { cmd: 'process'; inputHandle: FileSystemFileHandle; outputHandle?: FileSystemFileHandle; settings: ProcessSettings };

export interface InitData {
  bitmap: ImageBitmap;
  upscaled: OffscreenCanvas;
  original: OffscreenCanvas;
  resolution: Resolution;
  modelConfig: ModelConfig;
}

export interface SwitchModelData {
  bitmap: ImageBitmap;
  modelConfig: ModelConfig;
}

export interface ProcessSettings {
  outputFormat: OutputFormat;
  outputResolution: OutputResolution;
  targetHeight?: number;
}

// Messages sent FROM worker TO main thread
export type WorkerResponseMessage =
  | { cmd: 'isSupported'; data: boolean }
  | { cmd: 'modelLoading'; data: number }
  | { cmd: 'modelLoaded' }
  | { cmd: 'progress'; data: number }
  | { cmd: 'eta'; data: string }
  | { cmd: 'process' }
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
