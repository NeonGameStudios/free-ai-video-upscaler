/**
 * Type-safe worker message definitions for communication between
 * the main thread and the video processing worker.
 *
 * Updated for Real-ESRGAN integration.
 */

export interface Resolution {
  width: number;
  height: number;
}

export interface ModelConfig {
  modelPath: string;
  scale: number;
  tileSize: number;
  tilePadding: number;
}

// Messages sent FROM main thread TO worker
export type WorkerRequestMessage =
  | { cmd: 'isSupported' }
  | { cmd: 'init'; data: InitData }
  | { cmd: 'process'; inputHandle: FileSystemFileHandle; outputHandle?: FileSystemFileHandle };

export interface InitData {
  bitmap: ImageBitmap;
  upscaled: OffscreenCanvas;
  original: OffscreenCanvas;
  resolution: Resolution;
  modelConfig?: Partial<ModelConfig>;
}

// Messages sent FROM worker TO main thread
export type WorkerResponseMessage =
  | { cmd: 'isSupported'; data: boolean }
  | { cmd: 'modelLoading'; data: number } // Progress 0-100 for model loading
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
