/**
 * Model loader with IndexedDB caching for ONNX models.
 *
 * Downloads models from CDN URLs and caches them in IndexedDB
 * for faster subsequent loads.
 */

import type { DenoiseLevel, ModelType } from './types/worker-messages';

// IndexedDB configuration
const DB_NAME = 'upscaler-models';
const DB_VERSION = 1;
const STORE_NAME = 'models';

interface ExternalModelAsset {
  path: string;
  url: string;
  size: number;
}

interface ModelAsset {
  cacheKey: string;
  url: string;
  size: number;
  externalData?: ExternalModelAsset[];
}

export interface LoadedModel {
  data: ArrayBuffer;
  externalData?: Array<{ path: string; data: ArrayBuffer }>;
}

const HUGGLYBERRY_MODEL_BASE = 'https://huggingface.co/hugglyberry/upscale-and-refine-models/resolve/main';

// Model download URLs from Hugging Face. Using resolve/main/ pattern redirects to CDN.
const MODEL_URLS: Record<ModelType, string> = {
  // RealESR AnimeVideo v3 - compact model optimized for anime videos
  // Uses lightweight 4B32F architecture from xiongjie's repo
  'realesr-animevideov3': 'https://huggingface.co/xiongjie/lightweight-real-ESRGAN-anime/resolve/main/RealESRGAN_x4plus_anime_4B32F.onnx',
  // AnimeJaNai V3 models - hosted locally in public/models/
  // Source: https://github.com/the-database/mpv-upscale-2x_animejanai/releases
  'animejanai-v3-sd': '/models/2x_AnimeJaNai_SD_V1beta34_Compact.onnx',
  'animejanai-v3-hd': '/models/2x_AnimeJaNai_HD_V3_Compact.onnx',
  'animejanai-v3-hd-fast': '/models/2x_AnimeJaNai_HD_V3_UltraCompact.onnx',
  'animejanai-v3-hd-superfast': '/models/2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx',
  // Real-ESRGAN models - using deepghs/imgutils-models repository
  'realesrgan-anime-fast': 'https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx',
  'realesrgan-anime-plus': 'https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx',
  // Real-ESRGAN general models - need to be converted and hosted (Qualcomm removed theirs)
  'realesrgan-general-fast': '',
  'realesrgan-general-plus': '',
  // Real-CUGAN 2x is resolved by denoise level because its ONNX models use external weights.
  'realcugan-2x': `${HUGGLYBERRY_MODEL_BASE}/realcugan-2x-denoise1x.onnx`,
  'realcugan-4x': '',
  // Same-resolution cleanup models for compression/noise restoration
  'realplksr-deh264-1x': `${HUGGLYBERRY_MODEL_BASE}/1xDeH264_realplksr.onnx`,
  'realplksr-dejpg-1x': `${HUGGLYBERRY_MODEL_BASE}/1xDeJPG_realplksr_otf.onnx`,
  'realplksr-denoise-1x': `${HUGGLYBERRY_MODEL_BASE}/1xDeNoise_realplksr_otf.onnx`,
  'scunet-psnr': `${HUGGLYBERRY_MODEL_BASE}/SCUNet-PSNR.onnx`,
  'scunet-gan': `${HUGGLYBERRY_MODEL_BASE}/SCUNet-GAN.onnx`,
  'swinir-jpeg40-1x': '/models/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.onnx',
};

const REALCUGAN_2X_VARIANTS: Record<DenoiseLevel, { name: string; onnx: string; data: string; onnxSize: number; dataSize: number }> = {
  0: {
    name: 'no-denoise',
    onnx: 'realcugan-2x-no-denoise.onnx',
    data: 'realcugan-2x-no-denoise.onnx.data',
    onnxSize: 184_677,
    dataSize: 5_177_344,
  },
  1: {
    name: 'denoise1x',
    onnx: 'realcugan-2x-denoise1x.onnx',
    data: 'realcugan-2x-denoise1x.onnx.data',
    onnxSize: 184_639,
    dataSize: 5_177_344,
  },
  2: {
    name: 'denoise2x',
    onnx: 'realcugan-2x-denoise2x.onnx',
    data: 'realcugan-2x-denoise2x.onnx.data',
    onnxSize: 184_639,
    dataSize: 5_177_344,
  },
  3: {
    name: 'denoise3x',
    onnx: 'realcugan-2x-denoise3x.onnx',
    data: 'realcugan-2x-denoise3x.onnx.data',
    onnxSize: 184_639,
    dataSize: 5_177_344,
  },
};

// Model file sizes for progress calculation (approximate, in bytes)
const MODEL_SIZES: Record<ModelType, number> = {
  'realesr-animevideov3': 5_160_000,           // ~5.16 MB (compact model)
  'animejanai-v3-sd': 1_206_372,               // ~1.2 MB (compact)
  'animejanai-v3-hd': 1_210_839,               // ~1.2 MB (compact)
  'animejanai-v3-hd-fast': 614_866,            // ~600 KB (ultracompact)
  'animejanai-v3-hd-superfast': 95_642,        // ~96 KB (superultracompact)
  'realesrgan-anime-fast': 17_900_000,         // ~17.9 MB
  'realesrgan-anime-plus': 17_900_000,         // ~17.9 MB
  'realesrgan-general-fast': 67_100_000,       // ~67.1 MB
  'realesrgan-general-plus': 67_100_000,       // ~67.1 MB
  'realcugan-2x': 20_000_000,                  // Estimated ~20 MB
  'realcugan-4x': 40_000_000,                  // Estimated ~40 MB
  'realplksr-deh264-1x': 29_798_760,           // ~28.4 MiB
  'realplksr-dejpg-1x': 29_798_760,            // ~28.4 MiB
  'realplksr-denoise-1x': 29_874_321,          // ~28.5 MiB
  'scunet-psnr': 91_264_256,                   // ~87.0 MiB
  'scunet-gan': 91_264_256,                    // ~87.0 MiB
  'swinir-jpeg40-1x': 52_313_059,              // ~49.9 MiB local converted model
};

export type LoadProgressCallback = (progress: number, message: string) => void;

/**
 * Open the IndexedDB database.
 */
async function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      reject(new Error('Failed to open IndexedDB'));
    };

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
  });
}

/**
 * Get a cached model from IndexedDB.
 */
async function getCachedModel(cacheKey: string): Promise<ArrayBuffer | null> {
  try {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(cacheKey);

      request.onerror = () => {
        db.close();
        resolve(null);
      };

      request.onsuccess = () => {
        db.close();
        resolve(request.result || null);
      };
    });
  } catch (e) {
    console.warn('Failed to access IndexedDB cache:', e);
    return null;
  }
}

/**
 * Cache a model in IndexedDB.
 */
async function cacheModel(cacheKey: string, data: ArrayBuffer): Promise<void> {
  try {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.put(data, cacheKey);

      request.onerror = () => {
        console.warn('Failed to cache model:', request.error);
        db.close();
        resolve();
      };

      request.onsuccess = () => {
        db.close();
        resolve();
      };
    });
  } catch (e) {
    console.warn('Failed to cache model:', e);
  }
}

/**
 * Validate that downloaded data looks like an ONNX model.
 * ONNX files start with a protobuf header.
 */
function validateOnnxData(data: ArrayBuffer): boolean {
  if (data.byteLength < 8) return false;
  const view = new Uint8Array(data);
  // ONNX protobuf typically starts with field tag 0x08 (ir_version)
  // or 0x12 (producer_name) - check first few bytes are valid protobuf
  return view[0] === 0x08 || view[0] === 0x12 || view[0] === 0x0a;
}

/**
 * Download a model from URL with progress tracking and retry.
 */
async function downloadModel(
  url: string,
  expectedSize: number,
  onProgress?: LoadProgressCallback,
  retryCount: number = 1,
  validateOnnx: boolean = true
): Promise<ArrayBuffer> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retryCount; attempt++) {
    try {
      if (attempt > 0) {
        onProgress?.(0, `Retrying download (attempt ${attempt + 1})...`);
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      }

      const response = await fetch(url, {
        mode: 'cors',
        credentials: 'omit',
      });

      if (!response.ok) {
        throw new Error(`Failed to download model: ${response.status} ${response.statusText}`);
      }

  // Try to get content length for progress tracking
  const contentLength = response.headers.get('content-length');
  const totalSize = contentLength ? parseInt(contentLength, 10) : expectedSize;

  if (!response.body) {
    // Fallback for browsers without streaming support
    const buffer = await response.arrayBuffer();
    onProgress?.(100, 'Download complete');
    return buffer;
  }

  // Stream download with progress
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let receivedLength = 0;

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    chunks.push(value);
    receivedLength += value.length;

    const progress = Math.min(99, Math.round((receivedLength / totalSize) * 100));
    onProgress?.(progress, `Downloading: ${formatBytes(receivedLength)} / ${formatBytes(totalSize)}`);
  }

      // Combine chunks into a single ArrayBuffer
      const buffer = new Uint8Array(receivedLength);
      let position = 0;
      for (const chunk of chunks) {
        buffer.set(chunk, position);
        position += chunk.length;
      }

      // Validate the downloaded data
      if (validateOnnx && !validateOnnxData(buffer.buffer)) {
        throw new Error('Downloaded file does not appear to be a valid ONNX model');
      }

      onProgress?.(100, 'Download complete');
      return buffer.buffer;
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      if (attempt === retryCount) break;
    }
  }

  throw lastError || new Error('Download failed after retries');
}

/**
 * Format bytes into human-readable string.
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getModelAsset(modelId: ModelType, denoiseLevel: DenoiseLevel = 0): ModelAsset {
  if (modelId === 'realcugan-2x') {
    const variant = REALCUGAN_2X_VARIANTS[denoiseLevel] || REALCUGAN_2X_VARIANTS[0];
    return {
      cacheKey: `${modelId}:${variant.name}`,
      url: `${HUGGLYBERRY_MODEL_BASE}/${variant.onnx}`,
      size: variant.onnxSize,
      externalData: [{
        path: variant.data,
        url: `${HUGGLYBERRY_MODEL_BASE}/${variant.data}`,
        size: variant.dataSize,
      }],
    };
  }

  return {
    cacheKey: modelId,
    url: MODEL_URLS[modelId],
    size: MODEL_SIZES[modelId],
  };
}

function scaleProgress(
  onProgress: LoadProgressCallback | undefined,
  start: number,
  end: number
): LoadProgressCallback | undefined {
  if (!onProgress) return undefined;

  return (progress, message) => {
    const scaledProgress = Math.round(start + (progress / 100) * (end - start));
    onProgress(Math.min(end, scaledProgress), message);
  };
}

/**
 * Evict a cached model from IndexedDB.
 */
async function evictCachedModel(cacheKey: string): Promise<void> {
  try {
    const db = await openDatabase();
    return new Promise((resolve) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(cacheKey);
      request.onsuccess = () => { db.close(); resolve(); };
      request.onerror = () => { db.close(); resolve(); };
    });
  } catch {
    // Ignore eviction errors
  }
}

/**
 * Load a model by ID, downloading if necessary.
 * Returns an ArrayBuffer containing the model data.
 * If cached data is corrupt, evicts and re-downloads.
 */
export async function loadModel(
  modelId: ModelType,
  onProgress?: LoadProgressCallback,
  denoiseLevel?: DenoiseLevel
): Promise<LoadedModel> {
  const asset = getModelAsset(modelId, denoiseLevel);

  if (!asset.url) {
    throw new Error(
      `Model "${modelId}" is not available for download. ` +
      `Please run the conversion script: python scripts/convert_model.py ${modelId}`
    );
  }

  // Check cache first
  onProgress?.(0, 'Checking cache...');
  let modelData = await getCachedModel(asset.cacheKey);

  if (modelData) {
    // Validate cached data
    if (validateOnnxData(modelData)) {
      onProgress?.(asset.externalData ? 20 : 100, 'Loaded model from cache');
    } else {
      // Corrupt cache - evict and re-download
      console.warn(`Cached model ${asset.cacheKey} appears corrupt, re-downloading...`);
      await evictCachedModel(asset.cacheKey);
      modelData = null;
    }
  }

  if (!modelData) {
    // Download the model
    onProgress?.(0, 'Starting download...');
    modelData = await downloadModel(
      asset.url,
      asset.size,
      scaleProgress(onProgress, 0, asset.externalData ? 20 : 100)
    );

    // Cache for future use
    onProgress?.(asset.externalData ? 20 : 100, 'Caching model...');
    await cacheModel(asset.cacheKey, modelData);
  }

  if (!asset.externalData?.length) {
    return { data: modelData };
  }

  const externalData: LoadedModel['externalData'] = [];
  const progressRange = 80 / asset.externalData.length;

  for (let i = 0; i < asset.externalData.length; i++) {
    const externalAsset = asset.externalData[i];
    const cacheKey = `${asset.cacheKey}:external:${externalAsset.path}`;
    let externalBuffer = await getCachedModel(cacheKey);
    const start = 20 + i * progressRange;
    const end = 20 + (i + 1) * progressRange;

    if (externalBuffer) {
      onProgress?.(Math.round(end), `Loaded ${externalAsset.path} from cache`);
    } else {
      externalBuffer = await downloadModel(
        externalAsset.url,
        externalAsset.size,
        scaleProgress(onProgress, start, end),
        1,
        false
      );
      await cacheModel(cacheKey, externalBuffer);
    }

    externalData.push({
      path: externalAsset.path,
      data: externalBuffer,
    });
  }

  onProgress?.(100, 'Model ready');

  return {
    data: modelData,
    externalData,
  };
}

/**
 * Check if a model is available (either cached or has download URL).
 */
export function isModelAvailable(modelId: ModelType): boolean {
  return !!getModelAsset(modelId).url;
}

/**
 * Get the download URL for a model.
 */
export function getModelUrl(modelId: ModelType): string | null {
  return MODEL_URLS[modelId] || null;
}

/**
 * Clear all cached models from IndexedDB.
 */
export async function clearModelCache(): Promise<void> {
  try {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.clear();

      request.onerror = () => {
        db.close();
        reject(new Error('Failed to clear cache'));
      };

      request.onsuccess = () => {
        db.close();
        resolve();
      };
    });
  } catch (e) {
    console.warn('Failed to clear model cache:', e);
  }
}

/**
 * Check if a specific model is cached.
 */
export async function isModelCached(modelId: ModelType): Promise<boolean> {
  const cached = await getCachedModel(getModelAsset(modelId).cacheKey);
  return cached !== null;
}

export default {
  loadModel,
  isModelAvailable,
  getModelUrl,
  clearModelCache,
  isModelCached,
};
