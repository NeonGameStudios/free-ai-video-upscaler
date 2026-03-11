/**
 * Model loader with IndexedDB caching for ONNX models.
 *
 * Downloads models from CDN URLs and caches them in IndexedDB
 * for faster subsequent loads.
 */

import type { ModelType } from './types/worker-messages';

// IndexedDB configuration
const DB_NAME = 'upscaler-models';
const DB_VERSION = 1;
const STORE_NAME = 'models';

// Model download URLs from Hugging Face
// Using resolve/main/ pattern which redirects to CDN
const MODEL_URLS: Record<ModelType, string> = {
  // RealESR AnimeVideo v3 - compact model optimized for anime videos
  // Uses lightweight 4B32F architecture from xiongjie's repo
  'realesr-animevideov3': 'https://huggingface.co/xiongjie/lightweight-real-ESRGAN-anime/resolve/main/RealESRGAN_x4plus_anime_4B32F.onnx',
  // AnimeJaNai V3 models - placeholder URLs (need to be hosted)
  // Download from: https://github.com/the-database/mpv-upscale-2x_animejanai/releases
  // Convert .pth to .onnx using chaiNNer, then host on Hugging Face
  'animejanai-v3-sd': '',
  'animejanai-v3-hd': '',
  // Real-ESRGAN models - using deepghs/imgutils-models repository
  'realesrgan-anime-fast': 'https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx',
  'realesrgan-anime-plus': 'https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx',
  'realesrgan-general-fast': 'https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx',
  'realesrgan-general-plus': 'https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx',
  // Real-CUGAN models - placeholder URLs (need to be converted and hosted)
  'realcugan-2x': '',
  'realcugan-4x': '',
};

// Model file sizes for progress calculation (approximate, in bytes)
const MODEL_SIZES: Record<ModelType, number> = {
  'realesr-animevideov3': 5_160_000,    // ~5.16 MB (compact model)
  'animejanai-v3-sd': 5_000_000,        // ~5 MB estimated (compact)
  'animejanai-v3-hd': 5_000_000,        // ~5 MB estimated (compact)
  'realesrgan-anime-fast': 17_900_000,  // ~17.9 MB
  'realesrgan-anime-plus': 17_900_000,  // ~17.9 MB
  'realesrgan-general-fast': 67_100_000, // ~67.1 MB
  'realesrgan-general-plus': 67_100_000, // ~67.1 MB
  'realcugan-2x': 20_000_000,  // Estimated ~20 MB
  'realcugan-4x': 40_000_000,  // Estimated ~40 MB
};

// Maximum number of download retry attempts
const MAX_RETRIES = 3;

export type LoadProgressCallback = (progress: number, message: string) => void;

/**
 * Validate that a buffer contains a valid ONNX model.
 * ONNX uses protobuf format; the first bytes should contain a valid protobuf
 * field tag. A minimal check: the buffer must be non-empty and start with
 * a valid protobuf field header (tag 0x08 for the ir_version field).
 */
function isValidOnnxModel(data: ArrayBuffer): boolean {
  if (data.byteLength < 8) return false;
  const view = new Uint8Array(data);
  // ONNX protobuf starts with field 1 (ir_version), wire type 0 → tag byte 0x08
  return view[0] === 0x08;
}

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
async function getCachedModel(modelId: ModelType): Promise<ArrayBuffer | null> {
  try {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(modelId);

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
async function cacheModel(modelId: ModelType, data: ArrayBuffer): Promise<void> {
  try {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.put(data, modelId);

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
 * Download a model from URL with progress tracking.
 */
async function downloadModel(
  url: string,
  expectedSize: number,
  onProgress?: LoadProgressCallback
): Promise<ArrayBuffer> {
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

  onProgress?.(100, 'Download complete');
  return buffer.buffer;
}

/**
 * Format bytes into human-readable string.
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Evict a single model from the IndexedDB cache.
 */
async function evictCachedModel(modelId: ModelType): Promise<void> {
  try {
    const db = await openDatabase();
    return new Promise((resolve) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(modelId);

      request.onerror = () => {
        console.warn('Failed to evict cached model:', request.error);
        db.close();
        resolve();
      };

      request.onsuccess = () => {
        db.close();
        resolve();
      };
    });
  } catch (e) {
    console.warn('Failed to evict cached model:', e);
  }
}

/**
 * Load a model by ID, downloading if necessary.
 * Validates ONNX data and retries on corrupt downloads.
 * Returns an ArrayBuffer containing the model data.
 */
export async function loadModel(
  modelId: ModelType,
  onProgress?: LoadProgressCallback
): Promise<ArrayBuffer> {
  // Check cache first
  onProgress?.(0, 'Checking cache...');
  const cached = await getCachedModel(modelId);

  if (cached) {
    if (isValidOnnxModel(cached)) {
      onProgress?.(100, 'Loaded from cache');
      return cached;
    }
    // Cached data is corrupt — evict and re-download
    console.warn(`Cached model "${modelId}" is corrupt, re-downloading...`);
    await evictCachedModel(modelId);
  }

  // Get download URL
  const url = MODEL_URLS[modelId];

  if (!url) {
    throw new Error(
      `Model "${modelId}" is not available for download. ` +
      `Please run the conversion script: python scripts/convert_model.py ${modelId}`
    );
  }

  // Download the model with retry logic
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      onProgress?.(0, attempt > 1 ? `Retrying download (attempt ${attempt}/${MAX_RETRIES})...` : 'Starting download...');
      const data = await downloadModel(url, MODEL_SIZES[modelId], onProgress);

      // Validate the downloaded data
      if (!isValidOnnxModel(data)) {
        throw new Error('Downloaded file is not a valid ONNX model');
      }

      // Cache for future use
      onProgress?.(100, 'Caching model...');
      await cacheModel(modelId, data);

      return data;
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      console.warn(`Download attempt ${attempt}/${MAX_RETRIES} failed for "${modelId}":`, lastError.message);

      if (attempt < MAX_RETRIES) {
        // Exponential backoff: 1s, 2s, 4s
        const delay = 1000 * Math.pow(2, attempt - 1);
        onProgress?.(0, `Download failed, retrying in ${delay / 1000}s...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw new Error(
    `Failed to download model "${modelId}" after ${MAX_RETRIES} attempts: ${lastError?.message}`
  );
}

/**
 * Check if a model is available (either cached or has download URL).
 */
export function isModelAvailable(modelId: ModelType): boolean {
  return !!MODEL_URLS[modelId];
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
  const cached = await getCachedModel(modelId);
  return cached !== null;
}

export default {
  loadModel,
  isModelAvailable,
  getModelUrl,
  clearModelCache,
  isModelCached,
};
