/**
 * MKV to MP4 remuxing utility using FFmpeg.wasm
 *
 * Remuxing copies streams without re-encoding, preserving 100% quality.
 * Only the container format changes (MKV -> MP4).
 */

import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';

let ffmpeg: FFmpeg | null = null;
let loadPromise: Promise<void> | null = null;

/**
 * Initialize FFmpeg.wasm (lazy loaded on first use)
 */
async function initFFmpeg(onProgress?: (message: string) => void): Promise<FFmpeg> {
  if (ffmpeg && ffmpeg.loaded) {
    return ffmpeg;
  }

  if (loadPromise) {
    await loadPromise;
    return ffmpeg!;
  }

  ffmpeg = new FFmpeg();

  ffmpeg.on('log', ({ message }) => {
    console.log('[FFmpeg]', message);
  });

  // Get URLs for FFmpeg core files using webpack's asset handling
  const coreURL = new URL(
    '../node_modules/@ffmpeg/core/dist/esm/ffmpeg-core.js',
    import.meta.url
  ).href;
  const wasmURL = new URL(
    '../node_modules/@ffmpeg/core/dist/esm/ffmpeg-core.wasm',
    import.meta.url
  ).href;

  loadPromise = (async () => {
    onProgress?.('Loading FFmpeg...');
    await ffmpeg!.load({
      coreURL,
      wasmURL,
    });
  })();

  await loadPromise;
  return ffmpeg;
}

/**
 * Check if a file needs remuxing (is MKV format)
 */
export function needsRemux(filename: string): boolean {
  const ext = filename.toLowerCase().split('.').pop();
  return ext === 'mkv' || ext === 'matroska';
}

/**
 * Remux MKV to MP4 without re-encoding (lossless container conversion)
 *
 * @param file - The input MKV file
 * @param onProgress - Optional progress callback
 * @returns ArrayBuffer of the MP4 file
 */
export async function remuxToMp4(
  file: File,
  onProgress?: (message: string) => void
): Promise<ArrayBuffer> {
  const ff = await initFFmpeg(onProgress);

  const inputName = 'input.mkv';
  const outputName = 'output.mp4';

  onProgress?.('Preparing video for conversion...');

  // Write input file to FFmpeg's virtual filesystem
  await ff.writeFile(inputName, await fetchFile(file));

  onProgress?.('Converting container format (MKV → MP4)...');

  // Remux: copy all streams without re-encoding
  // -c copy = copy codecs (no re-encoding)
  // -map 0 = include all streams from input
  await ff.exec([
    '-i', inputName,
    '-c', 'copy',
    '-map', '0',
    outputName
  ]);

  onProgress?.('Finalizing conversion...');

  // Read output file
  const data = await ff.readFile(outputName);

  // Clean up virtual filesystem
  await ff.deleteFile(inputName);
  await ff.deleteFile(outputName);

  // Convert Uint8Array to ArrayBuffer
  if (data instanceof Uint8Array) {
    // Create a new ArrayBuffer to avoid SharedArrayBuffer issues
    const buffer = new ArrayBuffer(data.byteLength);
    new Uint8Array(buffer).set(data);
    return buffer;
  }

  throw new Error('Unexpected output format from FFmpeg');
}

/**
 * Get the base filename without extension
 */
export function getBaseName(filename: string): string {
  return filename.replace(/\.[^/.]+$/, '');
}
