/**
 * Unified upscaler module supporting Real-ESRGAN and Real-CUGAN models
 * using ONNX Runtime Web with WebGPU acceleration.
 *
 * Supports:
 * - Real-ESRGAN: anime_fast, anime_plus, general_fast, general_plus
 * - Real-CUGAN: 2x, 4x (with denoising support)
 */

import * as ort from 'onnxruntime-web';
import type { DenoiseLevel, ModelType } from './types/worker-messages';
import { loadModel, type LoadProgressCallback } from './model-loader';

/**
 * Convert a float32 value to float16 (IEEE 754 half-precision).
 */
function floatToFloat16(value: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = value;
  const x = int32View[0];

  let bits = (x >> 16) & 0x8000; // sign
  let m = (x >> 12) & 0x07ff;    // mantissa
  const e = (x >> 23) & 0xff;    // exponent

  if (e < 103) {
    return bits; // too small, return signed zero
  }

  if (e > 142) {
    bits |= 0x7c00; // infinity or NaN
    bits |= ((e === 255) ? 0 : 1) && (x & 0x007fffff);
    return bits;
  }

  if (e < 113) {
    m |= 0x0800;
    bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
    return bits;
  }

  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1;
  return bits;
}

/**
 * Convert a float16 value to float32.
 */
function float16ToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;

  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
  } else if (e === 0x1f) {
    return f ? NaN : ((s ? -1 : 1) * Infinity);
  }

  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / Math.pow(2, 10));
}

export interface UpscalerConfig {
  modelId: ModelType;
  scale: number;
  tileSize: number;
  tilePadding: number;
  denoiseLevel?: DenoiseLevel;
}

// Default configuration
const DEFAULT_CONFIG: UpscalerConfig = {
  modelId: 'realesr-animevideov3',
  scale: 4,
  tileSize: 256,
  tilePadding: 16,
  denoiseLevel: 0,
};

/**
 * Unified upscaler class supporting Real-ESRGAN and Real-CUGAN models.
 */
export class Upscaler {
  private session: ort.InferenceSession | null = null;
  private config: UpscalerConfig;
  private canvas: OffscreenCanvas | null = null;
  private ctx: OffscreenCanvasRenderingContext2D | null = null;
  private initialized: boolean = false;
  private useWebGPU: boolean = false;
  private useFloat16: boolean = false;

  constructor(config: Partial<UpscalerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Check if WebGPU is supported in this environment.
   */
  static async isWebGPUSupported(): Promise<boolean> {
    if (typeof navigator === 'undefined') return false;

    // Check for WebGPU support using type assertion
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as any).gpu;
    if (!gpu) return false;

    try {
      const adapter = await gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }

  /**
   * Initialize the ONNX Runtime environment.
   */
  static async initORT(): Promise<boolean> {
    try {
      // Configure ONNX Runtime
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
      ort.env.wasm.simd = true;

      // Set WASM paths for webpack bundling
      const basePath = self.location?.href?.replace(/\/[^\/]*$/, '/') || '/';
      ort.env.wasm.wasmPaths = basePath;

      return true;
    } catch (e) {
      console.error('Failed to initialize ONNX Runtime:', e);
      return false;
    }
  }

  /**
   * Initialize the model and create inference session.
   */
  async init(
    outputCanvas: OffscreenCanvas,
    onProgress?: LoadProgressCallback
  ): Promise<void> {
    if (this.initialized) return;

    // Initialize ONNX Runtime
    await Upscaler.initORT();

    // Check WebGPU support
    this.useWebGPU = await Upscaler.isWebGPUSupported();

    // Create session options
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: this.useWebGPU
        ? ['webgpu', 'wasm']
        : ['wasm'],
      graphOptimizationLevel: 'all',
    };

    console.log(`Creating inference session (WebGPU: ${this.useWebGPU})...`);
    console.log(`Loading model: ${this.config.modelId}`);

    try {
      // Download or retrieve model from cache
      onProgress?.(0, 'Loading model...');
      const modelData = await loadModel(this.config.modelId, onProgress);

      onProgress?.(100, 'Initializing model...');

      // Create session from ArrayBuffer
      this.session = await ort.InferenceSession.create(
        modelData,
        sessionOptions
      );

      console.log('Model loaded successfully');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);

      // AnimeJaNai models use float16 input
      this.useFloat16 = this.config.modelId.includes('animejanai');
      console.log('Using float16:', this.useFloat16);
    } catch (e) {
      console.error('Failed to load model:', e);
      throw new Error(`Failed to load upscaling model: ${e}`);
    }

    // Set up output canvas
    this.canvas = outputCanvas;
    this.ctx = outputCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;

    this.initialized = true;
  }

  /**
   * Switch to a different model.
   */
  async switchModel(
    newConfig: Partial<UpscalerConfig>,
    onProgress?: LoadProgressCallback
  ): Promise<void> {
    // Dispose current session
    if (this.session) {
      await this.session.release();
      this.session = null;
    }

    // Update config
    this.config = { ...this.config, ...newConfig };
    this.initialized = false;

    // Re-initialize with new model
    if (this.canvas) {
      await this.init(this.canvas, onProgress);
    }
  }

  /**
   * Get the current scale factor.
   */
  getScale(): number {
    return this.config.scale;
  }

  /**
   * Update configuration without reloading model.
   */
  updateConfig(config: Partial<UpscalerConfig>): void {
    // Only update non-model-related settings
    if (config.tileSize !== undefined) this.config.tileSize = config.tileSize;
    if (config.tilePadding !== undefined) this.config.tilePadding = config.tilePadding;
    if (config.denoiseLevel !== undefined) this.config.denoiseLevel = config.denoiseLevel;
  }

  /**
   * Get dimensions from source (ImageBitmap or VideoFrame).
   */
  private getSourceDimensions(source: ImageBitmap | VideoFrame): { width: number; height: number } {
    if ('codedWidth' in source) {
      // VideoFrame
      return { width: source.codedWidth, height: source.codedHeight };
    }
    // ImageBitmap
    return { width: source.width, height: source.height };
  }

  /**
   * Preprocess an image for model input.
   * Converts ImageBitmap/VideoFrame to normalized tensor (float32 or float16).
   */
  private async preprocess(
    source: ImageBitmap | VideoFrame
  ): Promise<{ tensor: ort.Tensor; width: number; height: number }> {
    const { width, height } = this.getSourceDimensions(source);

    // Create temporary canvas for pixel extraction
    const tempCanvas = new OffscreenCanvas(width, height);
    const tempCtx = tempCanvas.getContext('2d')!;

    // Draw source to canvas
    tempCtx.drawImage(source, 0, 0);

    // Get pixel data
    const imageData = tempCtx.getImageData(0, 0, width, height);
    const pixels = imageData.data;

    // Convert to tensor in NCHW format (normalized to 0-1)
    const tensorData = new Float32Array(3 * height * width);

    for (let i = 0; i < height * width; i++) {
      // RGB channels (skip alpha)
      tensorData[i] = pixels[i * 4] / 255.0;                          // R
      tensorData[height * width + i] = pixels[i * 4 + 1] / 255.0;     // G
      tensorData[2 * height * width + i] = pixels[i * 4 + 2] / 255.0; // B
    }

    let tensor: ort.Tensor;
    if (this.useFloat16) {
      // Convert to float16 for AnimeJaNai models
      const float16Data = new Uint16Array(tensorData.length);
      for (let i = 0; i < tensorData.length; i++) {
        float16Data[i] = floatToFloat16(tensorData[i]);
      }
      tensor = new ort.Tensor('float16', float16Data, [1, 3, height, width]);
    } else {
      tensor = new ort.Tensor('float32', tensorData, [1, 3, height, width]);
    }

    return { tensor, width, height };
  }

  /**
   * Postprocess model output to ImageData.
   * Converts tensor (float32 or float16) back to RGBA pixels.
   */
  private postprocess(
    output: ort.Tensor,
    inputWidth: number,
    inputHeight: number
  ): ImageData {
    const outputWidth = inputWidth * this.config.scale;
    const outputHeight = inputHeight * this.config.scale;

    // Get values from tensor, converting float16 if needed
    const getValue = this.useFloat16
      ? (i: number) => float16ToFloat((output.data as Uint16Array)[i])
      : (i: number) => (output.data as Float32Array)[i];

    // Create output pixel array (RGBA)
    const pixels = new Uint8ClampedArray(outputWidth * outputHeight * 4);

    for (let i = 0; i < outputHeight * outputWidth; i++) {
      // Convert from NCHW normalized floats back to RGBA bytes
      const r = Math.round(Math.max(0, Math.min(1, getValue(i))) * 255);
      const g = Math.round(Math.max(0, Math.min(1, getValue(outputHeight * outputWidth + i))) * 255);
      const b = Math.round(Math.max(0, Math.min(1, getValue(2 * outputHeight * outputWidth + i))) * 255);

      pixels[i * 4] = r;
      pixels[i * 4 + 1] = g;
      pixels[i * 4 + 2] = b;
      pixels[i * 4 + 3] = 255; // Alpha
    }

    return new ImageData(pixels, outputWidth, outputHeight);
  }

  /**
   * Upscale a single tile of the image.
   */
  private async upscaleTile(
    tensor: ort.Tensor
  ): Promise<ort.Tensor> {
    if (!this.session) {
      throw new Error('Model not initialized');
    }

    const feeds: Record<string, ort.Tensor> = {
      [this.session.inputNames[0]]: tensor
    };

    const results = await this.session.run(feeds);
    return results[this.session.outputNames[0]];
  }

  /**
   * Upscale an image/video frame using tiled processing.
   * This helps manage GPU memory for large images.
   */
  async upscale(source: ImageBitmap | VideoFrame): Promise<void> {
    if (!this.initialized || !this.session || !this.canvas || !this.ctx) {
      throw new Error('Upscaler not initialized');
    }

    const { width: inputWidth, height: inputHeight } = this.getSourceDimensions(source);
    const outputWidth = inputWidth * this.config.scale;
    const outputHeight = inputHeight * this.config.scale;

    // Resize output canvas if needed
    if (this.canvas.width !== outputWidth || this.canvas.height !== outputHeight) {
      this.canvas.width = outputWidth;
      this.canvas.height = outputHeight;
    }

    const { tileSize, tilePadding, scale } = this.config;

    // For small images, process in one go
    if (inputWidth <= tileSize && inputHeight <= tileSize) {
      let tensor: ort.Tensor | null = null;
      let output: ort.Tensor | null = null;
      try {
        const preprocessed = await this.preprocess(source);
        tensor = preprocessed.tensor;
        output = await this.upscaleTile(tensor);
        const imageData = this.postprocess(output, inputWidth, inputHeight);
        this.ctx.putImageData(imageData, 0, 0);
      } finally {
        tensor?.dispose();
        output?.dispose();
      }
      return;
    }

    // Tiled processing for larger images
    const effectiveTileSize = tileSize - 2 * tilePadding;
    const tilesX = Math.ceil(inputWidth / effectiveTileSize);
    const tilesY = Math.ceil(inputHeight / effectiveTileSize);

    // Create temporary canvas for tile extraction
    const tileCanvas = new OffscreenCanvas(tileSize, tileSize);
    const tileCtx = tileCanvas.getContext('2d')!;

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        let tileBitmap: ImageBitmap | null = null;
        let tensor: ort.Tensor | null = null;
        let output: ort.Tensor | null = null;

        try {
          // Calculate tile bounds with padding
          const srcX = Math.max(0, tx * effectiveTileSize - tilePadding);
          const srcY = Math.max(0, ty * effectiveTileSize - tilePadding);
          const srcW = Math.min(tileSize, inputWidth - srcX);
          const srcH = Math.min(tileSize, inputHeight - srcY);

          // Extract tile
          tileCtx.clearRect(0, 0, tileSize, tileSize);
          tileCtx.drawImage(source, srcX, srcY, srcW, srcH, 0, 0, srcW, srcH);

          // Get tile data
          const tileImageData = tileCtx.getImageData(0, 0, srcW, srcH);
          tileBitmap = await createImageBitmap(tileImageData);

          // Process tile
          const preprocessed = await this.preprocess(tileBitmap);
          tensor = preprocessed.tensor;
          output = await this.upscaleTile(tensor);
          const outputImageData = this.postprocess(output, srcW, srcH);

          // Destination position
          const dstX = tx * effectiveTileSize * scale;
          const dstY = ty * effectiveTileSize * scale;

          // Calculate the region to copy (excluding padding on edges)
          const copyStartX = srcX === 0 ? 0 : tilePadding * scale;
          const copyStartY = srcY === 0 ? 0 : tilePadding * scale;
          const copyEndX = (srcX + srcW >= inputWidth) ? srcW * scale : srcW * scale - tilePadding * scale;
          const copyEndY = (srcY + srcH >= inputHeight) ? srcH * scale : srcH * scale - tilePadding * scale;
          const copyW = copyEndX - copyStartX;
          const copyH = copyEndY - copyStartY;

          // Put tile on output canvas
          this.ctx.putImageData(
            outputImageData,
            dstX,
            dstY,
            copyStartX,
            copyStartY,
            copyW,
            copyH
          );
        } finally {
          // Cleanup - always dispose even on error
          tensor?.dispose();
          output?.dispose();
          tileBitmap?.close();
        }
      }
    }
  }

  /**
   * Render a frame directly (simplified path for video processing).
   */
  async render(frame: ImageBitmap | VideoFrame): Promise<void> {
    await this.upscale(frame);
  }

  /**
   * Check if the upscaler is ready.
   */
  isReady(): boolean {
    return this.initialized && this.session !== null;
  }

  /**
   * Dispose of resources.
   */
  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.initialized = false;
  }
}

export default Upscaler;
