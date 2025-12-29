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
   * Converts ImageBitmap/VideoFrame to normalized Float32 tensor.
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

    // Convert to Float32 tensor in NCHW format (normalized to 0-1)
    const tensorData = new Float32Array(3 * height * width);

    for (let i = 0; i < height * width; i++) {
      // RGB channels (skip alpha)
      tensorData[i] = pixels[i * 4] / 255.0;                          // R
      tensorData[height * width + i] = pixels[i * 4 + 1] / 255.0;     // G
      tensorData[2 * height * width + i] = pixels[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', tensorData, [1, 3, height, width]);

    return { tensor, width, height };
  }

  /**
   * Postprocess model output to ImageData.
   * Converts Float32 tensor back to RGBA pixels.
   */
  private postprocess(
    output: ort.Tensor,
    inputWidth: number,
    inputHeight: number
  ): ImageData {
    const data = output.data as Float32Array;
    const outputWidth = inputWidth * this.config.scale;
    const outputHeight = inputHeight * this.config.scale;

    // Create output pixel array (RGBA)
    const pixels = new Uint8ClampedArray(outputWidth * outputHeight * 4);

    for (let i = 0; i < outputHeight * outputWidth; i++) {
      // Convert from NCHW normalized floats back to RGBA bytes
      const r = Math.round(Math.max(0, Math.min(1, data[i])) * 255);
      const g = Math.round(Math.max(0, Math.min(1, data[outputHeight * outputWidth + i])) * 255);
      const b = Math.round(Math.max(0, Math.min(1, data[2 * outputHeight * outputWidth + i])) * 255);

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
      const { tensor } = await this.preprocess(source);
      const output = await this.upscaleTile(tensor);
      const imageData = this.postprocess(output, inputWidth, inputHeight);
      this.ctx.putImageData(imageData, 0, 0);
      tensor.dispose();
      output.dispose();
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
        const tileBitmap = await createImageBitmap(tileImageData);

        // Process tile
        const { tensor } = await this.preprocess(tileBitmap);
        const output = await this.upscaleTile(tensor);
        const outputImageData = this.postprocess(output, srcW, srcH);

        // Calculate output position (accounting for padding)
        const padOffsetX = srcX === 0 ? 0 : tilePadding * scale;
        const padOffsetY = srcY === 0 ? 0 : tilePadding * scale;

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

        // Cleanup
        tensor.dispose();
        output.dispose();
        tileBitmap.close();
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
