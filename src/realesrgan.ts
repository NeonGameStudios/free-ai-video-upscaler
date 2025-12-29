/**
 * Real-ESRGAN inference module using ONNX Runtime Web with WebGPU.
 *
 * This module provides browser-based video upscaling using the
 * realesr-animevideov3 model, optimized for anime content.
 */

import * as ort from 'onnxruntime-web/webgpu';

export interface RealESRGANConfig {
  modelPath: string;
  scale: number;
  tileSize: number;
  tilePadding: number;
}

export interface UpscaleResult {
  width: number;
  height: number;
  data: Float32Array;
}

// Default configuration
const DEFAULT_CONFIG: RealESRGANConfig = {
  modelPath: '/models/realesr-animevideov3.onnx',
  scale: 4,
  tileSize: 256,    // Process in tiles to manage memory
  tilePadding: 16,  // Overlap between tiles to avoid seams
};

/**
 * Real-ESRGAN upscaler class for browser-based video upscaling.
 */
export class RealESRGAN {
  private session: ort.InferenceSession | null = null;
  private config: RealESRGANConfig;
  private canvas: OffscreenCanvas | null = null;
  private ctx: OffscreenCanvasRenderingContext2D | null = null;
  private initialized: boolean = false;
  private useWebGPU: boolean = false;

  constructor(config: Partial<RealESRGANConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Check if WebGPU is supported in this environment.
   */
  static async isWebGPUSupported(): Promise<boolean> {
    if (typeof navigator === 'undefined') return false;
    if (!navigator.gpu) return false;

    try {
      const adapter = await navigator.gpu.requestAdapter();
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
  async init(outputCanvas: OffscreenCanvas): Promise<void> {
    if (this.initialized) return;

    // Initialize ONNX Runtime
    await RealESRGAN.initORT();

    // Check WebGPU support
    this.useWebGPU = await RealESRGAN.isWebGPUSupported();

    // Create session options
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: this.useWebGPU
        ? ['webgpu', 'wasm']
        : ['wasm'],
      graphOptimizationLevel: 'all',
    };

    console.log(`Creating inference session (WebGPU: ${this.useWebGPU})...`);

    try {
      // Load the model
      this.session = await ort.InferenceSession.create(
        this.config.modelPath,
        sessionOptions
      );

      console.log('Model loaded successfully');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);
    } catch (e) {
      console.error('Failed to load model:', e);
      throw new Error(`Failed to load Real-ESRGAN model: ${e}`);
    }

    // Set up output canvas
    this.canvas = outputCanvas;
    this.ctx = outputCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;

    this.initialized = true;
  }

  /**
   * Preprocess an image for model input.
   * Converts ImageBitmap/VideoFrame to normalized Float32 tensor.
   */
  private async preprocess(
    source: ImageBitmap | VideoFrame
  ): Promise<{ tensor: ort.Tensor; width: number; height: number }> {
    const width = source.width;
    const height = source.height;

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
      tensorData[i] = pixels[i * 4] / 255.0;                    // R
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
    width: number,
    height: number
  ): ImageData {
    const data = output.data as Float32Array;
    const outputWidth = width * this.config.scale;
    const outputHeight = height * this.config.scale;

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
      throw new Error('RealESRGAN not initialized');
    }

    const inputWidth = source.width;
    const inputHeight = source.height;
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
    const tilesX = Math.ceil(inputWidth / (tileSize - 2 * tilePadding));
    const tilesY = Math.ceil(inputHeight / (tileSize - 2 * tilePadding));

    // Create temporary canvas for tile extraction
    const tileCanvas = new OffscreenCanvas(tileSize, tileSize);
    const tileCtx = tileCanvas.getContext('2d')!;

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        // Calculate tile bounds with padding
        const srcX = Math.max(0, tx * (tileSize - 2 * tilePadding) - tilePadding);
        const srcY = Math.max(0, ty * (tileSize - 2 * tilePadding) - tilePadding);
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
        const dstX = srcX * scale + padOffsetX;
        const dstY = srcY * scale + padOffsetY;

        // Copy to output, excluding padded regions
        const copyW = outputImageData.width - (srcX === 0 ? 0 : padOffsetX) -
                     (srcX + srcW >= inputWidth ? 0 : tilePadding * scale);
        const copyH = outputImageData.height - (srcY === 0 ? 0 : padOffsetY) -
                     (srcY + srcH >= inputHeight ? 0 : tilePadding * scale);

        // Put tile on output canvas
        this.ctx.putImageData(
          outputImageData,
          dstX - padOffsetX,
          dstY - padOffsetY,
          padOffsetX / scale,
          padOffsetY / scale,
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
   * Get the current scale factor.
   */
  getScale(): number {
    return this.config.scale;
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

export default RealESRGAN;
