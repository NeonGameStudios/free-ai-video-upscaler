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
import { loadModel, type LoadedModel, type LoadProgressCallback } from './model-loader';

type UpscaleSource = ImageBitmap | VideoFrame | OffscreenCanvas;

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

const BYTE_TO_FLOAT32 = new Float32Array(256);
const BYTE_TO_FLOAT16 = new Uint16Array(256);
const FLOAT16_TO_BYTE = new Uint8ClampedArray(65536);

for (let i = 0; i < 256; i++) {
  const normalized = i / 255;
  BYTE_TO_FLOAT32[i] = normalized;
  BYTE_TO_FLOAT16[i] = floatToFloat16(normalized);
}

for (let i = 0; i < FLOAT16_TO_BYTE.length; i++) {
  const value = float16ToFloat(i);
  FLOAT16_TO_BYTE[i] = value <= 0 ? 0 : value >= 1 ? 255 : Math.round(value * 255);
}

export interface UpscalerConfig {
  modelId: ModelType;
  scale: number;
  tileSize: number;
  tilePadding: number;
  inputWidth?: number;
  inputHeight?: number;
  inputMultiple?: number;
  denoiseLevel?: DenoiseLevel;
}

// Default configuration
const DEFAULT_CONFIG: UpscalerConfig = {
  modelId: 'realesr-animevideov3',
  scale: 4,
  tileSize: 256,
  tilePadding: 32,
  inputWidth: undefined,
  inputHeight: undefined,
  inputMultiple: 1,
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
  private executionProvider: 'webgpu' | 'wasm' = 'wasm';
  private outputDataIsFloat32: boolean = false;

  // Reusable canvas for preprocessing (avoid allocation per frame)
  private preprocessCanvas: OffscreenCanvas | null = null;
  private preprocessCtx: OffscreenCanvasRenderingContext2D | null = null;
  private inputFloat32: Float32Array | null = null;
  private inputFloat16: Uint16Array | null = null;
  private outputImageData: ImageData | null = null;

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
    this.useFloat16 = this.config.modelId.includes('animejanai');
    this.outputDataIsFloat32 = false;

    console.log(`Creating inference session (WebGPU available: ${this.useWebGPU})...`);
    console.log(`Loading model: ${this.config.modelId}`);

    try {
      // Download or retrieve model from cache
      onProgress?.(0, 'Loading model...');
      const loadedModel = await loadModel(this.config.modelId, onProgress, this.config.denoiseLevel);

      const canTryWebGPU = this.useWebGPU && !this.useFloat16;
      const providerCandidates: ('webgpu' | 'wasm')[] = canTryWebGPU
        ? ['webgpu', 'wasm']
        : ['wasm'];

      let lastError: unknown = null;

      for (const provider of providerCandidates) {
        try {
          onProgress?.(100, `Initializing model with ${provider.toUpperCase()}...`);
          this.session = await this.createValidatedSession(loadedModel, provider);
          this.executionProvider = provider;
          break;
        } catch (e) {
          lastError = e;
          console.warn(`Failed to initialize ${provider.toUpperCase()} session:`, e);

          if (this.session) {
            await this.session.release();
            this.session = null;
          }
        }
      }

      if (!this.session) {
        throw lastError || new Error('No execution provider could load this model');
      }

      console.log('Model loaded successfully');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);
      console.log('Using float16:', this.useFloat16);
      console.log('Execution provider:', this.executionProvider);
      console.log('Float16 output storage:', this.outputDataIsFloat32 ? 'float32' : 'uint16');
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
   * Create a session and run a small validation inference before using it.
   * This lets float16 models try WebGPU safely while preserving WASM fallback.
   */
  private async createValidatedSession(
    loadedModel: LoadedModel,
    provider: 'webgpu' | 'wasm'
  ): Promise<ort.InferenceSession> {
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: provider === 'webgpu'
        ? ['webgpu', 'wasm']
        : ['wasm'],
      graphOptimizationLevel: 'all',
    };

    if (loadedModel.externalData?.length) {
      sessionOptions.externalData = loadedModel.externalData.map(({ path, data }) => ({ path, data }));
    }

    const session = await ort.InferenceSession.create(loadedModel.data, sessionOptions);

    try {
      await this.validateSession(session);
      return session;
    } catch (e) {
      await session.release();
      throw e;
    }
  }

  private async validateSession(session: ort.InferenceSession): Promise<void> {
    const defaultSize = this.config.scale === 1
      ? 64
      : Math.min(Math.max(this.config.tileSize, 64), 128);
    const width = this.config.inputWidth ?? defaultSize;
    const height = this.config.inputHeight ?? defaultSize;
    const tensorLength = 3 * width * height;
    const inputData = this.createValidationInput(tensorLength);
    const input = this.useFloat16
      ? new ort.Tensor('float16', inputData as Uint16Array, [1, 3, height, width])
      : new ort.Tensor('float32', inputData as Float32Array, [1, 3, height, width]);

    let outputs: ort.InferenceSession.OnnxValueMapType | null = null;

    try {
      outputs = await session.run({ [session.inputNames[0]]: input });

      const output = outputs[session.outputNames[0]];
      if (!output) {
        throw new Error('Validation inference produced no output tensor');
      }

      if (this.useFloat16) {
        this.outputDataIsFloat32 = output.data instanceof Float32Array;
      }

      if (!this.hasUsableOutput(output)) {
        throw new Error('Validation inference produced unusable output');
      }
    } finally {
      input.dispose();

      if (outputs) {
        for (const output of Object.values(outputs)) {
          output.dispose();
        }
      }
    }
  }

  private createValidationInput(length: number): Float32Array | Uint16Array {
    if (this.useFloat16) {
      const data = new Uint16Array(length);
      for (let i = 0; i < length; i++) {
        data[i] = BYTE_TO_FLOAT16[64 + (i % 128)];
      }
      return data;
    }

    const data = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      data[i] = BYTE_TO_FLOAT32[64 + (i % 128)];
    }
    return data;
  }

  private hasUsableOutput(output: ort.Tensor): boolean {
    const data = output.data as Float32Array | Uint16Array;
    const length = Math.min(data.length, 4096);
    let max = -Infinity;
    let min = Infinity;

    if (this.useFloat16 && data instanceof Uint16Array) {
      for (let i = 0; i < length; i++) {
        const value = float16ToFloat(data[i]);
        if (!Number.isFinite(value)) return false;
        max = Math.max(max, value);
        min = Math.min(min, value);
      }
    } else {
      for (let i = 0; i < length; i++) {
        const value = (data as Float32Array)[i];
        if (!Number.isFinite(value)) return false;
        max = Math.max(max, value);
        min = Math.min(min, value);
      }
    }

    return max > 0 && max >= min;
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
    if (config.inputWidth !== undefined) this.config.inputWidth = config.inputWidth;
    if (config.inputHeight !== undefined) this.config.inputHeight = config.inputHeight;
    if (config.inputMultiple !== undefined) this.config.inputMultiple = config.inputMultiple;
    if (config.denoiseLevel !== undefined) this.config.denoiseLevel = config.denoiseLevel;
  }

  /**
   * Get dimensions from source.
   */
  private getSourceDimensions(source: UpscaleSource): { width: number; height: number } {
    if ('codedWidth' in source) {
      // VideoFrame
      return { width: source.displayWidth, height: source.displayHeight };
    }
    // ImageBitmap or OffscreenCanvas
    return { width: source.width, height: source.height };
  }

  /**
   * Preprocess an image for model input.
   * Converts source pixels to normalized tensor (float32 or float16).
   */
  private async preprocess(
    source: UpscaleSource,
    sx: number = 0,
    sy: number = 0,
    sw?: number,
    sh?: number
  ): Promise<{ tensor: ort.Tensor; width: number; height: number }> {
    const sourceDimensions = this.getSourceDimensions(source);
    const width = sw ?? sourceDimensions.width;
    const height = sh ?? sourceDimensions.height;
    const tensorWidth = this.config.inputWidth ?? this.alignToInputMultiple(width);
    const tensorHeight = this.config.inputHeight ?? this.alignToInputMultiple(height);

    if (width > tensorWidth || height > tensorHeight) {
      throw new Error(`Source tile ${width}x${height} exceeds model input ${tensorWidth}x${tensorHeight}`);
    }

    // Reuse canvas if same size, otherwise create new one
    if (!this.preprocessCanvas || this.preprocessCanvas.width !== tensorWidth || this.preprocessCanvas.height !== tensorHeight) {
      this.preprocessCanvas = new OffscreenCanvas(tensorWidth, tensorHeight);
      this.preprocessCtx = this.preprocessCanvas.getContext('2d', { willReadFrequently: true })!;
    }

    // Draw source or source tile to canvas
    if (tensorWidth !== width || tensorHeight !== height) {
      this.preprocessCtx!.clearRect(0, 0, tensorWidth, tensorHeight);
    }
    this.preprocessCtx!.drawImage(source, sx, sy, width, height, 0, 0, width, height);
    this.padCanvasEdges(width, height, tensorWidth, tensorHeight);

    // Get pixel data
    const imageData = this.preprocessCtx!.getImageData(0, 0, tensorWidth, tensorHeight);
    const pixels = imageData.data;
    const planeSize = tensorHeight * tensorWidth;
    const tensorLength = 3 * planeSize;

    if (this.useFloat16) {
      if (!this.inputFloat16 || this.inputFloat16.length !== tensorLength) {
        this.inputFloat16 = new Uint16Array(tensorLength);
      }

      for (let i = 0, p = 0; i < planeSize; i++, p += 4) {
        this.inputFloat16[i] = BYTE_TO_FLOAT16[pixels[p]];
        this.inputFloat16[planeSize + i] = BYTE_TO_FLOAT16[pixels[p + 1]];
        this.inputFloat16[2 * planeSize + i] = BYTE_TO_FLOAT16[pixels[p + 2]];
      }

      return {
        tensor: new ort.Tensor('float16', this.inputFloat16, [1, 3, tensorHeight, tensorWidth]),
        width,
        height
      };
    }

    if (!this.inputFloat32 || this.inputFloat32.length !== tensorLength) {
      this.inputFloat32 = new Float32Array(tensorLength);
    }

    for (let i = 0, p = 0; i < planeSize; i++, p += 4) {
      this.inputFloat32[i] = BYTE_TO_FLOAT32[pixels[p]];
      this.inputFloat32[planeSize + i] = BYTE_TO_FLOAT32[pixels[p + 1]];
      this.inputFloat32[2 * planeSize + i] = BYTE_TO_FLOAT32[pixels[p + 2]];
    }

    return {
      tensor: new ort.Tensor('float32', this.inputFloat32, [1, 3, tensorHeight, tensorWidth]),
      width,
      height
    };
  }

  private alignToInputMultiple(value: number): number {
    const multiple = this.config.inputMultiple || 1;
    if (multiple <= 1) return value;
    return Math.ceil(value / multiple) * multiple;
  }

  private padCanvasEdges(
    width: number,
    height: number,
    tensorWidth: number,
    tensorHeight: number
  ): void {
    if (!this.preprocessCanvas || !this.preprocessCtx || (width === tensorWidth && height === tensorHeight)) {
      return;
    }

    if (tensorWidth > width) {
      this.preprocessCtx.drawImage(
        this.preprocessCanvas,
        width - 1,
        0,
        1,
        height,
        width,
        0,
        tensorWidth - width,
        height
      );
    }

    if (tensorHeight > height) {
      this.preprocessCtx.drawImage(
        this.preprocessCanvas,
        0,
        height - 1,
        tensorWidth,
        1,
        0,
        height,
        tensorWidth,
        tensorHeight - height
      );
    }
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
    const outputHeight = (output.dims[2] as number) || inputHeight * this.config.scale;
    const outputWidth = (output.dims[3] as number) || inputWidth * this.config.scale;
    const planeSize = outputHeight * outputWidth;

    if (!this.outputImageData ||
        this.outputImageData.width !== outputWidth ||
        this.outputImageData.height !== outputHeight) {
      this.outputImageData = new ImageData(outputWidth, outputHeight);
    }

    const pixels = this.outputImageData.data;

    if (this.useFloat16) {
      const data = output.data as Float32Array | Uint16Array;

      if (data instanceof Uint16Array) {
        for (let i = 0, p = 0; i < planeSize; i++, p += 4) {
          pixels[p] = FLOAT16_TO_BYTE[data[i]];
          pixels[p + 1] = FLOAT16_TO_BYTE[data[planeSize + i]];
          pixels[p + 2] = FLOAT16_TO_BYTE[data[2 * planeSize + i]];
          pixels[p + 3] = 255;
        }
      } else {
        for (let i = 0, p = 0; i < planeSize; i++, p += 4) {
          const r = data[i];
          const g = data[planeSize + i];
          const b = data[2 * planeSize + i];

          pixels[p] = r <= 0 ? 0 : r >= 1 ? 255 : (r * 255 + 0.5) | 0;
          pixels[p + 1] = g <= 0 ? 0 : g >= 1 ? 255 : (g * 255 + 0.5) | 0;
          pixels[p + 2] = b <= 0 ? 0 : b >= 1 ? 255 : (b * 255 + 0.5) | 0;
          pixels[p + 3] = 255;
        }
      }
    } else {
      const data = output.data as Float32Array;
      for (let i = 0, p = 0; i < planeSize; i++, p += 4) {
        const r = data[i];
        const g = data[planeSize + i];
        const b = data[2 * planeSize + i];

        pixels[p] = r <= 0 ? 0 : r >= 1 ? 255 : (r * 255 + 0.5) | 0;
        pixels[p + 1] = g <= 0 ? 0 : g >= 1 ? 255 : (g * 255 + 0.5) | 0;
        pixels[p + 2] = b <= 0 ? 0 : b >= 1 ? 255 : (b * 255 + 0.5) | 0;
        pixels[p + 3] = 255;
      }
    }

    return this.outputImageData;
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
  async upscale(source: UpscaleSource): Promise<void> {
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
    // Use overlap to ensure seamless blending
    const overlap = Math.min(tilePadding * 2, tileSize - 1);
    const step = tileSize - overlap;
    const tilesX = inputWidth <= tileSize ? 1 : Math.ceil((inputWidth - overlap) / step);
    const tilesY = inputHeight <= tileSize ? 1 : Math.ceil((inputHeight - overlap) / step);

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        let tensor: ort.Tensor | null = null;
        let output: ort.Tensor | null = null;

        try {
          // Calculate source position (where to extract from input)
          const srcX = tx * step;
          const srcY = ty * step;

          // Clamp to input boundaries
          const actualSrcX = inputWidth <= tileSize ? 0 : Math.min(srcX, inputWidth - tileSize);
          const actualSrcY = inputHeight <= tileSize ? 0 : Math.min(srcY, inputHeight - tileSize);

          // Handle small images or last tiles
          const srcW = Math.min(tileSize, inputWidth - actualSrcX);
          const srcH = Math.min(tileSize, inputHeight - actualSrcY);

          // Process tile directly from the source frame to avoid extra canvas/bitmap copies.
          const preprocessed = await this.preprocess(source, actualSrcX, actualSrcY, srcW, srcH);
          tensor = preprocessed.tensor;
          output = await this.upscaleTile(tensor);
          const outputImageData = this.postprocess(output, srcW, srcH);

          // Calculate destination position in output
          const dstX = actualSrcX * scale;
          const dstY = actualSrcY * scale;

          // For interior tiles, we only keep the center region (excluding overlap)
          // For edge tiles, we keep more
          const isLeftEdge = (actualSrcX === 0);
          const isTopEdge = (actualSrcY === 0);
          const isRightEdge = (actualSrcX + srcW >= inputWidth);
          const isBottomEdge = (actualSrcY + srcH >= inputHeight);

          // Calculate which region of the output tile to keep
          const keepStartX = isLeftEdge ? 0 : tilePadding * scale;
          const keepStartY = isTopEdge ? 0 : tilePadding * scale;
          const keepEndX = isRightEdge ? srcW * scale : (srcW - tilePadding) * scale;
          const keepEndY = isBottomEdge ? srcH * scale : (srcH - tilePadding) * scale;
          const keepW = keepEndX - keepStartX;
          const keepH = keepEndY - keepStartY;

          // putImageData applies dirtyX/dirtyY on top of dx/dy, so dx/dy must be
          // the full tile origin rather than the cropped region origin.
          this.ctx.putImageData(
            outputImageData,
            dstX,
            dstY,
            keepStartX,
            keepStartY,
            keepW,
            keepH
          );
        } finally {
          // Cleanup - always dispose even on error
          tensor?.dispose();
          output?.dispose();
        }
      }
    }
  }

  /**
   * Render a frame directly (simplified path for video processing).
   */
  async render(frame: UpscaleSource): Promise<void> {
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
    this.clearFrameResources();
    this.initialized = false;
  }

  clearFrameResources(): void {
    this.preprocessCanvas = null;
    this.preprocessCtx = null;
    this.inputFloat32 = null;
    this.inputFloat16 = null;
    this.outputImageData = null;
  }
}

export default Upscaler;
