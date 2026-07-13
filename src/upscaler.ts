/**
 * Unified upscaler module supporting Real-ESRGAN and Real-CUGAN models
 * using ONNX Runtime Web with WebGPU acceleration.
 *
 * Supports:
 * - Real-ESRGAN: anime_fast, anime_plus, general_fast, general_plus
 * - Real-CUGAN: 2x, 4x (with denoising support)
 */

import * as ort from 'onnxruntime-web';
import type { DenoiseLevel, FrameTiming, ModelType } from './types/worker-messages';
import { loadModel, type LoadedModel, type LoadProgressCallback } from './model-loader';
import { calculateTilePlan } from './tiling';
import { GPUFrameRenderer, type GPUFrameTile } from './webgpu-utils';

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

export type UpscaleTiming = Omit<FrameTiming, 'decodeMs' | 'audioMs' | 'encodeMs' | 'totalMs' | 'frames'>;

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
  private lastTiming: UpscaleTiming | null = null;
  private gpuRenderer: GPUFrameRenderer | null = null;

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
      let webGPUModel = loadedModel;
      if (this.useFloat16 && this.useWebGPU) {
        // Keep protobufjs and the graph-rewrite code out of the default
        // startup bundle. It is only needed for the experimental AnimeJaNai
        // WebGPU attempt; normal float32 models should not pay that cost.
        const { rewritePReluForWebGPU } = await import('./onnx-webgpu-compat');
        const rewritten = rewritePReluForWebGPU(loadedModel.data);
        if (rewritten.rewrittenNodes > 0) {
          console.log(`Rewrote ${rewritten.rewrittenNodes} PRelu nodes for WebGPU`);
          webGPUModel = { ...loadedModel, data: rewritten.data };
        }
      }

      // Float16 models are attempted on WebGPU too. If the graph or browser
      // cannot execute them, the validated session path falls back to WASM.
      const canTryWebGPU = this.useWebGPU;
      const providerCandidates: ('webgpu' | 'wasm')[] = canTryWebGPU
        ? ['webgpu', 'wasm']
        : ['wasm'];

      let lastError: unknown = null;

      for (const provider of providerCandidates) {
        try {
          onProgress?.(100, `Initializing model with ${provider.toUpperCase()}...`);
          this.session = await this.createValidatedSession(
            provider === 'webgpu' ? webGPUModel : loadedModel,
            provider
          );
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
    this.ctx = outputCanvas.getContext('2d', { alpha: false }) as OffscreenCanvasRenderingContext2D;

    // The GPU bridge currently supports dynamic float32 models only. Models
    // with fixed or padded input shapes stay on the verified CPU path until
    // their shader padding and compositing paths are validated.
    if (
      this.executionProvider === 'webgpu' &&
      !this.useFloat16 &&
      !this.config.inputWidth &&
      !this.config.inputHeight &&
      (this.config.inputMultiple || 1) <= 1
    ) {
      this.gpuRenderer = await GPUFrameRenderer.create(this.session, this.config.scale);
    }

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

    if (provider === 'webgpu') {
      sessionOptions.preferredOutputLocation = 'gpu-buffer';
    }

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

      const validationData = await this.getTensorData(output);
      if (this.useFloat16) {
        this.outputDataIsFloat32 = validationData instanceof Float32Array;
      }

      if (!this.hasUsableOutput(validationData)) {
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

  private hasUsableOutput(data: Float32Array | Uint16Array): boolean {
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
    this.gpuRenderer?.dispose();
    this.gpuRenderer = null;
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
  private async postprocess(
    output: ort.Tensor,
    inputWidth: number,
    inputHeight: number
  ): Promise<ImageData> {
    const outputHeight = (output.dims[2] as number) || inputHeight * this.config.scale;
    const outputWidth = (output.dims[3] as number) || inputWidth * this.config.scale;
    const planeSize = outputHeight * outputWidth;
    const data = await this.getTensorData(output);

    if (!this.outputImageData ||
        this.outputImageData.width !== outputWidth ||
        this.outputImageData.height !== outputHeight) {
      this.outputImageData = new ImageData(outputWidth, outputHeight);
    }

    const pixels = this.outputImageData.data;

    if (this.useFloat16) {
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

  private async getTensorData(output: ort.Tensor): Promise<Float32Array | Uint16Array> {
    if (output.location !== 'cpu') {
      return await output.getData() as Float32Array | Uint16Array;
    }
    return output.data as Float32Array | Uint16Array;
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
    const timing: UpscaleTiming = {
      preprocessMs: 0,
      inferenceMs: 0,
      postprocessMs: 0,
      canvasMs: 0,
      tileCount: 0,
      inputPixels: inputWidth * inputHeight,
      inferredPixels: 0,
    };
    const now = () => typeof performance !== 'undefined' ? performance.now() : Date.now();

    const tilePlan = calculateTilePlan(inputWidth, inputHeight, tileSize, tilePadding);
    const tilesX = tilePlan.x.count;
    const tilesY = tilePlan.y.count;
    const getTile = (tx: number, ty: number): GPUFrameTile => {
      const sourceX = Math.min(
        tx * tilePlan.x.step,
        Math.max(0, inputWidth - tilePlan.x.tileSize)
      );
      const sourceY = Math.min(
        ty * tilePlan.y.step,
        Math.max(0, inputHeight - tilePlan.y.tileSize)
      );
      const inputTileWidth = tilePlan.x.tileSize;
      const inputTileHeight = tilePlan.y.tileSize;
      const effectivePaddingX = Math.min(tilePadding, Math.floor(tilePlan.overlap / 2));
      const effectivePaddingY = Math.min(tilePadding, Math.floor(tilePlan.overlap / 2));
      const isLeftEdge = sourceX === 0;
      const isTopEdge = sourceY === 0;
      const isRightEdge = sourceX + inputTileWidth >= inputWidth;
      const isBottomEdge = sourceY + inputTileHeight >= inputHeight;
      const keepStartX = isLeftEdge ? 0 : effectivePaddingX * scale;
      const keepStartY = isTopEdge ? 0 : effectivePaddingY * scale;
      const keepEndX = isRightEdge
        ? inputTileWidth * scale
        : (inputTileWidth - effectivePaddingX) * scale;
      const keepEndY = isBottomEdge
        ? inputTileHeight * scale
        : (inputTileHeight - effectivePaddingY) * scale;

      return {
        sourceX,
        sourceY,
        inputWidth: inputTileWidth,
        inputHeight: inputTileHeight,
        destinationX: sourceX * scale,
        destinationY: sourceY * scale,
        keepStartX,
        keepStartY,
        keepWidth: keepEndX - keepStartX,
        keepHeight: keepEndY - keepStartY,
      };
    };

    if (this.gpuRenderer) {
      const gpuTiles: GPUFrameTile[] = [];
      for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
          gpuTiles.push(getTile(tx, ty));
        }
      }

      try {
        await this.gpuRenderer.render(
          source,
          this.canvas,
          outputWidth,
          outputHeight,
          gpuTiles
        );
        const gpuTiming = this.gpuRenderer.getLastTiming();
        timing.preprocessMs = gpuTiming.preprocessMs;
        timing.inferenceMs = gpuTiming.inferenceMs;
        timing.postprocessMs = gpuTiming.postprocessMs;
        timing.canvasMs = gpuTiming.canvasMs;
        timing.tileCount = gpuTiles.length;
        timing.inferredPixels = gpuTiles.reduce(
          (total, tile) => total + tile.inputWidth * tile.inputHeight,
          0
        );
        this.lastTiming = timing;
        return;
      } catch (error) {
        console.warn('GPU tiled render failed; falling back to CPU path:', error);
        this.gpuRenderer.dispose();
        this.gpuRenderer = null;
      }
    }

    // For small images, process in one go
    if (inputWidth <= tileSize && inputHeight <= tileSize) {
      let tensor: ort.Tensor | null = null;
      let output: ort.Tensor | null = null;
      try {
        const preprocessStarted = now();
        const preprocessed = await this.preprocess(source);
        timing.preprocessMs += now() - preprocessStarted;
        tensor = preprocessed.tensor;
        const inferenceStarted = now();
        output = await this.upscaleTile(tensor);
        timing.inferenceMs += now() - inferenceStarted;
        timing.inferredPixels += inputWidth * inputHeight;
        const postprocessStarted = now();
        const imageData = await this.postprocess(output, inputWidth, inputHeight);
        timing.postprocessMs += now() - postprocessStarted;
        const canvasStarted = now();
        this.ctx.putImageData(imageData, 0, 0);
        timing.canvasMs += now() - canvasStarted;
        timing.tileCount = 1;
      } finally {
        tensor?.dispose();
        output?.dispose();
      }
      this.lastTiming = timing;
      return;
    }

    // Tiled processing for larger images. Use one stable, adaptive tile shape
    // per axis so remainder tiles do not get clamped back to full size.
    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        let tensor: ort.Tensor | null = null;
        let output: ort.Tensor | null = null;

        try {
          const tile = getTile(tx, ty);
          const actualSrcX = tile.sourceX;
          const actualSrcY = tile.sourceY;
          const srcW = tile.inputWidth;
          const srcH = tile.inputHeight;

          // Process tile directly from the source frame to avoid extra canvas/bitmap copies.
          const preprocessStarted = now();
          const preprocessed = await this.preprocess(source, actualSrcX, actualSrcY, srcW, srcH);
          timing.preprocessMs += now() - preprocessStarted;
          tensor = preprocessed.tensor;
          const inferenceStarted = now();
          output = await this.upscaleTile(tensor);
          timing.inferenceMs += now() - inferenceStarted;
          timing.inferredPixels += srcW * srcH;
          const postprocessStarted = now();
          const outputImageData = await this.postprocess(output, srcW, srcH);
          timing.postprocessMs += now() - postprocessStarted;

          // Calculate destination position in output
          const dstX = actualSrcX * scale;
          const dstY = actualSrcY * scale;

          // Keep exactly the same overlap crop as the GPU compositor. The
          // effective padding is clamped by the tile planner for tiny/fixed
          // model shapes, so it can never produce a negative dirty rectangle.
          const keepStartX = tile.keepStartX;
          const keepStartY = tile.keepStartY;
          const keepW = tile.keepWidth;
          const keepH = tile.keepHeight;

          // putImageData applies dirtyX/dirtyY on top of dx/dy, so dx/dy must be
          // the full tile origin rather than the cropped region origin.
          const canvasStarted = now();
          this.ctx.putImageData(
            outputImageData,
            dstX,
            dstY,
            keepStartX,
            keepStartY,
            keepW,
            keepH
          );
          timing.canvasMs += now() - canvasStarted;
          timing.tileCount += 1;
        } finally {
          // Cleanup - always dispose even on error
          tensor?.dispose();
          output?.dispose();
        }
      }
    }

    this.lastTiming = timing;
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

  getLastTiming(): UpscaleTiming | null {
    return this.lastTiming ? { ...this.lastTiming } : null;
  }

  getExecutionProvider(): 'webgpu' | 'wasm' {
    return this.executionProvider;
  }

  isUsingGPUPath(): boolean {
    return this.gpuRenderer !== null;
  }

  /**
   * Dispose of resources.
   */
  async dispose(): Promise<void> {
    this.gpuRenderer?.dispose();
    this.gpuRenderer = null;
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
    this.lastTiming = null;
  }
}

export default Upscaler;
