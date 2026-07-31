/**
 * WebGPU utilities for zero-copy video frame processing.
 *
 * Provides compute pipelines for converting between:
 * - RGBA textures (from VideoFrame)
 * - NCHW buffers (for ONNX Runtime)
 */

import * as ort from 'onnxruntime-web';

// Import shaders as raw strings (webpack will handle this)
import preprocessShaderF32 from './shaders/preprocess.wgsl';
import postprocessShaderF32 from './shaders/postprocess.wgsl';
import preprocessShaderF16 from './shaders/preprocess-f16.wgsl';
import postprocessShaderF16 from './shaders/postprocess-f16.wgsl';
import compositeShader from './shaders/composite.wgsl';
import { calculateTileCopyRegion } from './tiling';

export interface WebGPUContext {
  device: GPUDevice;
  canvasFormat: GPUTextureFormat;
  pipelines: {
    preprocessF32: GPUComputePipeline;
    postprocessF32: GPUComputePipeline;
    preprocessF16: GPUComputePipeline;
    postprocessF16: GPUComputePipeline;
    composite: GPURenderPipeline;
  };
  bindGroupLayouts: {
    preprocess: GPUBindGroupLayout;
    postprocess: GPUBindGroupLayout;
    composite: GPUBindGroupLayout;
  };
}

export interface GPUBufferPool {
  // Input side
  inputTexture: GPUTexture;
  inputBuffer: GPUBuffer;

  // Output side
  outputBuffer: GPUBuffer;
  outputTexture: GPUTexture;

  // Uniform buffers for shader params
  preprocessParams: GPUBuffer;
  postprocessParams: GPUBuffer;

  // Dimensions
  inputWidth: number;
  inputHeight: number;
  outputWidth: number;
  outputHeight: number;
  scale: number;
  useFloat16: boolean;
}

export interface GPUFrameTile {
  sourceX: number;
  sourceY: number;
  /** Valid source pixels represented by this tile. */
  inputWidth: number;
  inputHeight: number;
  /** Model tensor shape after fixed-shape/input-multiple edge padding. */
  tensorWidth: number;
  tensorHeight: number;
  /** Absolute output origin before the kept-region crop is applied. */
  destinationX: number;
  destinationY: number;
  keepStartX: number;
  keepStartY: number;
  keepWidth: number;
  keepHeight: number;
}

export interface GPUFrameTiming {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  gpuWaitMs: number;
  gpuTimestampMs: number;
  canvasMs: number;
}

/**
 * Tiled WebGPU bridge for float32 and packed-float16 ONNX models.
 *
 * The renderer deliberately owns a separate WebGPU canvas. Preview renders
 * mirror once to the public 2D canvas; matching-size recording jobs can lease
 * the GPU canvas directly to CanvasSource and skip that readback entirely.
 */
export class GPUFrameRenderer {
  private readonly device: GPUDevice;
  private readonly session: ort.InferenceSession;
  private readonly scale: number;
  private readonly context: WebGPUContext;
  private readonly inputFloat16: boolean;
  private readonly outputFloat16: boolean;
  private readonly gpuTimestampsEnabled: boolean;
  private gpuCanvas: OffscreenCanvas | null = null;
  private gpuCanvasContext: GPUCanvasContext | null = null;
  private directOutputLease: { width: number; height: number } | null = null;
  private inputTexture: GPUTexture | null = null;
  private inputBuffer: GPUBuffer | null = null;
  private outputBuffer: GPUBuffer | null = null;
  private outputTexture: GPUTexture | null = null;
  private compositeTexture: GPUTexture | null = null;
  private inputTextureView: GPUTextureView | null = null;
  private outputTextureView: GPUTextureView | null = null;
  private compositeTextureView: GPUTextureView | null = null;
  private preprocessBindGroup: GPUBindGroup | null = null;
  private postprocessBindGroup: GPUBindGroup | null = null;
  private compositeBindGroup: GPUBindGroup | null = null;
  private inputTensor: ort.Tensor | null = null;
  private outputTensor: ort.Tensor | null = null;
  private preprocessParams: GPUBuffer | null = null;
  private postprocessParams: GPUBuffer | null = null;
  private readonly preprocessParamsData = new Uint32Array(8);
  private timestampQuerySet: GPUQuerySet | null = null;
  private timestampResolveBuffer: GPUBuffer | null = null;
  private timestampReadbackBuffer: GPUBuffer | null = null;
  private tileInputWidth = 0;
  private tileInputHeight = 0;
  private tileOutputWidth = 0;
  private tileOutputHeight = 0;
  private sourceWidth = 0;
  private sourceHeight = 0;
  private outputWidth = 0;
  private outputHeight = 0;
  private lastTiming: GPUFrameTiming = {
    preprocessMs: 0,
    inferenceMs: 0,
    postprocessMs: 0,
    gpuWaitMs: 0,
    gpuTimestampMs: 0,
    canvasMs: 0,
  };

  private constructor(
    device: GPUDevice,
    session: ort.InferenceSession,
    scale: number,
    context: WebGPUContext,
    inputFloat16: boolean,
    outputFloat16: boolean,
    gpuTimestampsEnabled: boolean
  ) {
    this.device = device;
    this.session = session;
    this.scale = scale;
    this.context = context;
    this.inputFloat16 = inputFloat16;
    this.outputFloat16 = outputFloat16;
    this.gpuTimestampsEnabled = gpuTimestampsEnabled;
    this.initializeTimestampQueries();
  }

  static async create(
    session: ort.InferenceSession,
    scale: number,
    inputFloat16: boolean,
    outputFloat16: boolean,
    gpuTimestampsEnabled = false
  ): Promise<GPUFrameRenderer | null> {
    try {
      const device = await (ort.env as any).webgpu?.device as GPUDevice | undefined;
      if (!device) return null;
      const context = await initWebGPUContext(device);
      return new GPUFrameRenderer(
        device,
        session,
        scale,
        context,
        inputFloat16,
        outputFloat16,
        gpuTimestampsEnabled
      );
    } catch (error) {
      console.warn('GPU frame renderer unavailable:', error);
      return null;
    }
  }

  private ensureResources(
    sourceWidth: number,
    sourceHeight: number,
    outputWidth: number,
    outputHeight: number,
    tileTensorWidth: number,
    tileTensorHeight: number
  ): void {
    const tileOutputWidth = tileTensorWidth * this.scale;
    const tileOutputHeight = tileTensorHeight * this.scale;
    const dimensionsChanged =
      this.sourceWidth !== sourceWidth ||
      this.sourceHeight !== sourceHeight ||
      this.outputWidth !== outputWidth ||
      this.outputHeight !== outputHeight ||
      this.tileInputWidth !== tileTensorWidth ||
      this.tileInputHeight !== tileTensorHeight;

    if (!dimensionsChanged) return;

    this.destroyTileResources();

    this.outputWidth = outputWidth;
    this.outputHeight = outputHeight;
    this.sourceWidth = sourceWidth;
    this.sourceHeight = sourceHeight;
    this.tileInputWidth = tileTensorWidth;
    this.tileInputHeight = tileTensorHeight;
    this.tileOutputWidth = tileOutputWidth;
    this.tileOutputHeight = tileOutputHeight;

    this.ensurePresentationCanvas(outputWidth, outputHeight);

    this.inputTexture = this.device.createTexture({
      // Import the decoded source once per frame; tile origins are handled by
      // the preprocess uniform rather than repeated external-image copies.
      size: [sourceWidth, sourceHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        // Dawn's copyExternalImageToTexture validation requires external-image
        // destinations to be renderable as well as copy destinations.
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const inputElementBytes = this.inputFloat16 ? 2 : 4;
    const outputElementBytes = this.outputFloat16 ? 2 : 4;
    this.inputBuffer = this.device.createBuffer({
      size: alignTo16(3 * tileTensorWidth * tileTensorHeight * inputElementBytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.outputBuffer = this.device.createBuffer({
      size: alignTo16(3 * tileOutputWidth * tileOutputHeight * outputElementBytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.outputTexture = this.device.createTexture({
      size: [tileOutputWidth, tileOutputHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.compositeTexture = this.device.createTexture({
      size: [outputWidth, outputHeight],
      format: 'rgba8unorm',
      // Chromium's Metal backend validates the destination as a renderable
      // texture when it is later sampled by the presentation pass.
      usage: GPUTextureUsage.COPY_DST |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.preprocessParams = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.postprocessParams = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      this.postprocessParams,
      0,
      new Uint32Array([tileOutputWidth, tileOutputHeight])
    );

    // These bindings are stable for the lifetime of the tile resources.
    // Reusing them avoids rebuilding WebGPU object graphs for every tile.
    this.inputTextureView = this.inputTexture.createView();
    this.outputTextureView = this.outputTexture.createView();
    this.compositeTextureView = this.compositeTexture.createView();
    this.preprocessBindGroup = this.device.createBindGroup({
      layout: this.context.bindGroupLayouts.preprocess,
      entries: [
        { binding: 0, resource: this.inputTextureView },
        { binding: 1, resource: { buffer: this.inputBuffer } },
        { binding: 2, resource: { buffer: this.preprocessParams } },
      ],
    });
    const inputDataType = this.inputFloat16 ? 'float16' : 'float32';
    const outputDataType = this.outputFloat16 ? 'float16' : 'float32';
    this.inputTensor = ort.Tensor.fromGpuBuffer(this.inputBuffer, {
      dataType: inputDataType,
      dims: [1, 3, tileTensorHeight, tileTensorWidth],
    });
    this.outputTensor = ort.Tensor.fromGpuBuffer(this.outputBuffer, {
      dataType: outputDataType,
      dims: [1, 3, tileOutputHeight, tileOutputWidth],
    });
    this.postprocessBindGroup = this.device.createBindGroup({
      layout: this.context.bindGroupLayouts.postprocess,
      entries: [
        { binding: 0, resource: { buffer: this.outputBuffer } },
        { binding: 1, resource: this.outputTextureView },
        { binding: 2, resource: { buffer: this.postprocessParams } },
      ],
    });
    this.compositeBindGroup = this.device.createBindGroup({
      layout: this.context.bindGroupLayouts.composite,
      entries: [{ binding: 0, resource: this.compositeTextureView }],
    });
  }

  private ensurePresentationCanvas(outputWidth: number, outputHeight: number): void {
    if (
      this.gpuCanvas &&
      this.gpuCanvasContext &&
      this.gpuCanvas.width === outputWidth &&
      this.gpuCanvas.height === outputHeight
    ) {
      return;
    }

    if (this.directOutputLease) {
      throw new Error(
        `Direct GPU output is leased at ${this.directOutputLease.width}x${this.directOutputLease.height}; ` +
        `cannot replace it with ${outputWidth}x${outputHeight}`
      );
    }

    this.gpuCanvas = new OffscreenCanvas(outputWidth, outputHeight);
    // OffscreenCanvas.getContext() is typed as a union of all rendering
    // contexts; the requested context string narrows this at runtime.
    this.gpuCanvasContext = this.gpuCanvas.getContext('webgpu') as unknown as GPUCanvasContext | null;
    if (!this.gpuCanvasContext) {
      throw new Error('Unable to create WebGPU output canvas');
    }
    this.gpuCanvasContext.configure({
      device: this.device,
      format: this.context.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      alphaMode: 'opaque',
    });
  }

  getOutputCanvas(outputWidth: number, outputHeight: number): OffscreenCanvas {
    this.ensurePresentationCanvas(outputWidth, outputHeight);
    this.directOutputLease = { width: outputWidth, height: outputHeight };
    return this.gpuCanvas!;
  }

  releaseOutputCanvas(): void {
    this.directOutputLease = null;
  }

  private initializeTimestampQueries(): void {
    if (!this.gpuTimestampsEnabled || !this.device.features.has('timestamp-query')) {
      return;
    }

    try {
      this.timestampQuerySet = this.device.createQuerySet({ type: 'timestamp', count: 4 });
      this.timestampResolveBuffer = this.device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.timestampReadbackBuffer = this.device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    } catch (error) {
      console.debug('GPU timestamp queries unavailable:', error);
      this.timestampQuerySet?.destroy();
      this.timestampResolveBuffer?.destroy();
      this.timestampReadbackBuffer?.destroy();
      this.timestampQuerySet = null;
      this.timestampResolveBuffer = null;
      this.timestampReadbackBuffer = null;
    }
  }

  private async readTimestampMs(): Promise<number> {
    if (!this.timestampReadbackBuffer) return 0;

    try {
      await this.timestampReadbackBuffer.mapAsync(GPUMapMode.READ);
      const values = new BigUint64Array(this.timestampReadbackBuffer.getMappedRange());
      const postprocessNs = values[1] - values[0];
      const presentNs = values[3] - values[2];
      this.timestampReadbackBuffer.unmap();
      return Number(postprocessNs + presentNs) / 1_000_000;
    } catch (error) {
      console.debug('GPU timestamp readback unavailable:', error);
      try {
        this.timestampReadbackBuffer.unmap();
      } catch {
        // Ignore an already-unmapped buffer while falling back to fence timing.
      }
      return 0;
    }
  }

  async render(
    source: ImageBitmap | VideoFrame | OffscreenCanvas,
    outputCanvas: OffscreenCanvas,
    sourceWidth: number,
    sourceHeight: number,
    outputWidth: number,
    outputHeight: number,
    tiles: readonly GPUFrameTile[],
    mirrorOutput = true
  ): Promise<void> {
    // WebGPU validation failures are otherwise reported only as uncaptured
    // console events and can silently encode an empty/stale frame. Scope the
    // whole frame so the caller can fall back (or abort a leased direct
    // surface) instead of accepting corrupted output.
    this.device.pushErrorScope('validation');
    let renderFailure: unknown = null;

    try {
      await this.renderUnchecked(
        source,
        outputCanvas,
        sourceWidth,
        sourceHeight,
        outputWidth,
        outputHeight,
        tiles,
        mirrorOutput,
      );
    } catch (error) {
      renderFailure = error;
    }

    let validationFailure: GPUError | null = null;
    try {
      validationFailure = await this.device.popErrorScope();
    } catch (error) {
      if (!renderFailure) renderFailure = error;
    }

    if (renderFailure) throw renderFailure;
    if (validationFailure) {
      throw new Error(`WebGPU frame validation failed: ${validationFailure.message}`);
    }
  }

  private async renderUnchecked(
    source: ImageBitmap | VideoFrame | OffscreenCanvas,
    outputCanvas: OffscreenCanvas,
    sourceWidth: number,
    sourceHeight: number,
    outputWidth: number,
    outputHeight: number,
    tiles: readonly GPUFrameTile[],
    mirrorOutput: boolean,
  ): Promise<void> {
    if (tiles.length === 0) return;
    const now = () => typeof performance !== 'undefined' ? performance.now() : Date.now();
    const timing: GPUFrameTiming = {
      preprocessMs: 0,
      inferenceMs: 0,
      postprocessMs: 0,
      gpuWaitMs: 0,
      gpuTimestampMs: 0,
      canvasMs: 0,
    };
    const firstTile = tiles[0];
    this.ensureResources(
      sourceWidth,
      sourceHeight,
      outputWidth,
      outputHeight,
      firstTile.tensorWidth,
      firstTile.tensorHeight
    );

    if (!this.gpuCanvasContext || !this.inputTexture || !this.inputBuffer ||
        !this.outputBuffer || !this.outputTexture || !this.compositeTexture ||
        !this.preprocessParams || !this.postprocessParams || !this.inputTensor ||
        !this.outputTensor || !this.postprocessBindGroup || !this.compositeBindGroup) {
      throw new Error('GPU frame renderer resources are unavailable');
    }

    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];

    const sourceUploadStarted = now();
    this.device.queue.copyExternalImageToTexture(
      { source },
      { texture: this.inputTexture },
      [sourceWidth, sourceHeight]
    );
    timing.preprocessMs += now() - sourceUploadStarted;

    for (let tileIndex = 0; tileIndex < tiles.length; tileIndex++) {
      const tile = tiles[tileIndex];
      if (tile.tensorWidth !== this.tileInputWidth || tile.tensorHeight !== this.tileInputHeight) {
        throw new Error('GPU frame renderer requires one tensor shape per frame');
      }
      if (tile.inputWidth <= 0 || tile.inputHeight <= 0 ||
          tile.inputWidth > tile.tensorWidth || tile.inputHeight > tile.tensorHeight) {
        throw new Error('GPU frame tile has invalid source/tensor dimensions');
      }

      const preprocessStarted = now();
      this.preprocessParamsData.set([
        tile.tensorWidth,
        tile.tensorHeight,
        tile.sourceX,
        tile.sourceY,
        tile.inputWidth,
        tile.inputHeight,
      ]);
      this.device.queue.writeBuffer(this.preprocessParams, 0, this.preprocessParamsData);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(
        this.inputFloat16
          ? this.context.pipelines.preprocessF16
          : this.context.pipelines.preprocessF32
      );
      pass.setBindGroup(0, this.preprocessBindGroup!);
      if (this.inputFloat16) {
        const packedValueCount = Math.ceil(3 * tile.tensorWidth * tile.tensorHeight / 2);
        pass.dispatchWorkgroups(Math.ceil(packedValueCount / 256));
      } else {
        pass.dispatchWorkgroups(
          Math.ceil(tile.tensorWidth / 16),
          Math.ceil(tile.tensorHeight / 16)
        );
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      timing.preprocessMs += now() - preprocessStarted;

      const inferenceStarted = now();
      const outputs = await this.session.run(
        { [inputName]: this.inputTensor },
        { [outputName]: this.outputTensor }
      );
      timing.inferenceMs += now() - inferenceStarted;
      const output = outputs[outputName];
      if (!output || output.location !== 'gpu-buffer') {
        throw new Error('ONNX Runtime returned a CPU output for GPU rendering');
      }
      if (output.gpuBuffer !== this.outputBuffer) {
        throw new Error('ONNX Runtime did not bind the preallocated GPU output buffer');
      }
      const modelOutputWidth = Number(output.dims[3]);
      const modelOutputHeight = Number(output.dims[2]);
      if (modelOutputWidth !== this.tileOutputWidth || modelOutputHeight !== this.tileOutputHeight) {
        throw new Error('ONNX output shape does not match GPU tile resources');
      }

      const postprocessStarted = now();
      const postprocessEncoder = this.device.createCommandEncoder();
      const postprocessPass = postprocessEncoder.beginComputePass(
        this.timestampQuerySet && tileIndex === 0
          ? {
            timestampWrites: {
              querySet: this.timestampQuerySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          }
          : undefined
      );
      postprocessPass.setPipeline(
        this.outputFloat16
          ? this.context.pipelines.postprocessF16
          : this.context.pipelines.postprocessF32
      );
      postprocessPass.setBindGroup(0, this.postprocessBindGroup);
      postprocessPass.dispatchWorkgroups(
        Math.ceil(this.tileOutputWidth / 16),
        Math.ceil(this.tileOutputHeight / 16)
      );
      postprocessPass.end();
      const copyRegion = calculateTileCopyRegion({
        tileOutputX: tile.destinationX,
        tileOutputY: tile.destinationY,
        cropX: tile.keepStartX,
        cropY: tile.keepStartY,
        width: tile.keepWidth,
        height: tile.keepHeight,
      });
      postprocessEncoder.copyTextureToTexture(
        {
          texture: this.outputTexture,
          origin: { x: copyRegion.sourceX, y: copyRegion.sourceY },
        },
        {
          texture: this.compositeTexture,
          origin: { x: copyRegion.destinationX, y: copyRegion.destinationY },
        },
        [copyRegion.width, copyRegion.height]
      );

      if (tileIndex === tiles.length - 1) {
        // Canvas textures can expire across an await/rendering update. Acquire
        // immediately before the synchronous encode+submit sequence instead
        // of retaining one across the ORT session.run() calls above.
        const presentTexture = this.gpuCanvasContext.getCurrentTexture();
        const presentPass = postprocessEncoder.beginRenderPass({
          colorAttachments: [{
            view: presentTexture.createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
          ...(this.timestampQuerySet
            ? {
              timestampWrites: {
                querySet: this.timestampQuerySet,
                beginningOfPassWriteIndex: 2,
                endOfPassWriteIndex: 3,
              },
            }
            : {}),
        });
        presentPass.setPipeline(this.context.pipelines.composite);
        presentPass.setBindGroup(0, this.compositeBindGroup);
        presentPass.draw(3);
        presentPass.end();

        if (this.timestampQuerySet && this.timestampResolveBuffer && this.timestampReadbackBuffer) {
          postprocessEncoder.resolveQuerySet(
            this.timestampQuerySet,
            0,
            4,
            this.timestampResolveBuffer,
            0
          );
          postprocessEncoder.copyBufferToBuffer(
            this.timestampResolveBuffer,
            0,
            this.timestampReadbackBuffer,
            0,
            32
          );
        }
      }

      this.device.queue.submit([postprocessEncoder.finish()]);
      timing.postprocessMs += now() - postprocessStarted;
    }

    // One frame fence replaces the previous four-output allocation window and
    // per-window waits. Stable external input/output buffers are safe because
    // preprocess, ORT, postprocess, and presentation share one ordered queue.
    const fenceStarted = now();
    await this.device.queue.onSubmittedWorkDone();
    timing.gpuWaitMs = now() - fenceStarted;
    timing.gpuTimestampMs = await this.readTimestampMs();

    if (mirrorOutput) {
      const canvasStarted = now();
      const outputContext = outputCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
      if (!outputContext || !this.gpuCanvas) {
        throw new Error('Unable to mirror GPU output to the public canvas');
      }
      outputContext.drawImage(this.gpuCanvas, 0, 0, outputWidth, outputHeight);
      timing.canvasMs = now() - canvasStarted;
    }
    this.lastTiming = timing;
  }

  getLastTiming(): GPUFrameTiming {
    return { ...this.lastTiming };
  }

  dispose(): void {
    this.destroyTileResources();
    this.destroyTimestampResources();
    this.directOutputLease = null;
    this.gpuCanvas = null;
    this.gpuCanvasContext = null;
  }

  private destroyTimestampResources(): void {
    this.timestampQuerySet?.destroy();
    this.timestampResolveBuffer?.destroy();
    this.timestampReadbackBuffer?.destroy();
    this.timestampQuerySet = null;
    this.timestampResolveBuffer = null;
    this.timestampReadbackBuffer = null;
  }

  private destroyTileResources(): void {
    this.inputTensor?.dispose();
    this.outputTensor?.dispose();
    this.inputTensor = null;
    this.outputTensor = null;
    this.inputTexture?.destroy();
    this.inputBuffer?.destroy();
    this.outputBuffer?.destroy();
    this.outputTexture?.destroy();
    this.compositeTexture?.destroy();
    this.preprocessParams?.destroy();
    this.postprocessParams?.destroy();
    this.inputTexture = null;
    this.inputBuffer = null;
    this.outputBuffer = null;
    this.outputTexture = null;
    this.compositeTexture = null;
    this.inputTextureView = null;
    this.outputTextureView = null;
    this.compositeTextureView = null;
    this.preprocessBindGroup = null;
    this.postprocessBindGroup = null;
    this.compositeBindGroup = null;
    this.preprocessParams = null;
    this.postprocessParams = null;
  }
}

/**
 * Get the WebGPU device from ONNX Runtime.
 * Must be called AFTER at least one inference has run.
 */
export function getORTWebGPUDevice(): GPUDevice | null {
  // Access ONNX Runtime's WebGPU device
  const env = ort.env as any;
  if (env.webgpu?.device) {
    return env.webgpu.device as GPUDevice;
  }
  return null;
}

/**
 * Initialize WebGPU compute pipelines for pre/post processing.
 */
export async function initWebGPUContext(device: GPUDevice): Promise<WebGPUContext> {
  // Metal commonly prefers BGRA; using the browser's native swapchain format
  // avoids an implicit format conversion in the final presentation pass.
  const canvasFormat = typeof navigator !== 'undefined' &&
    typeof navigator.gpu?.getPreferredCanvasFormat === 'function'
    ? navigator.gpu.getPreferredCanvasFormat()
    : 'rgba8unorm';
  // Create bind group layouts
  const preprocessLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const postprocessLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const compositeLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    ],
  });

  // Create pipeline layouts
  const preprocessPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [preprocessLayout],
  });

  const postprocessPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [postprocessLayout],
  });

  const compositePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [compositeLayout],
  });

  // Create compute pipelines
  const preprocessF32 = device.createComputePipeline({
    layout: preprocessPipelineLayout,
    compute: {
      module: device.createShaderModule({ code: preprocessShaderF32 }),
      entryPoint: 'main',
    },
  });

  const postprocessF32 = device.createComputePipeline({
    layout: postprocessPipelineLayout,
    compute: {
      module: device.createShaderModule({ code: postprocessShaderF32 }),
      entryPoint: 'main',
    },
  });

  const preprocessF16 = device.createComputePipeline({
    layout: preprocessPipelineLayout,
    compute: {
      module: device.createShaderModule({ code: preprocessShaderF16 }),
      entryPoint: 'main',
    },
  });

  const postprocessF16 = device.createComputePipeline({
    layout: postprocessPipelineLayout,
    compute: {
      module: device.createShaderModule({ code: postprocessShaderF16 }),
      entryPoint: 'main',
    },
  });

  const composite = device.createRenderPipeline({
    layout: compositePipelineLayout,
    vertex: {
      module: device.createShaderModule({ code: compositeShader }),
      entryPoint: 'vertex',
    },
    fragment: {
      module: device.createShaderModule({ code: compositeShader }),
      entryPoint: 'fragment',
      targets: [{ format: canvasFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });

  return {
    device,
    canvasFormat,
    pipelines: {
      preprocessF32,
      postprocessF32,
      preprocessF16,
      postprocessF16,
      composite,
    },
    bindGroupLayouts: {
      preprocess: preprocessLayout,
      postprocess: postprocessLayout,
      composite: compositeLayout,
    },
  };
}

/**
 * Create a pool of GPU buffers for a specific resolution.
 */
export function createBufferPool(
  device: GPUDevice,
  inputWidth: number,
  inputHeight: number,
  scale: number,
  useFloat16: boolean
): GPUBufferPool {
  const outputWidth = inputWidth * scale;
  const outputHeight = inputHeight * scale;

  // Calculate buffer sizes
  const inputPixels = inputWidth * inputHeight;
  const outputPixels = outputWidth * outputHeight;

  // Packed float16 tensors use two scalar values per u32 word.
  const bytesPerValue = useFloat16 ? 2 : 4;
  const inputBufferSize = alignTo16(3 * inputPixels * bytesPerValue);
  const outputBufferSize = alignTo16(3 * outputPixels * bytesPerValue);

  // Input texture (receives VideoFrame)
  const inputTexture = device.createTexture({
    size: [inputWidth, inputHeight],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING |
           GPUTextureUsage.COPY_DST |
           GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Input buffer (NCHW data for ONNX)
  const inputBuffer = device.createBuffer({
    size: inputBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Output buffer (NCHW data from ONNX)
  const outputBuffer = device.createBuffer({
    size: outputBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Output texture (for canvas rendering)
  const outputTexture = device.createTexture({
    size: [outputWidth, outputHeight],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  // Uniform buffers for shader parameters
  const preprocessParams = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const postprocessParams = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Initialize uniform buffers
  device.queue.writeBuffer(
    preprocessParams,
    0,
    new Uint32Array([inputWidth, inputHeight, 0, 0, inputWidth, inputHeight, 0, 0])
  );
  device.queue.writeBuffer(postprocessParams, 0, new Uint32Array([outputWidth, outputHeight]));

  return {
    inputTexture,
    inputBuffer,
    outputBuffer,
    outputTexture,
    preprocessParams,
    postprocessParams,
    inputWidth,
    inputHeight,
    outputWidth,
    outputHeight,
    scale,
    useFloat16,
  };
}

/**
 * Run the preprocess compute shader.
 * Converts RGBA texture to NCHW buffer.
 */
export function runPreprocessShader(
  ctx: WebGPUContext,
  pool: GPUBufferPool
): void {
  const { device, pipelines, bindGroupLayouts } = ctx;
  const pipeline = pool.useFloat16 ? pipelines.preprocessF16 : pipelines.preprocessF32;

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayouts.preprocess,
    entries: [
      { binding: 0, resource: pool.inputTexture.createView() },
      { binding: 1, resource: { buffer: pool.inputBuffer } },
      { binding: 2, resource: { buffer: pool.preprocessParams } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();

  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);

  if (pool.useFloat16) {
    const packedValueCount = Math.ceil(3 * pool.inputWidth * pool.inputHeight / 2);
    passEncoder.dispatchWorkgroups(Math.ceil(packedValueCount / 256));
  } else {
    passEncoder.dispatchWorkgroups(
      Math.ceil(pool.inputWidth / 16),
      Math.ceil(pool.inputHeight / 16)
    );
  }

  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
}

/**
 * Run the postprocess compute shader.
 * Converts NCHW buffer to RGBA texture.
 */
export function runPostprocessShader(
  ctx: WebGPUContext,
  pool: GPUBufferPool
): void {
  const { device, pipelines, bindGroupLayouts } = ctx;
  const pipeline = pool.useFloat16 ? pipelines.postprocessF16 : pipelines.postprocessF32;

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayouts.postprocess,
    entries: [
      { binding: 0, resource: { buffer: pool.outputBuffer } },
      { binding: 1, resource: pool.outputTexture.createView() },
      { binding: 2, resource: { buffer: pool.postprocessParams } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();

  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);

  // Dispatch workgroups
  const workgroupsX = Math.ceil(pool.outputWidth / 16);
  const workgroupsY = Math.ceil(pool.outputHeight / 16);
  passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
}

/**
 * Import a VideoFrame into the input texture.
 */
export function importVideoFrame(
  device: GPUDevice,
  pool: GPUBufferPool,
  frame: VideoFrame
): void {
  device.queue.copyExternalImageToTexture(
    { source: frame },
    { texture: pool.inputTexture },
    [pool.inputWidth, pool.inputHeight]
  );
}

/**
 * Copy output texture to a canvas.
 */
export async function copyToCanvas(
  device: GPUDevice,
  pool: GPUBufferPool,
  canvas: OffscreenCanvas
): Promise<void> {
  // Get WebGPU context from canvas
  const canvasCtx = canvas.getContext('webgpu') as unknown as GPUCanvasContext;

  if (!canvasCtx) {
    throw new Error('Failed to get WebGPU context from canvas');
  }

  // Configure canvas context if not already configured
  canvasCtx.configure({
    device,
    format: 'rgba8unorm',
    alphaMode: 'premultiplied',
  });

  // Copy from output texture to canvas texture
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyTextureToTexture(
    { texture: pool.outputTexture },
    { texture: canvasCtx.getCurrentTexture() },
    [pool.outputWidth, pool.outputHeight]
  );
  device.queue.submit([commandEncoder.finish()]);
}

/**
 * Destroy buffer pool resources.
 */
export function destroyBufferPool(pool: GPUBufferPool): void {
  pool.inputTexture.destroy();
  pool.inputBuffer.destroy();
  pool.outputBuffer.destroy();
  pool.outputTexture.destroy();
  pool.preprocessParams.destroy();
  pool.postprocessParams.destroy();
}

/**
 * Align a size to 16 bytes (required for WebGPU buffers).
 */
function alignTo16(size: number): number {
  return Math.ceil(size / 16) * 16;
}
