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

export interface WebGPUContext {
  device: GPUDevice;
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
  inputWidth: number;
  inputHeight: number;
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
 * Tiled WebGPU bridge for float32 ONNX models.
 *
 * The renderer deliberately owns a separate WebGPU canvas. The public output
 * canvas remains a 2D canvas for preview and CanvasSource compatibility; only
 * one GPU-to-canvas draw is performed after all tiles are composited.
 */
export class GPUFrameRenderer {
  private readonly device: GPUDevice;
  private readonly session: ort.InferenceSession;
  private readonly scale: number;
  private readonly context: WebGPUContext;
  private readonly gpuTimestampsEnabled: boolean;
  private gpuCanvas: OffscreenCanvas | null = null;
  private gpuCanvasContext: GPUCanvasContext | null = null;
  private inputTexture: GPUTexture | null = null;
  private inputBuffer: GPUBuffer | null = null;
  private outputTexture: GPUTexture | null = null;
  private compositeTexture: GPUTexture | null = null;
  private inputTextureView: GPUTextureView | null = null;
  private outputTextureView: GPUTextureView | null = null;
  private compositeTextureView: GPUTextureView | null = null;
  private preprocessBindGroup: GPUBindGroup | null = null;
  private compositeBindGroup: GPUBindGroup | null = null;
  private preprocessParams: GPUBuffer | null = null;
  private postprocessParams: GPUBuffer | null = null;
  private timestampQuerySet: GPUQuerySet | null = null;
  private timestampResolveBuffer: GPUBuffer | null = null;
  private timestampReadbackBuffer: GPUBuffer | null = null;
  private tileInputWidth = 0;
  private tileInputHeight = 0;
  private tileOutputWidth = 0;
  private tileOutputHeight = 0;
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
    gpuTimestampsEnabled: boolean
  ) {
    this.device = device;
    this.session = session;
    this.scale = scale;
    this.context = context;
    this.gpuTimestampsEnabled = gpuTimestampsEnabled;
    this.initializeTimestampQueries();
  }

  static async create(
    session: ort.InferenceSession,
    scale: number,
    gpuTimestampsEnabled = false
  ): Promise<GPUFrameRenderer | null> {
    try {
      const device = await (ort.env as any).webgpu?.device as GPUDevice | undefined;
      if (!device) return null;
      const context = await initWebGPUContext(device);
      return new GPUFrameRenderer(device, session, scale, context, gpuTimestampsEnabled);
    } catch (error) {
      console.warn('GPU frame renderer unavailable:', error);
      return null;
    }
  }

  private ensureResources(
    outputWidth: number,
    outputHeight: number,
    tileInputWidth: number,
    tileInputHeight: number
  ): void {
    const tileOutputWidth = tileInputWidth * this.scale;
    const tileOutputHeight = tileInputHeight * this.scale;
    const dimensionsChanged =
      this.outputWidth !== outputWidth ||
      this.outputHeight !== outputHeight ||
      this.tileInputWidth !== tileInputWidth ||
      this.tileInputHeight !== tileInputHeight;

    if (!dimensionsChanged) return;

    this.destroyTileResources();

    this.outputWidth = outputWidth;
    this.outputHeight = outputHeight;
    this.tileInputWidth = tileInputWidth;
    this.tileInputHeight = tileInputHeight;
    this.tileOutputWidth = tileOutputWidth;
    this.tileOutputHeight = tileOutputHeight;

    this.gpuCanvas = new OffscreenCanvas(outputWidth, outputHeight);
    // OffscreenCanvas.getContext() is typed as a union of all rendering
    // contexts; the requested context string narrows this at runtime.
    this.gpuCanvasContext = this.gpuCanvas.getContext('webgpu') as unknown as GPUCanvasContext | null;
    if (!this.gpuCanvasContext) {
      throw new Error('Unable to create WebGPU output canvas');
    }
    this.gpuCanvasContext.configure({
      device: this.device,
      format: 'rgba8unorm',
      // The canvas swapchain is only used as a render attachment. Tile
      // compositing happens in a regular texture below; copying directly into
      // a swapchain texture is rejected by Chromium on some Metal drivers.
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      alphaMode: 'premultiplied',
    });

    this.inputTexture = this.device.createTexture({
      size: [tileInputWidth, tileInputHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.inputBuffer = this.device.createBuffer({
      size: alignTo16(3 * tileInputWidth * tileInputHeight * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.postprocessParams = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      this.preprocessParams,
      0,
      new Uint32Array([tileInputWidth, tileInputHeight])
    );
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
    this.compositeBindGroup = this.device.createBindGroup({
      layout: this.context.bindGroupLayouts.composite,
      entries: [{ binding: 0, resource: this.compositeTextureView }],
    });
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
    outputWidth: number,
    outputHeight: number,
    tiles: readonly GPUFrameTile[]
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
      outputWidth,
      outputHeight,
      firstTile.inputWidth,
      firstTile.inputHeight
    );

    if (!this.gpuCanvasContext || !this.inputTexture || !this.inputBuffer ||
        !this.outputTexture || !this.compositeTexture || !this.preprocessParams ||
        !this.postprocessParams || !this.compositeBindGroup) {
      throw new Error('GPU frame renderer resources are unavailable');
    }

    // Acquire the swapchain texture once. The final presentation pass is
    // appended to the last tile submission below, so the compositor and
    // presentation share one queue submit.
    const presentTexture = this.gpuCanvasContext.getCurrentTexture();
    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];

    const pendingResources: Array<{ output: ort.Tensor; input: ort.Tensor }> = [];
    // Retain a small output window before fencing. Four tiles reduce queue
    // synchronization for large frames while bounding GPU memory growth.
    const maxPendingTiles = Math.min(Math.max(tiles.length, 1), 4);

    const flushPending = async (): Promise<void> => {
      if (pendingResources.length === 0) return;

      const resources = pendingResources.splice(0, pendingResources.length);
      const fenceStarted = now();
      try {
        await this.device.queue.onSubmittedWorkDone();
        timing.gpuWaitMs += now() - fenceStarted;
      } finally {
        // ORT may recycle GPU output buffers after dispose(), so release them
        // only after postprocess and tile-copy commands have completed.
        for (const resource of resources) {
          resource.output.dispose();
          resource.input.dispose();
        }
      }
    };

    try {
      for (let tileIndex = 0; tileIndex < tiles.length; tileIndex++) {
        const tile = tiles[tileIndex];
        if (tile.inputWidth !== this.tileInputWidth || tile.inputHeight !== this.tileInputHeight) {
          throw new Error('GPU frame renderer requires one tile shape per frame');
        }

        const preprocessStarted = now();
        this.device.queue.copyExternalImageToTexture(
          {
            source,
            origin: { x: tile.sourceX, y: tile.sourceY },
          },
          { texture: this.inputTexture },
          [tile.inputWidth, tile.inputHeight]
        );

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.context.pipelines.preprocessF32);
        pass.setBindGroup(0, this.preprocessBindGroup!);
        pass.dispatchWorkgroups(
          Math.ceil(tile.inputWidth / 16),
          Math.ceil(tile.inputHeight / 16)
        );
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        timing.preprocessMs += now() - preprocessStarted;

        const inputTensor = ort.Tensor.fromGpuBuffer(this.inputBuffer, {
          dataType: 'float32',
          dims: [1, 3, tile.inputHeight, tile.inputWidth],
        });
        const inferenceStarted = now();
        let outputs: ort.InferenceSession.OnnxValueMapType;
        try {
          outputs = await this.session.run({ [inputName]: inputTensor });
        } catch (error) {
          inputTensor.dispose();
          throw error;
        }
        timing.inferenceMs += now() - inferenceStarted;
        const output = outputs[outputName];
        let queued = false;

        try {
          if (!output || output.location !== 'gpu-buffer') {
            throw new Error('ONNX Runtime returned a CPU output for GPU rendering');
          }
          const outputWidth = Number(output.dims[3]);
          const outputHeight = Number(output.dims[2]);
          if (outputWidth !== this.tileOutputWidth || outputHeight !== this.tileOutputHeight) {
            throw new Error('ONNX output shape does not match GPU tile resources');
          }

          const postprocessStarted = now();
          const postprocessBindGroup = this.device.createBindGroup({
            layout: this.context.bindGroupLayouts.postprocess,
            entries: [
              { binding: 0, resource: { buffer: output.gpuBuffer } },
              { binding: 1, resource: this.outputTextureView! },
              { binding: 2, resource: { buffer: this.postprocessParams } },
            ],
          });
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
          postprocessPass.setPipeline(this.context.pipelines.postprocessF32);
          postprocessPass.setBindGroup(0, postprocessBindGroup);
          postprocessPass.dispatchWorkgroups(
            Math.ceil(this.tileOutputWidth / 16),
            Math.ceil(this.tileOutputHeight / 16)
          );
          postprocessPass.end();
          postprocessEncoder.copyTextureToTexture(
            {
              texture: this.outputTexture,
              origin: { x: tile.keepStartX, y: tile.keepStartY },
            },
            {
              texture: this.compositeTexture,
              origin: { x: tile.destinationX, y: tile.destinationY },
            },
            [tile.keepWidth, tile.keepHeight]
          );

          if (tileIndex === tiles.length - 1) {
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
          pendingResources.push({ output, input: inputTensor });
          queued = true;

          if (pendingResources.length >= maxPendingTiles) {
            await flushPending();
          }
        } finally {
          if (!queued) {
            output?.dispose();
            inputTensor.dispose();
          }
        }
      }

      await flushPending();
      timing.gpuTimestampMs = await this.readTimestampMs();
    } finally {
      // Ensure resources submitted before an exception are released too.
      await flushPending();
    }

    const canvasStarted = now();
    const outputContext = outputCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
    if (!outputContext || !this.gpuCanvas) {
      throw new Error('Unable to mirror GPU output to the public canvas');
    }
    outputContext.drawImage(this.gpuCanvas, 0, 0, outputWidth, outputHeight);
    timing.canvasMs = now() - canvasStarted;
    this.lastTiming = timing;
  }

  getLastTiming(): GPUFrameTiming {
    return { ...this.lastTiming };
  }

  dispose(): void {
    this.destroyTileResources();
    this.destroyTimestampResources();
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
    this.inputTexture?.destroy();
    this.inputBuffer?.destroy();
    this.outputTexture?.destroy();
    this.compositeTexture?.destroy();
    this.preprocessParams?.destroy();
    this.postprocessParams?.destroy();
    this.inputTexture = null;
    this.inputBuffer = null;
    this.outputTexture = null;
    this.compositeTexture = null;
    this.inputTextureView = null;
    this.outputTextureView = null;
    this.compositeTextureView = null;
    this.preprocessBindGroup = null;
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
      targets: [{ format: 'rgba8unorm' }],
    },
    primitive: { topology: 'triangle-list' },
  });

  return {
    device,
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

  // For float32: 4 bytes per value, 3 channels
  // For float16: 2 bytes per value, but we use u32 storage (4 bytes) for alignment
  const bytesPerValue = useFloat16 ? 4 : 4; // Both use 4 bytes per slot in our shaders
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
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Output buffer (NCHW data from ONNX)
  const outputBuffer = device.createBuffer({
    size: outputBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Output texture (for canvas rendering)
  const outputTexture = device.createTexture({
    size: [outputWidth, outputHeight],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  // Uniform buffers for shader parameters
  const preprocessParams = device.createBuffer({
    size: 16, // 2 x u32 + padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const postprocessParams = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Initialize uniform buffers
  device.queue.writeBuffer(preprocessParams, 0, new Uint32Array([inputWidth, inputHeight]));
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

  // Dispatch workgroups (16x16 threads each)
  const workgroupsX = Math.ceil(pool.inputWidth / 16);
  const workgroupsY = Math.ceil(pool.inputHeight / 16);
  passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

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
