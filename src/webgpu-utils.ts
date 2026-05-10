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

export interface WebGPUContext {
  device: GPUDevice;
  pipelines: {
    preprocessF32: GPUComputePipeline;
    postprocessF32: GPUComputePipeline;
    preprocessF16: GPUComputePipeline;
    postprocessF16: GPUComputePipeline;
  };
  bindGroupLayouts: {
    preprocess: GPUBindGroupLayout;
    postprocess: GPUBindGroupLayout;
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

  // Create pipeline layouts
  const preprocessPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [preprocessLayout],
  });

  const postprocessPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [postprocessLayout],
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

  return {
    device,
    pipelines: {
      preprocessF32,
      postprocessF32,
      preprocessF16,
      postprocessF16,
    },
    bindGroupLayouts: {
      preprocess: preprocessLayout,
      postprocess: postprocessLayout,
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
