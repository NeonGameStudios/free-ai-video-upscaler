# Wishlist / Known Limitations

This document tracks known limitations in the current implementation and potential future improvements.

## WebGPU Zero-Copy Path

### 1. CPU Round-Trip Still Required Per Frame

**Current behavior:** The ONNX tensor data must be read from GPU buffer to CPU ArrayBuffer, then written back to GPU after inference.

**Reasoning:** ONNX Runtime Web doesn't clearly expose APIs for accepting external GPU buffers. The `ort.Tensor.fromGpuBuffer()` API exists but documentation on sharing buffers with custom WebGPU pipelines is sparse. We use `ort.env.webgpu.device` to share the device, but buffer interop isn't straightforward.

**Ideal solution:** ONNX Runtime accepts our `GPUBuffer` directly, keeping data on GPU throughout: VideoFrame → texture → our buffer → ONNX tensor → ONNX output buffer → our buffer → texture → canvas.

---

### 2. No Tiled Processing in GPU Path

**Current behavior:** `renderGPU()` processes the entire frame at once. Large resolutions (4K+) may exceed GPU memory limits.

**Reasoning:** The CPU path implements tiling with overlap blending for seamless results. Porting this to the GPU path requires:
- Multiple buffer pools or dynamic sizing
- Compute shader modifications for tile coordinates
- Overlap blending logic in the postprocess shader

**Ideal solution:** Implement GPU-side tiling with configurable tile sizes (1024-2048px) and seamless blending.

---

### 3. VideoFrame Only (No ImageBitmap Support)

**Current behavior:** GPU path only activates for `VideoFrame` objects. `ImageBitmap` inputs fall back to CPU path.

**Reasoning:** `VideoFrame` has `codedWidth`/`codedHeight` properties and can be efficiently imported to GPU texture via `copyExternalImageToTexture()`. ImageBitmap support would require different dimension detection and potentially less efficient texture upload.

**Ideal solution:** Extend GPU path to handle ImageBitmap for image upscaling use cases.

---

### 4. AnimeJaNai Models Excluded from WebGPU

**Current behavior:** Float16 AnimeJaNai models use WASM execution provider instead of WebGPU.

**Reasoning:** ONNX Runtime Web's WebGPU execution provider has reported issues with float16 models. The WASM provider is more reliable for these models, though slower.

**Ideal solution:** Monitor ONNX Runtime Web updates for improved float16 WebGPU support. Our shaders already support float16 (`preprocess-f16.wgsl`, `postprocess-f16.wgsl`).

---

### 5. No Overlap Blending in GPU Path

**Current behavior:** When tiling is eventually added to GPU path, it won't have seamless blending initially.

**Reasoning:** The CPU path uses `putImageData()` with dirty rectangle parameters to blend overlapping tile regions. GPU equivalent requires:
- Storing overlap regions in separate buffers
- Alpha blending in postprocess shader
- Careful coordinate math for tile boundaries

**Ideal solution:** Implement feathered blending in postprocess shader using distance-from-edge weights.

---

### 6. Canvas Context Recreation

**Current behavior:** `copyToCanvas()` gets a new WebGPU context and calls `configure()` each frame.

**Reasoning:** The output canvas is an `OffscreenCanvas` passed from the worker. WebGPU context configuration is idempotent but may have overhead. Caching the configured context would require tracking canvas identity.

**Ideal solution:** Cache the `GPUCanvasContext` and only reconfigure if canvas changes.

---

### 7. Buffer Pool Recreation on Resolution Change

**Current behavior:** If video frame dimensions change mid-stream, the entire `GPUBufferPool` is destroyed and recreated.

**Reasoning:** GPU buffers have fixed sizes. Variable resolution video (rare) or seeking between different-resolution segments triggers full reallocation.

**Ideal solution:** Pool multiple buffer sets for common resolutions, or allocate for max expected resolution upfront.

---

## Other Improvements

### 8. MKV Container Support

**Current behavior:** File picker only accepts MP4 and WebM. MKV files are rejected.

**Reasoning:** Browsers cannot decode MKV natively. FFmpeg.wasm remuxing was implemented but adds complexity and load time. The remux code exists in `src/remux.ts` but is disabled in the UI.

**Ideal solution:** Re-enable MKV support with clear user feedback about the remuxing step.

---

### 9. Progress Reporting During GPU Inference

**Current behavior:** Progress updates are frame-based but don't reflect actual GPU work completion.

**Reasoning:** GPU operations are asynchronous. `device.queue.submit()` returns immediately; actual completion is later. Accurate progress would require `GPUQueue.onSubmittedWorkDone()` callbacks.

**Ideal solution:** Use WebGPU fence/callback APIs for accurate progress reporting.
