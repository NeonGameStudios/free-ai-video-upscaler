# Wishlist / Known Limitations

This document tracks current limitations and future improvements for the local fork.

## Completed

### 1. Remove Broken Custom WebGPU Zero-Copy Path

The custom shader path that imported `VideoFrame` into WebGPU textures has been removed from the active render path. It was faster in theory, but it bypassed the proven tiled renderer and could generate corrupted "snow" output.

The app now uses ONNX Runtime's execution providers for model acceleration while keeping preprocessing, tiling, overlap handling, and canvas output on the reliable path shared by preview and full video processing.

### 2. Preserve Tiled Processing for Video Frames

Video processing now uses the same tiled path as preview rendering. This keeps overlap handling intact and avoids whole-frame inference for larger inputs.

The 360p case is fixed: frames such as 640x360 no longer produce negative tile origins when one dimension is smaller than the tile size.

### 3. MKV Container Support

The file picker accepts `.mkv` files, and the app remuxes MKV input to MP4 before browser decoding. The UI shows the remuxing step while it runs.

### 4. Output Resolution Presets Affect Encoding

The 720p/1080p/etc. selector now caps the actual encoded canvas size instead of only changing the displayed estimate. This is especially useful for restoring 360p sources to 720p without writing unnecessary 4x output when a 4x model is selected.

### 5. Safer WebGPU Provider Selection

Model loading now tries ONNX Runtime WebGPU when WebGPU is available, validates the session with a small inference, and falls back to WASM if the model/provider combination fails.

This allows AnimeJaNai float16 models to use WebGPU when the browser and ONNX Runtime support it, while preserving WASM reliability on unsupported setups.

### 6. Progress Reports Completed Work

Progress is reported after each frame finishes rendering and is queued for encoding. Finalization now posts `100%` after the output container is finalized.

### 7. AnimeJaNai Float16 Output Handling

AnimeJaNai models still use float16 input tensors, but ONNX Runtime Web may return float16 model output in a `Float32Array` depending on the execution provider. The renderer now detects that storage type during session validation and converts either float32-backed or uint16-backed float16 output correctly.

This fixes the all-black AnimeJaNai output seen with the 15 second test clip.

### 8. Audio Passthrough And Runtime Cleanup

When the source audio codec is supported by the selected output container, the worker now copies encoded audio packets directly instead of decoding and re-encoding. This keeps audio untouched for common MP4/AAC inputs and avoids the breakup introduced by unnecessary re-encoding. AAC priming packets with negative timestamps are skipped or retimed to begin at zero so the MP4 muxer accepts them.

The worker also awaits video encoder backpressure, closes media sources, disposes the MediaBunny input, clears per-frame upscaler buffers after preview/conversion, and releases temporary resize canvases. The main thread now revokes old video/download object URLs and closes stale preview bitmaps when loading another file.

### 9. Target-Resolution Preview And Inference

Preview rendering and full conversion now resize before model inference when the selected output preset caps the final resolution. For example, a 960x540 source targeting 720p with a 4x model now infers a 320x180 frame and writes 1280x720 output instead of previewing or processing a 3840x2160 intermediate.

When output is left on Auto, preview rendering is still capped at 1080p to avoid crashing during model selection. Full conversion still honors Auto/native scale.

### 10. Real-CUGAN 2x Denoise Variants

Real-CUGAN 2x is available through the model picker and maps the denoise slider to the hosted ONNX variants: no-denoise, denoise1x, denoise2x, and denoise3x. The loader supports the model's paired `.onnx` and `.onnx.data` files and caches both in IndexedDB.

For low-bitrate 80s cartoon cleanup, start with denoise level 1, then try level 2 if action scenes still show block noise or shimmer.

### 11. RealPLKSR Same-Resolution Cleanup Models

The model picker now has a dedicated same-resolution cleanup group. These models run at 1x and automatically select "Keep Existing Resolution" when chosen:

- `1xDeH264_realplksr` for low-bitrate video block noise
- `1xDeJPG_realplksr_otf` for JPEG-like artifact removal
- `1xDeNoise_realplksr_otf` for general image/video denoising

These should be benchmarked against Real-CUGAN 2x denoise level 1 on the local cartoon clips before changing the default recommendation.

### 12. SCUNet Blind Denoising

SCUNet PSNR and SCUNet GAN are available as 1x same-resolution cleanup models. The hosted ONNX files use float32 NCHW input/output and return same-resolution frames. Runtime validation showed SCUNet requires dimensions that are multiples of 64, so the tiled renderer now pads non-conforming tiles by extending edge pixels before inference and crops the padded output when writing back to the canvas.

SCUNet models are significantly larger than the RealPLKSR cleanup models at about 87 MiB each. Use SCUNet PSNR first for conservative blind denoising, and compare SCUNet GAN only when stronger cleanup is worth the extra risk of hallucinated texture.

### 13. SwinIR JPEG Artifact Reduction

SwinIR JPEG40 is available as a 1x same-resolution cleanup model. No public browser-ready SwinIR JPEG ONNX artifact was found, so the local fork includes `scripts/convert_swinir_jpeg_to_onnx.py` to convert the official color JPEG artifact reduction weights from `deepinv/swinir`.

The generated model is fixed at 126x126 float32 NCHW input/output, uses JPEG quality 40 weights, and is served locally from `public/models/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.onnx`. The renderer uses fixed 126px SwinIR tiles with overlap so arbitrary video frame sizes can be processed through the fixed-size ONNX model.

Benchmark this against `RealPLKSR DeJPG 1x`, `RealPLKSR DeH264 1x`, and `Real-CUGAN 2x` denoise level 1 before making it a default recommendation.

### 14. Adaptive Tiling And Stage Benchmarks

The renderer now computes a uniform tile shape per axis, with explicit edge
coverage and overlap. This keeps ONNX input shapes stable while avoiding the
old clamped final-tile inference. `benchmark:tiling` checks coverage and
reports inferred-pixel reductions for 360p, 720p, and 1080p cases.

Worker timing messages now average decode, audio, preprocess, inference,
postprocess, canvas, encode, and total frame time, along with tile and pixel
counts.

### 15. GPU-Resident Float32 Interop

Dynamic float32 models can use a separate WebGPU canvas for external-image
upload, WGSL NCHW preprocessing, `Tensor.fromGpuBuffer`, GPU-resident ONNX
outputs, WGSL postprocessing, and GPU tile compositing. The public 2D canvas
is updated once per frame for preview and encoding compatibility. A failed
interop step disposes the bridge and falls back to the CPU-tiled renderer.

The compositor now copies tile results into a regular GPU texture and presents
that texture with one render pass. This avoids copying directly into the
canvas swapchain, which triggered Chromium/Metal validation errors and added
avoidable synchronization on the measured 720p/1080p paths.

The final presentation pass is submitted with the last tile, and tiled frames
retain up to four GPU outputs before fencing. This removes a redundant queue
submit per frame while keeping tensor disposal behind the GPU completion fence.

AnimeJaNai float16 graphs now attempt WebGPU after a PRelu-to-primitive rewrite,
but remain explicitly experimental: WebGPU validation failures use the
original model through WASM. `verify:onnx-rewrite` covers every local
AnimeJaNai model.

The GPU compositor now configures its presentation surface with the browser's
preferred WebGPU swapchain format and an opaque alpha mode. This matches the
Metal-native path on macOS without changing the RGBA tile-processing format.

## Remaining

### 1. RVRT / VRT Temporal Video Restoration

Investigate RVRT or VRT for temporal cleanup across multiple frames. This is deferred because these models need multi-frame input windows or recurrent state, which requires a different worker pipeline than the current frame-by-frame renderer.

### 2. Full End-To-End Clip Encoding Benchmark

`benchmark:encoding` now uses both repository clips as fixed inputs and runs
480p, 720p, and 1080p target-height cases. It records:

- model selected
- input and output resolution
- per-frame render time
- total encode time
- output file size
- a deterministic pixel checksum for visual sanity checks

The default run can be bounded with `clip`, `targets`, and `frames` query
parameters while comparing experimental branches. Full-duration runs remain
useful for final validation of snow/corruption across the complete output.

### 3. Model/Content Recommendations

After testing real clips, document which available model works best for noisy VHS-style footage. Anime-focused models may not be ideal for live-action VHS restoration, so this needs evidence from the local test clips rather than assumptions.
