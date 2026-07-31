# Free AI Video Upscaler

A simple, quick and free no-nonsense tool for upscaling video with AI upscaling algorithms right in your browser - no signups, no downloads, just choose a video and download your upscaled video after it's done processing. Powered by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [AnimeJaNai](https://github.com/the-database/mpv-upscale-2x_animejanai), and [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) via ONNX Runtime Web with WebGPU acceleration.

You can get started at [Free AI Video Upscaler](https://free.upscaler.video/)

<img src="https://github.com/sb2702/free-ai-video-upscaler/assets/5678502/60ed1132-b21d-4ecf-917d-f4ae831bb91c"  width="600" />

## Available Models

| Model | Scale | Type | Description |
|---|---|---|---|
| RealESR AnimeVideo v3 | 4x | Anime | Compact, fast anime video upscaling (default) |
| Real-ESRGAN Anime Fast | 4x | Anime | Fast anime upscaling |
| Real-ESRGAN Anime Plus | 4x | Anime | High quality anime upscaling |
| Real-ESRGAN General Fast | 4x | General | Fast general content upscaling (coming soon) |
| Real-ESRGAN General Plus | 4x | General | High quality general content upscaling (coming soon) |
| AnimeJaNai V3 - SD | 2x | Anime | Soft upscaling, faithful to source |
| AnimeJaNai V3 - HD | 2x | Anime | Sharp upscaling for high quality sources |
| AnimeJaNai V3 - HD Fast | 2x | Anime | Fast HD upscaling, good speed/quality balance |
| AnimeJaNai V3 - HD Superfast | 2x | Anime | Fastest HD upscaling, smallest model |
| Real-CUGAN 2x | 2x | Anime | Conservative upscaling with denoising (coming soon) |
| Real-CUGAN 4x | 4x | Anime | High quality upscaling with denoising (coming soon) |

## Setup & Installation

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Development

```bash
# Clone the repository
git clone https://github.com/NeonGameStudios/free-ai-video-upscaler.git
cd free-ai-video-upscaler

# Install dependencies
npm install

# Start development server
npm run dev
```

The dev server runs at `http://localhost:8080` with hot reloading.

### Production Build

```bash
# Build for production
npm run build

# Output is in dist/
```

### Docker (Alternative)

```bash
# Build and run with Docker Compose
docker compose up

# Or build manually
docker build -t video-upscaler .
docker run -p 8080:8080 video-upscaler
```

The dev server runs at `http://localhost:8080`. Open in Chrome/Safari to use WebGPU acceleration on M-series Macs.

### Browser Requirements

- **Recommended**: Chrome/Edge with WebGPU support (best performance)
- **Fallback**: Any modern browser with WebAssembly support. The app validates
  the selected ONNX execution provider and falls back to WASM when WebGPU or
  a model graph is unsupported. Browsers without File System Access use a
  standard file input; large streamed outputs still require a save-file picker.

## Performance behavior

`Auto` chooses a conservative output-height cap from the hardware signals the
browser exposes. It uses 1080p by default, 1440p with at least 4 GiB of reported
memory and 4 logical processors, and 4K with at least 8 GiB and 8 processors.
When memory is not reported, 8 processors select 1440p. Auto caps content at
the model's native scale and does not downscale a source already above the cap;
final codec dimensions are rounded upward to even values when necessary. A
1080p input with a 4x model therefore no longer becomes an implicit 8K job. The
active Auto cap is shown in the resolution control.

Outputs estimated at 192 MiB or less use the convenient in-memory download
path. Larger outputs, and inputs whose duration is unknown, ask for a save
location before processing and stream muxed chunks directly to that file. MP4
fast-start relocation is disabled so the in-memory convenience path does not
also retain a second set of encoded packets until finalization. If the browser
does not expose a save-file picker, select a smaller output or use Chrome/Edge
for a streamed job.

On a validated WebGPU session, each source frame is uploaded once and tile
preprocessing, inference, postprocessing, overlap composition, and presentation
remain GPU-resident. Float32 and packed-float16 tensors, including fixed-shape
and padded model inputs, use this path when ONNX Runtime accepts GPU-buffer
interop. Input/output GPU buffers, ORT tensors, and bind groups are reused for
every tile of the same shape. When inference already produces the requested
encode size and the browser can snapshot a WebGPU canvas, the encoder consumes
that canvas directly, avoiding the extra WebGPU-to-2D-canvas copy. Setup-time
validation or interop failures fall back to the compatible CPU/WASM path. A
runtime GPU failure after the encoder binds a direct canvas stops that job
instead of changing encoder surfaces mid-stream.

ONNX Runtime graph capture is deliberately disabled. In ORT 1.23.2 it requires
GPU-only validation and immutable external buffer identities and shapes, while
preview-to-job changes and adaptive/fixed-padding tile dimensions can rebuild
buffers, and the recovery path must be able to bind CPU tensors. Persistent,
preallocated GPU tensors remove per-tile allocation without those capture
constraints.

WebGPU initialization requests the high-performance adapter on macOS so a
machine with multiple adapters does not select a low-power device for model
inference. Worker diagnostics report decoder-iterator wait, frame conversion,
preprocess, inference, postprocess, GPU queue-wait, GPU timestamp, canvas,
encode, finalization, and end-to-end wall FPS.

## Performance checks

```bash
# Validate overlap cropping and destination placement for edge/interior tiles
npm run verify:tile-placement

# Validate codec-safe odd dimensions and inference sizing
npm run verify:resolution

# Compare adaptive tiling with the previous full-size edge-tile plan
npm run benchmark:tiling

# Build the repeatable browser encoding benchmark
npm run benchmark:encoding

# Or build and serve it directly on http://localhost:8080
npm run benchmark:encoding:serve
```

The deterministic tiling benchmark currently reports the following reduction
in model-input pixels versus the previous full-size edge-tile plan (measured
July 31, 2026). These figures isolate inference work; they are not claimed as
equivalent wall-clock speedups.

| Input / max tile | Previous pixels | Adaptive pixels | Reduction |
| --- | ---: | ---: | ---: |
| 640×360 / 512 | 368,640 | 253,440 | 31.3% |
| 1280×720 / 512 | 1,572,864 | 1,105,440 | 29.7% |
| 1920×1080 / 512 | 3,932,160 | 2,635,620 | 33.0% |
| 1920×1080 / 1024 | 4,194,304 | 2,269,696 | 45.9% |

A local direct-canvas A/B on July 31, 2026 used Chrome 150.0.7871.49
headless on arm64 macOS 26.5.2, the short bundled clip, RealESR AnimeVideo,
960×540 output, a 256-pixel tile limit, disabled GPU timestamp queries, 5
warmup frames, and 20 measured frames. The 540p output cap reduced inference to
one 240×135 tile before the 4x model restored 960×540 output, so this isolates
encoder-surface overhead rather than measuring a full 960×540-to-4K upscale.
Each variant ran in a fresh Chrome process; both produced the same 184,410-byte
output and pixel validation checksum.

| Encode surface | Wall FPS | Mean frame time | FPS incl. finalize |
| --- | ---: | ---: | ---: |
| WebGPU → 2D mirror | 4.33 | 230.93 ms | 4.31 |
| Direct WebGPU canvas | 4.44 | 225.38 ms | 4.42 |

In that diagnostic, direct encoding improved measured wall throughput by
2.5% and reduced mean frame time by 2.4%. Treat the figures as a reproducible
local result rather than a universal browser/GPU guarantee; use `directGpu=0`
to repeat the control on another machine. The 20-frame measurement windows
were 4.619 seconds for the mirrored path and 4.508 seconds for the direct path;
including five warmup frames plus source close/finalization took 5.798 and
5.654 seconds respectively. Chrome startup, model initialization, output setup,
and validation were outside that pipeline timer.

A separate 1920×1080 WebGPU smoke used six tiles and the direct encoder path.
All 35 border/interior/seam probes were opaque and non-black; nine seam pairs
had a maximum adjacent-pixel RGB delta of 5. This checks placement and output
integrity, not visual equivalence to a CPU reference implementation.

With the benchmark server running, open a URL such as:

`http://localhost:8080/encoding-benchmark.html?clip=short&targets=720,1080&frames=30&warmup=3&tileSize=256&gpuTiming=1&model=realesr-animevideov3`

The encoding benchmark embeds the repository's `short` and `long` clips and
tests 480p, 720p, and 1080p targets by default. Its query parameters are:

- `clip=short|long` (omit it to run both clips)
- `targets=480,720,1080` (comma-separated output heights)
- `frames=30|all` (omit it or use `all` for the full clip)
- `warmup=3` (measured frames exclude these warmup frames)
- `tileSize=256` (minimum 32; defaults to 256 for 1x and 512 for scaled models)
- `gpuTiming=0|1` (enabled by default when timestamp queries are available)
- `directGpu=0|1` (disable the direct encoder canvas for an A/B comparison)
- `model=realesr-animevideov3` (any model ID available to the app)

The benchmark intentionally writes video-only MP4 output so audio passthrough
does not obscure decode, upscale, and encoder comparisons. Other targeted
checks remain available through `npm run verify:onnx-rewrite`,
`npm run benchmark:webgpu`, and `npm run benchmark:clip`.

## Model Conversion

To convert PyTorch models to ONNX format for browser inference:

```bash
pip install torch onnx basicsr realesrgan

# Convert a single model
python scripts/convert_model.py realesr-animevideov3

# Convert all models (skips those requiring manual download)
python scripts/convert_model.py --all

# List available models
python scripts/convert_model.py --list
```

AnimeJaNai V3 models are already included as ONNX files in `public/models/`. Source: [the-database/mpv-upscale-2x_animejanai](https://github.com/the-database/mpv-upscale-2x_animejanai/releases).

Based on my [WebSR](https://github.com/sb2702/websr) SDK.
