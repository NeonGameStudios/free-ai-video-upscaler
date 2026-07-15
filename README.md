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
  a model graph is unsupported.

## Performance checks

The renderer uses an adaptive, overlap-aware tile plan so edge tiles do not
repeat a full maximum-size inference. It also reports averaged decode,
preprocess, inference, postprocess, canvas, and encode timings in the worker's
diagnostic `timing` message.

```bash
# Check the adaptive tile plan against representative frame sizes
npm run benchmark:tiling

# Verify the AnimeJaNai PRelu graph rewrite used by the experimental WebGPU path
npm run verify:onnx-rewrite

# Build a small browser WebGPU smoke benchmark, then serve dist/ on port 8080
npm run benchmark:webgpu

# Decode and upscale six frames from test-clips/BotsMaster-15sec.mp4
npm run benchmark:clip

# Build the repeatable two-clip encoding benchmark
npm run benchmark:encoding
```

The browser benchmarks emit `gpu-benchmark.js`, `clip-regression.js`, and a
content-hashed `encoding-benchmark.*.js`; load their corresponding temporary HTML pages from
the development server. The encoding benchmark uses both repository clips and
tests 480p, 720p, and 1080p target heights by default. Use query parameters to
bound a run, for example:

`encoding-benchmark.html?clip=short&targets=720,1080&frames=30`

For tiled-path experiments, add `tileSize=256` (the default is 512 for 4x
models); the selected tile size is reported with each case.

The encoding benchmark intentionally writes video-only MP4 output so audio
passthrough does not obscure decode, upscale, and video-encoder comparisons.
For reproducible local runs, the temporary benchmark bundle embeds the two
repository clips; the production bundle does not include them.

AnimeJaNai float16 WebGPU is deliberately experimental: the graph rewrite is
validated structurally, but browsers/ONNX Runtime builds that cannot execute
the rewritten graph automatically use the original model through WASM. The
default float32 models use the GPU-buffer path when ONNX Runtime returns a
GPU-resident output; any interop failure falls back to the existing CPU-tiled
renderer for correctness.

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
