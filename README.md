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
| Real-ESRGAN General Fast | 4x | General | Fast general content upscaling |
| Real-ESRGAN General Plus | 4x | General | High quality general content upscaling |
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
- **Fallback**: Any modern browser with WebAssembly support

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
