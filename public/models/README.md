# AI Upscaling Models

This directory contains ONNX model files for video upscaling. Models are auto-downloaded from Hugging Face when selected, but you can also place local files here.

## Available Models

| Model ID | Name | Scale | Status | Description |
|----------|------|-------|--------|-------------|
| `realesr-animevideov3` | RealESR AnimeVideo v3 | 4x | ✅ Auto-download | Optimized for anime videos (Recommended) |
| `animejanai-v3-sd` | AnimeJaNai V3 - SD | 2x | ✅ Local | Soft upscaling, faithful to source |
| `animejanai-v3-hd` | AnimeJaNai V3 - HD | 2x | ✅ Local | Sharp upscaling for HQ sources |
| `animejanai-v3-hd-fast` | AnimeJaNai V3 - HD Fast | 2x | ✅ Local | Fast HD, good speed/quality balance |
| `animejanai-v3-hd-superfast` | AnimeJaNai V3 - HD Superfast | 2x | ✅ Local | Fastest HD, smallest model |
| `realesrgan-anime-fast` | Real-ESRGAN Anime Fast | 4x | ✅ Auto-download | Fast anime upscaling |
| `realesrgan-anime-plus` | Real-ESRGAN Anime Plus | 4x | ✅ Auto-download | High quality anime upscaling |
| `realesrgan-general-fast` | Real-ESRGAN General Fast | 4x | ⏳ Coming Soon | Fast general content upscaling |
| `realesrgan-general-plus` | Real-ESRGAN General Plus | 4x | ⏳ Coming Soon | High quality general upscaling |
| `realcugan-2x` | Real-CUGAN 2x | 2x | ✅ Auto-download | Conservative anime with denoising |
| `realcugan-4x` | Real-CUGAN 4x | 4x | ⏳ Coming Soon | High quality anime with denoising |
| `realplksr-deh264-1x` | RealPLKSR DeH264 1x | 1x | ✅ Auto-download | Same-resolution H.264 artifact cleanup |
| `realplksr-dejpg-1x` | RealPLKSR DeJPG 1x | 1x | ✅ Auto-download | Same-resolution JPEG/block cleanup |
| `realplksr-denoise-1x` | RealPLKSR Denoise 1x | 1x | ✅ Auto-download | Same-resolution general denoising |
| `scunet-psnr` | SCUNet PSNR 1x | 1x | ✅ Auto-download | Conservative blind denoising |
| `scunet-gan` | SCUNet GAN 1x | 1x | ✅ Auto-download | Stronger blind denoising |
| `swinir-jpeg40-1x` | SwinIR JPEG40 1x | 1x | ✅ Local | Same-resolution JPEG artifact and ringing cleanup |

## Auto-Download Sources

Models are downloaded from Hugging Face and cached in IndexedDB:

- **RealESR AnimeVideo v3**: [xiongjie/lightweight-real-ESRGAN-anime](https://huggingface.co/xiongjie/lightweight-real-ESRGAN-anime)
- **Real-ESRGAN Anime**: [deepghs/imgutils-models](https://huggingface.co/deepghs/imgutils-models)
- **Real-ESRGAN General**: [qualcomm/Real-ESRGAN-x4plus](https://huggingface.co/qualcomm/Real-ESRGAN-x4plus)
- **Real-CUGAN, RealPLKSR, SCUNet**: [hugglyberry/upscale-and-refine-models](https://huggingface.co/hugglyberry/upscale-and-refine-models)
- **SwinIR JPEG**: converted locally from [deepinv/swinir](https://huggingface.co/deepinv/swinir) weights using `scripts/convert_swinir_jpeg_to_onnx.py`

## Converting Models

For models not yet available online, use the conversion script:

```bash
cd scripts

# Install dependencies
pip install torch onnx basicsr realesrgan

# Convert a specific model
python convert_model.py realesr-animevideov3

# List available models
python convert_model.py --list
```

### AnimeJaNai V3 Models

These models are included in this repository (in `public/models/`):

- `2x_AnimeJaNai_SD_V1beta34_Compact.onnx` - SD variant (~1.2 MB)
- `2x_AnimeJaNai_HD_V3_Compact.onnx` - HD variant (~1.2 MB)
- `2x_AnimeJaNai_HD_V3_UltraCompact.onnx` - HD Fast variant (~600 KB)
- `2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx` - HD Superfast variant (~96 KB)

Source: [mpv-upscale-2x_animejanai releases](https://github.com/the-database/mpv-upscale-2x_animejanai/releases/tag/3.0.0)

### Real-CUGAN Models

Real-CUGAN uses a different architecture (CUGAN) that requires additional setup. Consider using the TensorFlow.js version from [web-realesrgan](https://github.com/nicholashz/web-realesrgan).

## Model Information

### Real-ESRGAN Models
- **Source**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **License**: BSD-3-Clause
- **Architectures**: SRVGGNetCompact (fast/compact), RRDBNet (plus)

### AnimeJaNai V3 Models
- **Source**: [the-database/mpv-upscale-2x_animejanai](https://github.com/the-database/mpv-upscale-2x_animejanai)
- **Architecture**: SRVGGNetCompact (2x scale)
- **Variants**: 
  - SD (soft/conservative)
  - HD Compact (sharp/aggressive, best quality)
  - HD UltraCompact (faster, good balance)
  - HD SuperUltraCompact (fastest, smallest)

### Real-CUGAN Models
- **Source**: [bilibili/ailab](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)
- **License**: MIT
- **Features**: Built-in denoising (levels 0-3)

## Notes

- Models are cached in IndexedDB after first download
- WebGPU provides best performance; WASM fallback available
- Model sizes: ~96 KB-91 MB depending on architecture
- Clear browser cache to re-download models
