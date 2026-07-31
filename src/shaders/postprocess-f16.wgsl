// Float16 variant of postprocess shader for AnimeJaNai models
// Converts NCHW float16 buffer to RGBA texture

@group(0) @binding(0) var<storage, read> inputBuffer: array<u32>; // Two adjacent float16 values per u32
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

struct Params {
  width: u32,
  height: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

fn loadFloat16(index: u32) -> f32 {
  let pair = unpack2x16float(inputBuffer[index / 2u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = params.width;
  let height = params.height;

  // Bounds check
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  // Calculate indices for NCHW format [1, 3, H, W]
  let pixelIdx = gid.y * width + gid.x;
  let planeSize = width * height;

  let r = clamp(loadFloat16(pixelIdx), 0.0, 1.0);
  let g = clamp(loadFloat16(planeSize + pixelIdx), 0.0, 1.0);
  let b = clamp(loadFloat16(2u * planeSize + pixelIdx), 0.0, 1.0);

  // Write to texture
  textureStore(outputTexture, vec2<i32>(gid.xy), vec4<f32>(r, g, b, 1.0));
}
