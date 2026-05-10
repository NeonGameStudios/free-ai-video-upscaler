// Float16 variant of postprocess shader for AnimeJaNai models
// Converts NCHW float16 buffer to RGBA texture

@group(0) @binding(0) var<storage, read> inputBuffer: array<u32>; // Packed float16 (lower 16 bits)
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

struct Params {
  width: u32,
  height: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

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

  // Read packed float16 values and convert to f32
  let rPacked = inputBuffer[pixelIdx];
  let gPacked = inputBuffer[planeSize + pixelIdx];
  let bPacked = inputBuffer[2u * planeSize + pixelIdx];

  // Unpack float16 to float32 (value is in lower 16 bits)
  let r = clamp(unpack2x16float(rPacked).x, 0.0, 1.0);
  let g = clamp(unpack2x16float(gPacked).x, 0.0, 1.0);
  let b = clamp(unpack2x16float(bPacked).x, 0.0, 1.0);

  // Write to texture
  textureStore(outputTexture, vec2<i32>(gid.xy), vec4<f32>(r, g, b, 1.0));
}
