// Float16 variant of preprocess shader for AnimeJaNai models
// Converts RGBA texture to NCHW float16 buffer

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<u32>; // Packed float16 pairs

struct Params {
  width: u32,
  height: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

// Pack two f32 values into one u32 as two float16
fn packFloat16x2(a: f32, b: f32) -> u32 {
  let aHalf = pack2x16float(vec2<f32>(a, 0.0));
  let bHalf = pack2x16float(vec2<f32>(b, 0.0));
  return (aHalf & 0xFFFFu) | ((bHalf & 0xFFFFu) << 16u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = params.width;
  let height = params.height;

  // Bounds check
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  // Load pixel from texture
  let pixel = textureLoad(inputTexture, vec2<i32>(gid.xy), 0);

  // Calculate indices for NCHW format [1, 3, H, W]
  let pixelIdx = gid.y * width + gid.x;
  let planeSize = width * height;

  // For float16, we need to pack values differently
  // Each f16 is 2 bytes, so we pack 2 values per u32
  // But for simplicity with NCHW layout, we'll store each channel separately
  // using the lower 16 bits of each u32 slot

  // Convert to float16 and store (using pack2x16float)
  let rPacked = pack2x16float(vec2<f32>(pixel.r, 0.0));
  let gPacked = pack2x16float(vec2<f32>(pixel.g, 0.0));
  let bPacked = pack2x16float(vec2<f32>(pixel.b, 0.0));

  // Store as u32 (only lower 16 bits contain our f16 value)
  outputBuffer[pixelIdx] = rPacked;
  outputBuffer[planeSize + pixelIdx] = gPacked;
  outputBuffer[2u * planeSize + pixelIdx] = bPacked;
}
