// Preprocess shader: Convert RGBA texture to NCHW float32/float16 buffer
// Input: RGBA texture from VideoFrame
// Output: NCHW buffer normalized to 0-1 range

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<f32>;

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

  // Load pixel from texture (already normalized 0-1 in f32 texture)
  let pixel = textureLoad(inputTexture, vec2<i32>(gid.xy), 0);

  // Calculate indices for NCHW format [1, 3, H, W]
  let pixelIdx = gid.y * width + gid.x;
  let planeSize = width * height;

  // Write RGB channels to separate planes (NCHW layout)
  outputBuffer[pixelIdx] = pixel.r;                      // R plane
  outputBuffer[planeSize + pixelIdx] = pixel.g;          // G plane
  outputBuffer[2u * planeSize + pixelIdx] = pixel.b;     // B plane
}
