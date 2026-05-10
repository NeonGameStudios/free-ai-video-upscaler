// Postprocess shader: Convert NCHW float32 buffer to RGBA texture
// Input: NCHW buffer from model output (values in 0-1 range)
// Output: RGBA texture for canvas rendering

@group(0) @binding(0) var<storage, read> inputBuffer: array<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

struct Params {
  width: u32,   // Output width (input_width * scale)
  height: u32,  // Output height (input_height * scale)
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

  // Read RGB channels from separate planes, clamp to valid range
  let r = clamp(inputBuffer[pixelIdx], 0.0, 1.0);
  let g = clamp(inputBuffer[planeSize + pixelIdx], 0.0, 1.0);
  let b = clamp(inputBuffer[2u * planeSize + pixelIdx], 0.0, 1.0);

  // Write to texture (alpha = 1.0)
  textureStore(outputTexture, vec2<i32>(gid.xy), vec4<f32>(r, g, b, 1.0));
}
