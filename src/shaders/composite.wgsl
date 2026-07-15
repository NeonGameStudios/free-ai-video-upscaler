// Present a fully composited frame to the WebGPU canvas swapchain.
// The source texture is sampled with integer framebuffer coordinates so the
// copy preserves the exact RGBA pixels produced by postprocess.wgsl.

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  return vec4<f32>(positions[index], 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  return textureLoad(sourceTexture, vec2<i32>(position.xy), 0);
}
