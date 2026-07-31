// Float16 variant of preprocess shader for AnimeJaNai models
// Converts RGBA texture to NCHW float16 buffer

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<u32>; // Packed float16 pairs

struct Params {
  tensorWidth: u32,
  tensorHeight: u32,
  sourceX: u32,
  sourceY: u32,
  sourceWidth: u32,
  sourceHeight: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

fn loadScalar(index: u32) -> f32 {
  let width = params.tensorWidth;
  let planeSize = width * params.tensorHeight;
  let channel = index / planeSize;
  let pixelIndex = index % planeSize;
  let localX = pixelIndex % width;
  let localY = pixelIndex / width;
  let sourceCoord = vec2<u32>(
    params.sourceX + min(localX, params.sourceWidth - 1u),
    params.sourceY + min(localY, params.sourceHeight - 1u)
  );
  let pixel = textureLoad(inputTexture, vec2<i32>(sourceCoord), 0);

  if (channel == 0u) { return pixel.r; }
  if (channel == 1u) { return pixel.g; }
  return pixel.b;
}

// ORT stores float16 GPU tensors as two adjacent IEEE half values per u32.
// One invocation owns one complete word, avoiding races between neighboring
// pixels and correctly handling pairs that cross NCHW channel boundaries.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let valueCount = 3u * params.tensorWidth * params.tensorHeight;
  let first = gid.x * 2u;
  if (first >= valueCount) {
    return;
  }

  let second = min(first + 1u, valueCount - 1u);
  outputBuffer[gid.x] = pack2x16float(vec2<f32>(loadScalar(first), loadScalar(second)));
}
