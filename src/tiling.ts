/**
 * Compute a tiled inference layout without forcing the final tile back to a
 * full-size origin. Every tile on an axis has the same shape, which keeps
 * ONNX Runtime's input shape stable while avoiding redundant edge inference.
 */

export interface TileAxisPlan {
  count: number;
  tileSize: number;
  step: number;
}

export interface TilePlan {
  overlap: number;
  x: TileAxisPlan;
  y: TileAxisPlan;
}

export interface TileCopyRegionInput {
  /** Absolute output-canvas origin of the uncropped model output tile. */
  tileOutputX: number;
  tileOutputY: number;
  /** Crop origin within the model output tile, in output pixels. */
  cropX: number;
  cropY: number;
  width: number;
  height: number;
}

export interface TileCopyRegion {
  sourceX: number;
  sourceY: number;
  destinationX: number;
  destinationY: number;
  width: number;
  height: number;
}

function planAxis(inputSize: number, maxTileSize: number, overlap: number): TileAxisPlan {
  if (inputSize <= maxTileSize) {
    return { count: 1, tileSize: inputSize, step: inputSize };
  }

  const maxStep = Math.max(1, maxTileSize - overlap);
  const count = Math.max(1, Math.ceil((inputSize - overlap) / maxStep));

  // count * tileSize - (count - 1) * overlap must cover the input.
  const tileSize = Math.min(
    maxTileSize,
    Math.max(1, Math.ceil((inputSize + (count - 1) * overlap) / count))
  );

  return {
    count,
    tileSize,
    step: Math.max(1, tileSize - overlap),
  };
}

export function calculateTilePlan(
  inputWidth: number,
  inputHeight: number,
  tileSize: number,
  tilePadding: number
): TilePlan {
  const safeTileSize = Math.max(1, Math.floor(tileSize));
  const overlap = Math.min(
    Math.max(0, Math.floor(tilePadding) * 2),
    Math.max(0, safeTileSize - 1)
  );

  return {
    overlap,
    x: planAxis(Math.max(1, Math.floor(inputWidth)), safeTileSize, overlap),
    y: planAxis(Math.max(1, Math.floor(inputHeight)), safeTileSize, overlap),
  };
}

/**
 * Place a cropped model-output tile on the output canvas.
 *
 * GPU texture copies place the crop's first pixel exactly at the destination
 * origin. This differs from putImageData's dirty-rectangle overload, which
 * implicitly offsets the destination by dirtyX/dirtyY. Keep this calculation
 * shared and testable so the CPU and GPU compositors preserve identical global
 * pixel coordinates.
 */
export function calculateTileCopyRegion(input: TileCopyRegionInput): TileCopyRegion {
  return {
    sourceX: input.cropX,
    sourceY: input.cropY,
    destinationX: input.tileOutputX + input.cropX,
    destinationY: input.tileOutputY + input.cropY,
    width: input.width,
    height: input.height,
  };
}

export default calculateTilePlan;
