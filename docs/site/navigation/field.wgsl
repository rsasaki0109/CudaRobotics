@group(0) @binding(0) var<storage, read> occupancy: array<u32>;
@group(0) @binding(1) var<storage, read_write> horizontal: array<f32>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;
override GRID_WIDTH: u32 = 128u;
override GRID_HEIGHT: u32 = 96u;
override CELL_SIZE: f32 = 0.125;

@compute @workgroup_size(64)
fn row_distance(@builtin(global_invocation_id) invocation: vec3u) {
  let index = invocation.x;
  if (index >= GRID_WIDTH * GRID_HEIGHT) { return; }
  let column = index % GRID_WIDTH;
  let row = index / GRID_WIDTH;
  var nearest = 100000000.0;
  for (var obstacle = 0u; obstacle < GRID_WIDTH; obstacle++) {
    if (occupancy[row * GRID_WIDTH + obstacle] != 0u) {
      let delta = f32(column) - f32(obstacle);
      nearest = min(nearest, delta * delta);
    }
  }
  horizontal[index] = nearest;
}

@compute @workgroup_size(64)
fn column_distance(@builtin(global_invocation_id) invocation: vec3u) {
  let index = invocation.x;
  if (index >= GRID_WIDTH * GRID_HEIGHT) { return; }
  let column = index % GRID_WIDTH;
  let row = index / GRID_WIDTH;
  var nearest = 100000000.0;
  for (var obstacle = 0u; obstacle < GRID_HEIGHT; obstacle++) {
    let delta = f32(row) - f32(obstacle);
    nearest = min(nearest, horizontal[obstacle * GRID_WIDTH + column] + delta * delta);
  }
  let boundary = min(min(f32(column) + 0.5, f32(row) + 0.5), min(f32(GRID_WIDTH) - 0.5 - f32(column), f32(GRID_HEIGHT) - 0.5 - f32(row))) * CELL_SIZE;
  field[index] = max(0.0, min(boundary, sqrt(nearest) * CELL_SIZE - CELL_SIZE * 0.7071067812));
}
