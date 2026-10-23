@group(0) @binding(0) var<storage, read> occupancy: array<u32>;
@group(0) @binding(1) var<storage, read_write> horizontal: array<f32>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;
override GRID_WIDTH: u32 = 128u;
override GRID_HEIGHT: u32 = 96u;
override CELL_SIZE: f32 = 0.125;

@compute @workgroup_size(32)
fn row_edt(@builtin(global_invocation_id) invocation: vec3u) {
  let row = invocation.x;
  if (row >= GRID_HEIGHT) { return; }
  var previous_site = -1;
  for (var column = 0u; column < GRID_WIDTH; column++) {
    let index = row * GRID_WIDTH + column;
    if (occupancy[index] != 0u) { previous_site = i32(column); }
    var distance = 100000000.0;
    if (previous_site >= 0) {
      let delta = f32(i32(column) - previous_site);
      distance = delta * delta;
    }
    horizontal[index] = distance;
  }
  previous_site = -1;
  for (var column = i32(GRID_WIDTH) - 1; column >= 0; column--) {
    let index = row * GRID_WIDTH + u32(column);
    if (occupancy[index] != 0u) { previous_site = column; }
    if (previous_site >= 0) {
      let delta = f32(previous_site - column);
      horizontal[index] = min(horizontal[index], delta * delta);
    }
  }
}

@compute @workgroup_size(32)
fn column_edt(@builtin(global_invocation_id) invocation: vec3u) {
  let column = invocation.x;
  if (column >= GRID_WIDTH) { return; }
  var sites: array<u32, 256>;
  var boundaries: array<f32, 257>;
  var last = -1;
  for (var candidate = 0u; candidate < GRID_HEIGHT; candidate++) {
    let cost = horizontal[candidate * GRID_WIDTH + column];
    if (cost >= 100000000.0) { continue; }
    var intersection = -1e20;
    loop {
      if (last < 0) { break; }
      let anchor = sites[u32(last)];
      intersection = ((cost + f32(candidate * candidate)) - (horizontal[anchor * GRID_WIDTH + column] + f32(anchor * anchor))) / (2.0 * f32(candidate - anchor));
      if (intersection > boundaries[u32(last)]) { break; }
      last--;
    }
    last++;
    sites[u32(last)] = candidate;
    boundaries[u32(last)] = select(intersection, -1e20, last == 0);
    boundaries[u32(last) + 1u] = 1e20;
  }
  var envelope_index = 0u;
  for (var row = 0u; row < GRID_HEIGHT; row++) {
    var nearest = 100000000.0;
    if (last >= 0) {
      loop {
        if (envelope_index >= u32(last) || boundaries[envelope_index + 1u] >= f32(row)) { break; }
        envelope_index++;
      }
      let anchor = sites[envelope_index];
      let delta = f32(i32(row) - i32(anchor));
      nearest = delta * delta + horizontal[anchor * GRID_WIDTH + column];
    }
    let boundary = min(min(f32(column) + 0.5, f32(row) + 0.5), min(f32(GRID_WIDTH) - 0.5 - f32(column), f32(GRID_HEIGHT) - 0.5 - f32(row))) * CELL_SIZE;
    field[row * GRID_WIDTH + column] = max(0.0, min(boundary, sqrt(nearest) * CELL_SIZE - CELL_SIZE * 0.7071067812));
  }
}
