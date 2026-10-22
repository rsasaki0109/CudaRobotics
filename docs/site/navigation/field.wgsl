@group(0) @binding(0) var<storage, read> occupancy: array<u32>;
@group(0) @binding(1) var<storage, read_write> horizontal: array<f32>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;

@compute @workgroup_size(64)
fn row_distance(@builtin(global_invocation_id) invocation: vec3u) {
  let index = invocation.x;
  if (index >= 12288u) { return; }
  let column = index % 128u;
  let row = index / 128u;
  var nearest = 100000000.0;
  for (var obstacle = 0u; obstacle < 128u; obstacle++) {
    if (occupancy[row * 128u + obstacle] != 0u) {
      let delta = f32(column) - f32(obstacle);
      nearest = min(nearest, delta * delta);
    }
  }
  horizontal[index] = nearest;
}

@compute @workgroup_size(64)
fn column_distance(@builtin(global_invocation_id) invocation: vec3u) {
  let index = invocation.x;
  if (index >= 12288u) { return; }
  let column = index % 128u;
  let row = index / 128u;
  var nearest = 100000000.0;
  for (var obstacle = 0u; obstacle < 96u; obstacle++) {
    let delta = f32(row) - f32(obstacle);
    nearest = min(nearest, horizontal[obstacle * 128u + column] + delta * delta);
  }
  let boundary = min(min(f32(column) + 0.5, f32(row) + 0.5), min(127.5 - f32(column), 95.5 - f32(row))) * 0.125;
  field[index] = max(0.0, min(boundary, sqrt(nearest) * 0.125 - 0.08838834765));
}
