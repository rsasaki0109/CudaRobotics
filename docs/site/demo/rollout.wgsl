struct Params {
  pose: vec4f,
  goal: vec2f,
  count: u32,
  obstacle_count: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> controls: array<f32>;
@group(0) @binding(2) var<storage, read> obstacles: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> costs: array<f32>;

fn clearance(point: vec2f) -> f32 {
  var distance = min(min(point.x, point.y), min(10.0 - point.x, 10.0 - point.y)) - 0.22;
  for (var obstacle = 0u; obstacle < params.obstacle_count; obstacle++) {
    let bounds = obstacles[obstacle];
    let outside = max(max(bounds.xy - point, point - bounds.zw), vec2f(0.0));
    distance = min(distance, length(outside) - 0.22);
  }
  return distance;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3u) {
  let sample = invocation.x;
  if (sample >= params.count) { return; }
  var pose = params.pose.xyz;
  var cost = 0.0;
  var previous_angular = 0.0;
  for (var step = 0u; step < 32u; step++) {
    let index = (sample * 32u + step) * 2u;
    let velocity = controls[index];
    let angular = controls[index + 1u];
    for (var substep = 0u; substep < 2u; substep++) {
      let heading = pose.z + angular * 0.025;
      pose = vec3f(pose.xy + velocity * vec2f(cos(heading), sin(heading)) * 0.05,
        pose.z + angular * 0.05);
      if (clearance(pose.xy) <= 0.0) {
        costs[sample] = 1000000.0;
        return;
      }
    }
    let distance = length(pose.xy - params.goal);
    let margin = max(0.0, 0.65 - clearance(pose.xy));
    cost += 0.4 * distance + 8.0 * margin * margin + 0.04 * angular * angular
      + 0.08 * (angular - previous_angular) * (angular - previous_angular);
    previous_angular = angular;
  }
  let terminal_distance = length(pose.xy - params.goal);
  costs[sample] = cost + 12.0 * terminal_distance * terminal_distance;
}
