struct Params {
  motion: vec4f,
  settings: vec4u,
  screen: vec4f,
}
struct Stats { totals: vec4f, moments: vec4f }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> scored: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> cdf: array<f32>;
@group(0) @binding(4) var<storage, read_write> groups: array<Stats>;
@group(0) @binding(5) var<storage, read_write> destination: array<vec4f>;
@group(0) @binding(6) var<storage, read> ranges: array<f32>;
@group(0) @binding(7) var<storage, read_write> summary: array<vec4f>;
@group(0) @binding(8) var<storage, read> boxes: array<vec4f>;
var<workgroup> prefix: array<f32, 256>;
var<workgroup> totals: array<vec4f, 256>;
var<workgroup> moments: array<vec4f, 256>;

fn hash(value: u32) -> u32 {
  var result = value;
  result = (result ^ (result >> 16u)) * 2146121005u;
  result = (result ^ (result >> 15u)) * 2221713035u;
  return result ^ (result >> 16u);
}
fn random(value: u32) -> f32 { return f32(hash(value) & 16777215u) / 16777216.0; }
fn is_free(point: vec2f) -> bool {
  if (any(point <= vec2f(0.0)) || any(point >= vec2f(16.0, 12.0))) { return false; }
  for (var obstacle = 0u; obstacle < 3u; obstacle++) {
    let bounds = boxes[obstacle];
    if (all(point >= bounds.xy) && all(point <= bounds.zw)) { return false; }
  }
  return true;
}
fn surface_distance(point: vec2f) -> f32 {
  var distance = min(min(abs(point.x), abs(point.y)), min(abs(16.0 - point.x), abs(12.0 - point.y)));
  for (var obstacle = 0u; obstacle < 3u; obstacle++) {
    let bounds = boxes[obstacle];
    let delta = max(bounds.xy - point, point - bounds.zw);
    let signed_distance = length(max(delta, vec2f(0.0))) + min(max(delta.x, delta.y), 0.0);
    distance = min(distance, abs(signed_distance));
  }
  return distance;
}

@compute @workgroup_size(256)
fn predict_score(@builtin(global_invocation_id) invocation: vec3u) {
  let index = invocation.x;
  if (index >= params.settings.x) { return; }
  let seed = hash(index + params.settings.z * 747796405u);
  var pose = source[index].xyz;
  if (params.settings.w == 1u || random(seed) < params.motion.w) {
    pose = vec3f(0.2 + random(seed + 1u) * 15.6, 0.2 + random(seed + 2u) * 11.6, random(seed + 3u) * 6.283185307);
  } else {
    let translation = params.motion.x + (random(seed + 4u) - 0.5) * 0.07;
    let rotation = params.motion.y + (random(seed + 5u) - 0.5) * 0.05;
    let heading = pose.z + rotation * 0.5;
    pose = vec3f(pose.xy + translation * vec2f(cos(heading), sin(heading)), pose.z + rotation);
  }
  var weight = 0.00000001;
  if (is_free(pose.xy)) {
    var error = 0.0;
    for (var beam = 0u; beam < 32u; beam++) {
      let angle = pose.z + f32(beam) / 32.0 * 6.283185307;
      let endpoint = pose.xy + vec2f(cos(angle), sin(angle)) * ranges[beam];
      let distance = surface_distance(endpoint);
      error += min(4.0, distance * distance);
    }
    weight += exp(-error / 32.0 / (2.0 * params.motion.z * params.motion.z));
  }
  scored[index] = vec4f(pose, weight);
}

fn inclusive_scan(local: u32) {
  workgroupBarrier();
  for (var offset = 1u; offset < 256u; offset *= 2u) {
    var add = 0.0;
    if (local >= offset) { add = prefix[local - offset]; }
    workgroupBarrier();
    prefix[local] += add;
    workgroupBarrier();
  }
}
fn reduce_stats(local: u32) {
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local < stride) {
      totals[local] += totals[local + stride];
      moments[local] += moments[local + stride];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn scan_particles(@builtin(global_invocation_id) invocation: vec3u,
  @builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3u) {
  let index = invocation.x;
  var pose = vec4f(0.0);
  if (index < params.settings.x) { pose = scored[index]; }
  prefix[local] = pose.w;
  totals[local] = vec4f(pose.w, pose.w * pose.w, pose.x * pose.w, pose.y * pose.w);
  moments[local] = vec4f(sin(pose.z) * pose.w, cos(pose.z) * pose.w, 0.0, 0.0);
  inclusive_scan(local);
  if (index < params.settings.x) { cdf[index] = prefix[local]; }
  reduce_stats(local);
  if (local == 0u) { groups[group.x] = Stats(totals[0], moments[0]); }
}

@compute @workgroup_size(256)
fn scan_groups(@builtin(local_invocation_index) local: u32) {
  let count = (params.settings.x + 255u) / 256u;
  var stats = Stats(vec4f(0.0), vec4f(0.0));
  if (local < count) { stats = groups[local]; }
  prefix[local] = stats.totals.x;
  totals[local] = stats.totals;
  moments[local] = stats.moments;
  inclusive_scan(local);
  if (local < count) { groups[local].totals.x = prefix[local]; }
  reduce_stats(local);
  if (local == 0u) {
    let total = totals[0].x;
    summary[0] = vec4f(totals[0].zw / total, atan2(moments[0].x, moments[0].y), total * total / totals[0].y);
    summary[1] = vec4f(total, totals[0].y, 0.0, 0.0);
  }
}

@compute @workgroup_size(256)
fn resample(@builtin(global_invocation_id) invocation: vec3u) {
  let index = invocation.x;
  let count = params.settings.x;
  if (index >= count) { return; }
  let threshold = (f32(index) + random(index + params.settings.z * 2891336453u)) / f32(count) * summary[1].x;
  var low = 0u;
  var high = (count + 255u) / 256u - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (groups[middle].totals.x < threshold) { low = middle + 1u; } else { high = middle; }
  }
  let group = low;
  var previous = 0.0;
  if (group > 0u) { previous = groups[group - 1u].totals.x; }
  let within = max(0.0, threshold - previous);
  low = group * 256u;
  high = min(count, low + 256u) - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (cdf[middle] < within) { low = middle + 1u; } else { high = middle; }
  }
  destination[index] = scored[low];
}
