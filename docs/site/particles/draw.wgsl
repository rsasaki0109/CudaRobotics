struct Params { motion: vec4f, settings: vec4u, screen: vec4f }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<vec4f>;
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f, @location(1) strength: f32 }
@vertex
fn vertex_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Vertex {
  let corners = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1));
  let pose = particles[instance];
  let corner = corners[vertex];
  let position = pose.xy / vec2f(16,12) * 2.0 - 1.0;
  let radius = 2.3 / params.screen.zw * 2.0;
  return Vertex(vec4f(position + corner * radius, 0, 1), corner, 0.04 + sqrt(pose.w) * 0.2);
}
@fragment
fn fragment_main(input: Vertex) -> @location(0) vec4f {
  let intensity = exp(-dot(input.uv, input.uv) * 3.5) * input.strength;
  return vec4f(vec3f(0.2, 1.0, 0.65) * intensity, intensity);
}
