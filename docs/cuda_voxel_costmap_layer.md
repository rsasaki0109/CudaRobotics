# CudaNav Nav2 Costmap Bridge

The `cuda_voxel_costmap_layer` ROS 2 package converts the standard
`nav_msgs/OccupancyGrid` projection from `cuda_voxel_mapping` into the layered
costmap consumed by `cuda_mppi_controller`.

## Contract

- The subscription is reliable, transient-local, depth 1.
- The default topic is relative and can be remapped by the stack launch.
  A fully qualified parameter is accepted for Nav2 child costmap nodes whose
  namespace cannot address a sibling mapper with a relative name.
- The occupancy frame must exactly match the Nav2 costmap global frame.
- Width, height, resolution, origin quaternion, data length, and every value in
  `[-1, 100]` are validated before a map becomes current.
- The complete planar map origin is applied. A master-grid cell center is
  inverse-rotated and translated into the source occupancy grid before lookup.
- `-1` becomes `NO_INFORMATION` by default, `0` becomes `FREE_SPACE`, values at
  or above `lethal_threshold` become `LETHAL_OBSTACLE`, and intermediate values
  are scaled below the inscribed cost.
- A malformed, frame-mismatched, or stale input marks the plugin non-current.
  The last map is not silently relabelled as fresh.
- Bounds from every unprocessed rolling-map position are accumulated so a fast
  sequence of origin shifts cannot leave an old obstacle region dirty.
- `footprint_clearing_radius` clears only the declared disc around the current
  robot pose before inflation. CudaNav uses 0.30 m, preventing self-occupancy
  from invalidating every controller rollout without erasing nearby obstacles.

The default authoritative merge writes source values into the master grid.
`use_maximum: true` is available when earlier layers must retain larger costs.
Inflation should be listed after this plugin.

## Verification

`occupancy_bridge_test` checks schema rejection, standard occupancy conversion,
a 90-degree rotated map sampled at cell centers, and the exact circular
footprint-clearing boundary. `plugin_load_test` loads
the class through pluginlib using the same base interface as Nav2. Both are
included in the ROS Jazzy workflow; the branch still requires that workflow to
run before this component is release evidence.
