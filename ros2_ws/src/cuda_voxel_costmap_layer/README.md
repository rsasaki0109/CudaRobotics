# CUDA Voxel Costmap Layer

`cuda_voxel_costmap_layer::CudaVoxelCostmapLayer` is the supported bridge from
the CudaNav rolling `nav_msgs/OccupancyGrid` projection into a Nav2 layered
costmap.

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      global_frame: odom
      plugins: [cuda_voxel, inflation_layer]
      cuda_voxel:
        plugin: "cuda_voxel_costmap_layer::CudaVoxelCostmapLayer"
        enabled: true
        occupancy_topic: occupancy
        lethal_threshold: 50
        unknown_is_free: false
        use_maximum: false
        max_map_age_sec: 0.5
```

The topic must be relative. The occupancy frame must exactly match the Nav2
costmap global frame. Map origin yaw is supported; non-planar origins, malformed
shapes, non-standard occupancy values, and stale maps are rejected. Unknown
space maps to `NO_INFORMATION` by default and can explicitly map to free space.
When the Nav2 costmap node is nested below the mapper namespace, remap its
relative `occupancy` subscription to the mapper's `/cuda_nav/occupancy` topic
in the stack launch file.

`use_maximum: false` makes this layer authoritative inside the received rolling
map. Set it to `true` when another layer must retain a higher cost.
