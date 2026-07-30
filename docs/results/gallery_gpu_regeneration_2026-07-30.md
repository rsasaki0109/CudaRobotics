# GPU gallery regeneration — 2026-07-30

Status: **PASS**

Seven README gallery links had no retained source GIF. The corresponding CUDA
demos were rebuilt and executed from clean source commit
`6bf402ba7ded5fc56142a60a069f8794f52f6bb0` on an NVIDIA GeForce GTX 1660 Ti
(`GPU-f635286a-d68f-5039-cbc9-22d7f295b3a3`, driver 596.36).

Environment: Ubuntu 24.04 under WSL2, CUDA 12.6.85 targeting sm_75, OpenCV
4.6.0, and FFmpeg 6.1.1.

| Demo | Size | Frames | Resolution | SHA-256 |
|---|---:|---:|---:|---|
| `gpu_batched_ilqr` | 1,366,802 B | 39 | 900x506 | `70a7871312840843c274ab7dfbf0cc93b9e9ebd24a54a984d821e4b4ec938c02` |
| `gpu_correlative_scan_matching` | 2,889,784 B | 44 | 760x280 | `125f8ea4274585c1e7ea963289cfe394b793ceb7717ee1939cfbf6a41bc1420e` |
| `gpu_csm_submap_slam` | 1,543,194 B | 207 | 900x325 | `1e5497e1586605c106a6277afae910197bcbc624d33306e847956aa270710e79` |
| `gpu_jfa_edt` | 2,308,918 B | 12 | 640x640 | `fd7c1e9fde918b9d7f3dfc557118dd2fecb7046a8071ff5e8cf28e35db705198` |
| `gpu_lk_optical_flow` | 509,255 B | 24 | 760x600 | `1d1fef5facd75189e39c8daf51516089804732314ed131dde18561037a7ad87c` |
| `gpu_marching_cubes` | 1,017,440 B | 36 | 760x600 | `952c429ce438e6444c243f01ebec77708538cf47896cf94e74524c20489d0d4a` |
| `gpu_tsdf_fusion` | 940,037 B | 24 | 760x600 | `a3d4834218605c1c869bbd609239a791426f39b7a5f72f7a65d80550df6461b1` |

Every target built and ran successfully. `file` identified every output as
GIF89a, FFprobe decoded every animation and frame count, and Pillow confirmed
that the first and last RGB frames differ for all seven outputs. Representative
frames were also inspected visually.

Machine-readable provenance and the complete artifact table are retained in
[`gallery_gpu_regeneration_2026-07-30.json`](gallery_gpu_regeneration_2026-07-30.json).
