# Performance Profile

Reproduce with `python scripts/profile_unet.py` (optionally `--device cuda`).

- Device: `cpu`
- Platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`
- PyTorch: `2.11.0`
- Model: U-Net, num_classes=4, base_filters=32, parameters=7,852,068

| Input | Forward (ms) | Train step (ms) | Forward throughput (img/s) | Peak mem (MB) (peak GPU memory shown only on CUDA) |
|---|---|---|---|---|
| 128x128 | 27.09 | 97.89 | 36.9 | - |
| 256x256 | 103.28 | 318.23 | 9.7 | - |

Notes: timings are wall-clock averages with warmup; CUDA timings include `torch.cuda.synchronize()`. Run on a CUDA machine to populate the GPU memory column and measure GPU throughput.
