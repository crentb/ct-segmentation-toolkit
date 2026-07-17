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

## Flame-graph profile — `som_segment` (2026-07-15)

`som_segment` (the label-free SOM spectral–spatial segmenter) profiled
end-to-end with cProfile on a fixed-seed 512×512 synthetic texture slice;
flame graphs rendered from the cProfile data via flameprof's
collapsed-stack export and flamegraph.pl (classic bottom-up layout; the
SVGs scale to fit the browser window with click-to-zoom intact).
Apple-Silicon macOS, Python 3.13.

| | Baseline | After |
|---|---|---|
| Total wall time | 3.89 s | **2.31 s (1.68×)** |
| `MiniSom.winner` calls | 312,144 | 50,000 |

- Baseline: 59% of runtime sat in 312,144 Python-level `MiniSom.winner`
  calls — 262,144 from the per-pixel BMU mapping loop plus exactly 50,000
  from inside `train`. The pre-profiling suspect, the 18-kernel Gabor
  bank, measured 5% — the profile refuted the guess.
- Fix: the per-pixel loop became chunked BLAS matrix products — squared
  Euclidean distance via (a−b)² = a² − 2·a·b + b² (argmin unchanged since
  sqrt is monotonic), row-major argmin matching MiniSom's
  `unravel_index` tie-breaking exactly. Equivalence verified: identical
  BMU indices on 5,000 samples vs the original loop.
- After: only `train`'s internal 50,000 `winner` calls remain; the
  residual cost is SOM training itself — algorithmic and
  user-parameterized (`iterations`), deliberately left alone.

Flame graphs (open in a browser; box width is cumulative time, click to
zoom):

- [`profiling/flame_som_baseline.svg`](profiling/flame_som_baseline.svg)
- [`profiling/flame_som_after.svg`](profiling/flame_som_after.svg)

The matching `*.collapsed.txt` files alongside them import directly into
[speedscope](https://www.speedscope.app/) for interactive exploration
(fully client-side; nothing is uploaded).
