# Acknowledgments — `ct_seg.denoise`

The denoising subpackage is Cameron Renteria's own implementation of the **Noise2Inverse
(N2I)** self-supervised tomography-denoising framework. The original method is due to:

> A. A. Hendriksen, D. M. Pelt, and K. J. Batenburg,
> "Noise2Inverse: Self-Supervised Deep Convolutional Denoising for Tomography,"
> *IEEE Transactions on Computational Imaging*, vol. 6, pp. 1320-1335, 2020.
> doi: 10.1109/TCI.2020.3019647
>
> Original code: https://github.com/ahendriksen/noise2inverse

This subpackage is **original code** (Copyright 2026 Cameron Renteria, licensed Apache-2.0)
that implements and extends the N2I framework. Cameron's contributions include:

- a **2.5D** input (five adjacent slices in, one denoised slice out),
- a **Laplacian Contrast Loss (LCL)** that preserves edges while smoothing flat regions,
- **edge-aware model selection** (keeping the checkpoint with the best Laplacian edge score),
- **automatic GPU batch-size optimization** for memory-safe inference, and
- the surrounding training/inference pipeline and TIFF I/O.

If you use this code, please cite the original N2I paper above in addition to this software.
