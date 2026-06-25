"""
Performance profile of the U-Net: parameter count, forward and forward+backward
throughput, and peak memory, on CPU or CUDA. Writes a small markdown report so the
"GPU acceleration and performance profiling" story has a concrete, reproducible artifact.

Usage:
    python scripts/profile_unet.py                       # CPU (or CUDA if available), default sizes
    python scripts/profile_unet.py --device cuda --sizes 256 512
    python scripts/profile_unet.py --out docs/PERF.md
"""

from __future__ import annotations

import argparse
import platform

import torch

from ct_seg.model import UNetSegmentation, count_parameters


def _time_ms(fn, iters, is_cuda):
    """Average wall-clock ms per call over `iters`, with CUDA sync if needed."""
    import time

    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def profile(device="cpu", sizes=(128, 256), num_classes=4, base_filters=32, iters=5, warmup=2):
    """Return (resolved_device, n_params, rows) where each row is a dict of measurements."""
    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")

    model = UNetSegmentation(in_channels=1, num_classes=num_classes, base_filters=base_filters).to(
        dev
    )
    n_params = count_parameters(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    rows = []
    for hw in sizes:
        x = torch.randn(1, 1, hw, hw, device=dev)
        target = torch.randint(0, num_classes, (1, hw, hw), device=dev)

        for _ in range(warmup):
            model(x)

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        def _forward():
            with torch.no_grad():
                model(x)

        def _step():
            optimizer.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()

        fwd_ms = _time_ms(_forward, iters, use_cuda)
        step_ms = _time_ms(_step, iters, use_cuda)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else None

        rows.append(
            {
                "size": f"{hw}x{hw}",
                "forward_ms": round(fwd_ms, 2),
                "train_step_ms": round(step_ms, 2),
                "fwd_throughput_fps": round(1000.0 / fwd_ms, 1) if fwd_ms else 0.0,
                "peak_mem_mb": round(peak_mb, 1) if peak_mb is not None else None,
            }
        )
    return dev, n_params, rows


def render_markdown(dev, n_params, rows, num_classes, base_filters):
    """Render a markdown report string from the profiling results."""
    mem_note = "" if dev.type == "cuda" else " (peak GPU memory shown only on CUDA)"
    lines = [
        "# Performance Profile",
        "",
        "Reproduce with `python scripts/profile_unet.py` (optionally `--device cuda`).",
        "",
        f"- Device: `{dev.type}`",
        f"- Platform: `{platform.platform()}`",
        f"- PyTorch: `{torch.__version__}`",
        f"- Model: U-Net, num_classes={num_classes}, base_filters={base_filters}, "
        f"parameters={n_params:,}",
        "",
        f"| Input | Forward (ms) | Train step (ms) | Forward throughput (img/s) | Peak mem (MB){mem_note} |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        mem = r["peak_mem_mb"] if r["peak_mem_mb"] is not None else "-"
        lines.append(
            f"| {r['size']} | {r['forward_ms']} | {r['train_step_ms']} | "
            f"{r['fwd_throughput_fps']} | {mem} |"
        )
    lines += [
        "",
        "Notes: timings are wall-clock averages with warmup; CUDA timings include "
        "`torch.cuda.synchronize()`. Run on a CUDA machine to populate the GPU memory "
        "column and measure GPU throughput.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Profile the U-Net (CPU/CUDA).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out", default=None, help="Write the markdown report to this path")
    args = parser.parse_args()

    dev, n_params, rows = profile(
        device=args.device,
        sizes=tuple(args.sizes),
        num_classes=args.num_classes,
        base_filters=args.base_filters,
        iters=args.iters,
    )
    report = render_markdown(dev, n_params, rows, args.num_classes, args.base_filters)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
