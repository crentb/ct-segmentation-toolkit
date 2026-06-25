"""
Optional MLflow experiment tracking with a graceful no-op fallback.

MLflow is imported lazily (only when tracking is actually used and enabled), so simply
importing ``ct_seg`` never pays MLflow's import cost and the package has no hard MLflow
dependency. If MLflow is installed (``pip install -e ".[mlops]"``) and tracking is
enabled, the training script logs run parameters, per-epoch metrics, and the best
checkpoint. Otherwise every function here is a harmless no-op.
"""

from __future__ import annotations

import os

_UNSET = object()
_mlflow_mod = _UNSET  # cache: _UNSET (untried) -> module or None


def _mlflow():
    """Return the mlflow module if importable (lazily, cached), else None."""
    global _mlflow_mod
    if _mlflow_mod is _UNSET:
        try:
            import mlflow

            _mlflow_mod = mlflow
        except ImportError:  # pragma: no cover - only when mlflow is not installed
            _mlflow_mod = None
    return _mlflow_mod


def is_available() -> bool:
    """Return True if MLflow is importable."""
    return _mlflow() is not None


def start(run_name=None, enabled=True) -> bool:
    """Begin an MLflow run. No-op if disabled or MLflow is absent.

    Returns True if a real MLflow run was started, else False.
    """
    if not enabled:
        return False
    m = _mlflow()
    if m is not None:
        m.start_run(run_name=run_name)
        return True
    return False


def end(enabled=True) -> None:
    """End the active MLflow run, if any."""
    if not enabled:
        return
    m = _mlflow()
    if m is not None and m.active_run() is not None:
        m.end_run()


def log_params(params, enabled=True) -> None:
    """Log a dict of run parameters."""
    if not enabled:
        return
    m = _mlflow()
    if m is not None:
        m.log_params(params)


def log_metrics(metrics, step=None, enabled=True) -> None:
    """Log a dict of metrics, optionally at a given step (e.g., epoch)."""
    if not enabled:
        return
    m = _mlflow()
    if m is not None:
        m.log_metrics(metrics, step=step)


def log_artifact(path, enabled=True) -> None:
    """Log a file artifact (e.g., a checkpoint) if it exists."""
    if not enabled:
        return
    m = _mlflow()
    if m is not None and path and os.path.exists(path):
        m.log_artifact(path)
