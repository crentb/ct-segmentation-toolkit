"""The MLflow tracking wrapper must be a safe no-op when disabled or absent."""

from ct_seg import tracking


def test_is_available_returns_bool():
    assert isinstance(tracking.is_available(), bool)


def test_calls_are_noops_when_disabled():
    # None of these should raise, regardless of whether mlflow is installed.
    assert tracking.start(run_name="x", enabled=False) is False
    tracking.log_params({"a": 1}, enabled=False)
    tracking.log_metrics({"loss": 0.1}, step=1, enabled=False)
    tracking.log_artifact("/nonexistent/path.pth", enabled=False)
    tracking.end(enabled=False)
