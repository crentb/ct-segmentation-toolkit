# Contributing

Thanks for your interest in `ct-segmentation-toolkit`.

## Development setup

```bash
git clone https://github.com/crentb/ct-segmentation-toolkit.git
cd ct-segmentation-toolkit
python -m pip install -e ".[dev]"     # core + dev tools (pytest, ruff, black, mypy, pre-commit)
pre-commit install                    # run ruff/black/mypy on every commit
```

The interactive labeling/viewing tools need the optional viz stack:

```bash
python -m pip install -e ".[viz]"     # napari + pyvista
```

## Checks (what CI runs)

Run these before opening a PR; CI runs the same on Python 3.10-3.12:

```bash
ruff check .                  # lint
black --check .               # formatting
mypy -p ct_seg                # type-checking (advisory)
pytest -m "not slow" --cov    # fast tests with coverage
```

Dev-tool versions (ruff/black/mypy) are pinned in `pyproject.toml` so local and CI agree
exactly.

## Pull requests

1. Branch from `main`.
2. Add or update tests for any behavior change (tests use small synthetic arrays, so they
   stay fast and need no real data).
3. Update `CHANGELOG.md` under `[Unreleased]`.
4. Ensure the checks above pass locally.

## License

By contributing, you agree your contributions are licensed under the project's
**Apache-2.0** license (see `LICENSE` and `NOTICE`).
