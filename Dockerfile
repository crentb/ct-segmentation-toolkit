# Minimal container that installs the package (core + dev tools) and can run the
# fast test suite. CPU-only torch wheel is pulled from PyPI. CI builds this image
# and runs `pytest -m "not slow"` inside it as a clean-environment smoke test.
FROM python:3.14-slim

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel "jaraco.context>=6.1.0" \
 && python -m pip install --no-cache-dir -e ".[dev]"

# Default: run the fast, pure-Python test suite.
CMD ["pytest", "-m", "not slow"]
