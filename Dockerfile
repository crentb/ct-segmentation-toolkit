# Minimal container that installs the package (core + dev tools) and can run the
# fast test suite. CPU-only torch wheel is pulled from PyPI. CI builds this image
# and runs `pytest -m "not slow"` inside it as a clean-environment smoke test.
FROM python:3.11-slim

# Apply Debian security updates on top of the base image.
# The upstream python:*-slim tag is rebuilt on Docker's own cadence (the tag
# current when this was added was last pushed 2026-09-02), so a freshly pulled
# base image can still contain OS packages for which Debian has ALREADY
# published fixed versions. The CI container gate scans CRITICAL/HIGH with
# ignore-unfixed:true, so those already-fixed-upstream packages are exactly
# what trips it: the 2026-09-14 weekly re-scan failed on 12 findings
# (9 HIGH / 3 CRITICAL) in gzip, libpcre2-8-0, libsqlite3-0 and perl-base,
# every one of them fixed in the Debian archive but not yet in the base image.
# Upgrading here closes the window between Debian shipping a fix and Docker
# rebuilding the base, so the weekly gate stops going red each time that gap
# opens. `upgrade` (not `dist-upgrade`) keeps this to in-place version bumps
# within the stable release: no packages are added or removed.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel "jaraco.context>=6.1.0" \
 && python -m pip install --no-cache-dir -e ".[dev]"

# Default: run the fast, pure-Python test suite.
CMD ["pytest", "-m", "not slow"]
