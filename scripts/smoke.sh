#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python -m lowrank_lab run --config configs/smoke.yaml
python -m lowrank_lab analyze --root results --output reports/figures
