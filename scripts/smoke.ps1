$ErrorActionPreference = 'Stop'
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m lowrank_lab run --config configs/smoke.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m lowrank_lab analyze --root results --output reports/figures
