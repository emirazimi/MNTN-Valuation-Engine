#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m mntn_valuation run \
  --company MELI \
  --data-dir data \
  --config configs/meli.json \
  --output-dir MELI/out \
  "$@"
