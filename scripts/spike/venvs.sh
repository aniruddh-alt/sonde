#!/usr/bin/env bash
# Builds the Spark's two sonde venvs from uv.lock (spec §12). On the Spark,
# from ~/experiments/sonde-spike/code: bash scripts/spike/venvs.sh
set -eu
uv=~/.local/bin/uv
for extra in hf vllm; do
  venv=../.venv-$extra
  UV_PROJECT_ENVIRONMENT=$venv $uv sync --locked --extra $extra --extra dev \
    --python 3.12 --managed-python
  cp ~/experiments/.venv/lib/python3.12/site-packages/sitecustomize.py \
    $venv/lib/python3.12/site-packages/
  $venv/bin/python -c "import torch; print('$extra', torch.__version__,
    torch.cuda.is_available(), torch.cuda.get_device_capability())"
done
