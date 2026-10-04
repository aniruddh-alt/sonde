#!/usr/bin/env bash
# Spec §13 check 1: nnsight's own vLLM tests, at the pinned nnsight commit,
# against one vLLM version. On the Spark, from ~/experiments/sonde-spike/code:
#   bash scripts/spike/smoke.sh 0.30.0
set -eu
v=$1
sha=b71780727ea9713f74ce12f76b1d3548b91a74a0
uv=~/.local/bin/uv
venv=../.venv-smoke-$v
mkdir -p ../outputs/spike
if [ ! -d ../nnsight ]; then
  git clone -q https://github.com/ndif-team/nnsight ../nnsight
  git -C ../nnsight checkout -q $sha
  # tests/vllm/conftest.py puts src/ first on sys.path, and src/ has no built
  # C extension; without src/ the tests import the installed same-commit copy.
  rm -rf ../nnsight/src
fi
if [ ! -d $venv ]; then
  $uv venv $venv --python 3.12 --managed-python
  $uv pip install --python $venv/bin/python "vllm==$v" ninja pytest \
    "nnsight[vllm] @ git+https://github.com/ndif-team/nnsight@$sha"
  cp ~/experiments/.venv/lib/python3.12/site-packages/sitecustomize.py \
    $venv/lib/python3.12/site-packages/
fi
$venv/bin/python -c "import nnsight._c, vllm; print('vllm', vllm.__version__)"
cd ../nnsight
$venv/bin/python -m pytest tests/vllm/test_registration.py \
  tests/vllm/test_tracing.py tests/vllm/test_chunked_prefill.py -q 2>&1 \
  | tee ../outputs/spike/smoke-vllm-$v.txt
