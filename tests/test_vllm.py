"""vLLM backend checks. Run on a GPU in .venv-vllm with -m gpu."""

import pytest

from sondekit import backends
from sondekit import config

pytestmark = pytest.mark.gpu

CFG = config.ModelConfig(
    name="Qwen/Qwen3-0.6B",
    backend="vllm",
    vllm={"gpu_memory_utilization": 0.15, "max_model_len": 1024},
)


def test_load():
    import vllm  # pyright: ignore[reportMissingImports]

    loaded = backends.load_model(CFG)
    assert loaded.backend == "vllm"
    assert (loaded.num_layers, loaded.hidden) == (28, 1024)
    assert loaded.engine == f"vllm=={vllm.__version__}+nnsight==b717807"


def test_refuses_unlisted_layers():
    # Last: a refused build leaves nnsight's client process group up.
    gpt2 = config.ModelConfig(
        name="gpt2", backend="vllm", vllm={"gpu_memory_utilization": 0.1}
    )
    with pytest.raises(ValueError, match="GPT2"):
        backends.load_model(gpt2)
