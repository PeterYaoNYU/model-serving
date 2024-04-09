## Build


```bash
# Conda
mamba create -y -n framework python=3.10
mamba activate framework

# Punica Requirement: CUDA 12.1, PyTorch 2.1
# GPU Requirement: >= sm_80 (e.g., RTX 4090, RTX A6000, A100, A10G, L4)
pip install ninja torch
git submodule sync
git submodule update --init --recursive
pip install -v --no-build-isolation third_party/punica

# Framework
pip install -v --no-build-isolation -e .[dev]
```

## Example

```bash
python examples/textgen-api-demo.py
```
