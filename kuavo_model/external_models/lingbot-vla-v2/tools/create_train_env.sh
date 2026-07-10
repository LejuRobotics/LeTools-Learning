#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1

ENV_NAME="lingbotvla2"
RECREATE=0
RESUME=0
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"

usage() {
  cat <<'USAGE'
Usage: bash tools/create_train_env.sh [--env-name NAME] [--recreate] [--resume] [--flash-attn-wheel PATH]

Creates a clean Python 3.12 conda environment for lingbotvla2 training.
Depth dependencies and local depth packages are always installed.
If --flash-attn-wheel or FLASH_ATTN_WHEEL is provided, flash-attn is installed
from that wheel. Otherwise flash-attn==2.8.3 is installed from pip.
Use --resume to continue installing into an existing environment.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:?--env-name requires a value}"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --flash-attn-wheel)
      FLASH_ATTN_WHEEL="${2:?--flash-attn-wheel requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_BASE="$(conda info --base)"
eval "$(conda shell.bash hook)"

ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
ENV_EXISTS=0
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  ENV_EXISTS=1
fi

if [[ "${ENV_EXISTS}" == "1" || -d "${ENV_PREFIX}" ]]; then
  if [[ "${RECREATE}" == "1" ]]; then
    if [[ "${ENV_EXISTS}" == "1" ]]; then
      conda env remove -n "${ENV_NAME}" -y
    fi
    if [[ -d "${ENV_PREFIX}" ]]; then
      case "${ENV_PREFIX}" in
        "${CONDA_BASE}/envs/"*) rm -rf "${ENV_PREFIX}" ;;
        *) echo "Refusing to remove unexpected env prefix: ${ENV_PREFIX}" >&2; exit 1 ;;
      esac
    fi
    ENV_EXISTS=0
  elif [[ "${RESUME}" == "1" ]]; then
    echo "Resuming install in existing conda env: ${ENV_NAME}"
  else
    echo "Conda env already exists: ${ENV_NAME}" >&2
    echo "Pass --resume to continue installing into it, or --recreate to remove and rebuild it." >&2
    exit 1
  fi
fi

if [[ "${RESUME}" != "1" || "${ENV_EXISTS}" != "1" ]]; then
  conda create -n "${ENV_NAME}" python=3.12 pip -y
fi
conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

assert_torch_stack() {
  python - <<'PY'
import torch

expected = "2.8.0"
actual = torch.__version__.split("+", 1)[0]
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert actual == expected, f"torch version changed: expected {expected}, got {torch.__version__}"
assert torch.cuda.is_available(), "torch CUDA is not available"
PY
}

python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  torchdata==0.11.0 torchcodec==0.6.0
assert_torch_stack

python -m pip install -r "${REPO_ROOT}/requirements.txt"
assert_torch_stack

python -m pip install numpydantic==1.9.0 --no-deps
assert_torch_stack

if [[ -n "${FLASH_ATTN_WHEEL}" ]]; then
  if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
    echo "flash-attn wheel not found: ${FLASH_ATTN_WHEEL}" >&2
    exit 1
  fi
  python -m pip install --no-deps "${FLASH_ATTN_WHEEL}"
else
  python -m pip install --no-build-isolation flash-attn==2.8.3
fi
assert_torch_stack
python - <<PY
import flash_attn
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
PY

python -m pip install --no-deps \
  "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
assert_torch_stack

python -m pip install -e "${REPO_ROOT}" --no-deps
assert_torch_stack

python -m pip install -r "${REPO_ROOT}/requirements-depth.txt"
assert_torch_stack
# mlflow/depth dependencies can pull broad transitive requirements. Restore
# stableVLA's pinned core stack after resolving those runtime dependencies.
python -m pip install -r "${REPO_ROOT}/requirements.txt"
python -m pip install numpydantic==1.9.0 --no-deps
assert_torch_stack
# Follow the open-source lingbot-vla layout, but do not let local depth
# packages resolve their own broad dependencies. MoGe's pyproject allows
# unpinned huggingface_hub/numpy/opencv/gradio, which breaks the training pins.
# Its optional train/test dataloader imports `pipeline`, but LingBot training
# only needs the MoGe model code, so skip that PyPI-only dependency here.
python - <<PY
import site
from pathlib import Path

site_packages = Path(site.getsitepackages()[0])
pth = site_packages / "stablevla_local_depth.pth"
pth.write_text("${REPO_ROOT}/lingbotvla/models/vla/vision_models/morgbd_clean/3rd/utils3d\n")
print("wrote", pth)
PY
python -m pip install -e "${REPO_ROOT}/lingbotvla/models/vla/vision_models/lingbot-depth" --no-deps
python -m pip install -e "${REPO_ROOT}/lingbotvla/models/vla/vision_models/MoGe"
assert_torch_stack

python - <<'PY'
import cv2
import accelerate
import mlflow
import trimesh
import moge
import mdm
import utils3d

print("depth imports ok")
PY
python -m pip install huggingface_hub==0.34.0
if ! python -m pip check; then
  echo "[WARN] pip check reported dependency metadata issues." >&2
  echo "[WARN] lerobot and depth subpackages are installed with --no-deps intentionally to preserve training pins." >&2
fi

echo "Environment ready: ${ENV_NAME}"
