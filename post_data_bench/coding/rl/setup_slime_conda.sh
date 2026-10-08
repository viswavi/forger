#!/usr/bin/env bash
# Build Slime in a separate env using pins from $SLIME_DIR/build_conda.sh.
# Its Torch/CUDA stack is incompatible with TRL’s; keep the envs separate.
# Steps resume from $BASE_DIR/.stamps; keep build/cache storage off $HOME.
# Usage: ./rl/setup_slime_conda.sh [--check-only | --redo STEP]
# Apex, FlashQLA (SM90+ GDN), and int4_qat are optional and skipped.
# Set INSTALL_APEX=1, INSTALL_FLASHQLA=1, or INSTALL_INT4QAT=1 to enable them.
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-${SLIME_ENV:-/home/vijayv/miniconda3/envs/slime}}"
BASE_DIR="${BASE_DIR:-/data/user_data/vijayv/slime_deps}"
SLIME_DIR="${SLIME_DIR:-/home/vijayv/slime}"

# Keep in sync with $SLIME_DIR/build_conda.sh.
SGLANG_COMMIT="0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
PATCH_VERSION="v0.5.15.post1"
TMS_COMMIT="8d30c59ca12a68d9deccbc9c6599076a1218cbc5"

# Use upstream’s micromamba CUDA solver; classic conda exhausts memory.
INSTALL_APEX="${INSTALL_APEX:-0}"
INSTALL_FLASHQLA="${INSTALL_FLASHQLA:-0}"
INSTALL_INT4QAT="${INSTALL_INT4QAT:-0}"

CHECK_ONLY=0
REDO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) CHECK_ONLY=1; shift ;;
    --env-prefix) ENV_PREFIX="$2"; shift 2 ;;
    --base-dir) BASE_DIR="$2"; shift 2 ;;
    --redo) REDO="$2"; shift 2 ;;
    *) echo "!!! unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Resolved once, absolutely: smoke_test cd's to $BASE_DIR, after which a
# relative "$0" no longer points anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

STAMPS="$BASE_DIR/.stamps"
mkdir -p "$BASE_DIR" "$STAMPS"

# See the note above install_cuda on why this exists.
MICROMAMBA="$BASE_DIR/bin/micromamba"
PROTOC_BIN="$BASE_DIR/bin/protoc"

# $HOME has ~27G free; pip's wheel cache and build scratch both belong on /data.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BASE_DIR/.pip_cache}"
# flashinfer defaults its JIT scratch to ~/.cache/flashinfer; $HOME is at 92%.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$BASE_DIR/.flashinfer}"
export TMPDIR="${TMPDIR:-$BASE_DIR/.tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# Build only SM80/86/89 variants used here. Initialize NVCC flags before
# activation because cuda-nvcc hooks read them under set -u.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export NVCC_APPEND_FLAGS="--threads ${MAX_JOBS}"

if [[ -n "$REDO" ]]; then
  grep -qE "^step +${REDO//[^a-zA-Z0-9_]/INVALID} +" "$0" || { echo "Unknown --redo step: $REDO" >&2; exit 2; }
  rm -f "$STAMPS/$REDO"
fi

step() {
  local name="$1"; shift
  if [ -f "$STAMPS/$name" ]; then
    echo ">>> [skip] $name (stamped $(cat "$STAMPS/$name"))"
    return 0
  fi
  echo ">>> [run ] $name"
  "$@"
  date -Is > "$STAMPS/$name"
  echo ">>> [done] $name"
}

# `conda` is aliased to a nonexistent /home/vijayv/anaconda3/bin/conda in this
# user's interactive shells. Aliases don't apply in a non-interactive script, but
# resolve the base from $CONDA_EXE anyway so this cannot depend on that.
CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE:-$(command -v conda)}")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
create_env() {
  conda create -y -p "$ENV_PREFIX" python=3.12 pip
}

fetch_micromamba() {
  # Solve CUDA with micromamba to avoid classic conda’s memory growth.
  mkdir -p "$BASE_DIR/bin"
  [ -x "$MICROMAMBA" ] || curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/2.9.0 \
    | tar -xj -C "$BASE_DIR" bin/micromamba
  "$MICROMAMBA" --version
}

install_cuda() {
  "$MICROMAMBA" install -y --prefix "$ENV_PREFIX" \
    cuda=12.9.1 cuda-nvtx=12.9.79 cuda-nvtx-dev=12.9.79 nccl \
    -c nvidia/label/cuda-12.9.1 -c nvidia -c conda-forge
  # sglang's editable install builds a Rust extension (sglang-grpc, via
  # setuptools-rust), so the env needs a working rustc + cargo of its own --
  # the host has rustc but no cargo.
  "$MICROMAMBA" install -y --prefix "$ENV_PREFIX" -c conda-forge cudnn rust
}

fetch_protoc() {
  # sglang-grpc needs protoc during its Rust build. Install a standalone
  # binary to avoid conflicting with the env’s pip protobuf runtime.
  mkdir -p "$BASE_DIR/bin"
  if [ ! -x "$PROTOC_BIN" ]; then
    local ver url
    ver="36.1"
    url="https://github.com/protocolbuffers/protobuf/releases/download/v${ver}/protoc-${ver}-linux-x86_64.zip"
    echo "    fetching protoc ${ver}"
    curl -fsSL "$url" -o "$TMPDIR/protoc.zip"
    python3 -c "
import zipfile, sys, os, stat
z = zipfile.ZipFile('$TMPDIR/protoc.zip')
with open('$PROTOC_BIN', 'wb') as f:
    f.write(z.read('bin/protoc'))
os.chmod('$PROTOC_BIN', 0o755)
"
  fi
  "$PROTOC_BIN" --version
}

clone_sglang() {
  [ -d "$BASE_DIR/sglang" ] || git clone https://github.com/sgl-project/sglang.git "$BASE_DIR/sglang"
  git -C "$BASE_DIR/sglang" fetch --all --tags
  git -C "$BASE_DIR/sglang" checkout "$SGLANG_COMMIT"
}

install_sglang() {
  cd "$BASE_DIR/sglang"
  # prost-build looks up $PROTOC before falling back to PATH.
  PROTOC="$PROTOC_BIN" PATH="$BASE_DIR/bin:$PATH" \
    pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
}

# SGLang's dependency set upgrades cuda-python past 13 and defaults torch and the
# native kernels to cu13 wheels; slime's stack is cu129 throughout. These three
# steps put that back, exactly as build_conda.sh does.
fix_cuda_python() {
  pip install --force-reinstall "cuda-python==12.9"
}

fix_torch() {
  pip install --force-reinstall --no-deps \
    torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129
}

install_flashinfer_jit_cache() {
  # Install pinned FlashInfer kernels to avoid first-use JIT on pre-Hopper GPUs.
  pip install "flashinfer-jit-cache==0.6.12+cu129" \
    --index-url https://flashinfer.ai/whl/cu129/ --no-deps
}

fix_sgl_kernels() {
  pip install --force-reinstall --no-deps \
    sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
    --index-url https://docs.sglang.ai/whl/cu129/
}

fix_nvidia_libs() {
  # pip uninstall stomps the site-packages/nvidia/* shared dirs that cu12 and
  # cu13 co-own, so the cu13 packages come out and the cu12 equivalents go back
  # in to repair them.
  pip uninstall -y \
    nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
    nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
    nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
    nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \
    nvidia-cutlass-dsl-libs-cu13 || true
  pip install --force-reinstall --no-deps \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12==9.16.0.29 nvidia-cufft-cu12 \
    nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
    --index-url https://download.pytorch.org/whl/cu129 \
    --extra-index-url https://pypi.org/simple
}

install_flash_attn() {
  pip install cmake ninja
  # SGLang ships FA4; the validated TE 2.16 context-parallel stack wants FA2
  # 2.7.4-2.8.3. Upstream publishes no py312/torch2.11/cu12/cxx11abi build, so
  # build_conda.sh pins this community wheel by digest -- kept verbatim.
  pip uninstall -y flash-attn-4 flash_attn_4 || true
  pip install --no-deps \
    "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697"
  pip install flash-linear-attention==0.4.2
  # Optional kernel dependency must be an explicit immutable wheel, not nightly.
  if [[ -n "${TILELANG_WHEEL_URL:-}" ]]; then
    [[ "$TILELANG_WHEEL_URL" == *'#sha256='* ]] || { echo 'TILELANG_WHEEL_URL requires a sha256 fragment' >&2; exit 2; }
    pip install "$TILELANG_WHEEL_URL"
  fi
  if [ "$INSTALL_FLASHQLA" = "1" ]; then
    pip install "git+https://github.com/QwenLM/FlashQLA.git@821fd9d37ede18fdc2a4e707fefe3770bfc32e58" --no-build-isolation
  fi
}

install_te() {
  # Resolves to the prebuilt transformer_engine_cu12 binary wheel rather than a
  # source build (confirmed against the reference image, which carries
  # transformer_engine_cu12 2.16.1).
  pip install --no-build-isolation "transformer_engine[pytorch]==2.16.1"
}

install_apex() {
  [ "$INSTALL_APEX" = "1" ] || { echo ">>> apex skipped (INSTALL_APEX=1 to build it)"; return 0; }
  pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel ${MAX_JOBS}" \
    "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"
}

install_tms() {
  # Build torch_memory_saver against installed CUDA/Torch for colocated rollouts;
  # build isolation otherwise omits the required CUDA extension.
  TMS_CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" \
    pip install -v "git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT}" \
    --no-cache-dir --force-reinstall --no-build-isolation
  pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
  pip install "https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl" --force-reinstall
  python -c "import sglang_router; assert 'slime' in sglang_router.__version__"
}

install_megatron() {
  [ -d "$BASE_DIR/Megatron-LM" ] || \
    git clone --recursive https://github.com/NVIDIA/Megatron-LM.git "$BASE_DIR/Megatron-LM"
  git -C "$BASE_DIR/Megatron-LM" fetch --all --tags
  git -C "$BASE_DIR/Megatron-LM" checkout "$MEGATRON_COMMIT"
  # --no-build-isolation: setup.py builds megatron.core.datasets.helpers_cpp by
  # shelling out to `python3 -m pybind11`; under isolation pip hides the env's
  # pybind11 and the extension is silently skipped, which breaks dataset loading.
  pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
  cd "$BASE_DIR/Megatron-LM" && pip install -e . --no-build-isolation
}

install_slime() {
  cd "$SLIME_DIR"
  # requirements.txt first (pure-python runtime deps), then slime itself with
  # --no-deps so pip cannot re-resolve and stomp the pinned native libs above.
  pip install -r requirements.txt
  pip install -e . --no-deps
  if [ "$INSTALL_INT4QAT" = "1" ]; then
    cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat" && pip install . --no-build-isolation
  fi
}

pin_runtime() {
  # https://github.com/pytorch/pytorch/issues/168167
  pip install nvidia-cudnn-cu12==9.16.0.29
  pip install "numpy==1.26.4" "scipy==1.17.1"
  # kernels 0.15.x raises ValueError("Either a revision or a version must be
  # specified") from transformers.integrations.hub_kernels on import, which
  # breaks `import sglang`.
  pip install "kernels<0.15.0"
}

apply_patches() {
  local patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
  [ -d "$patch_dir" ] || { echo "!!! no patch dir $patch_dir" >&2; exit 1; }

  # `patch_name` and `p` are local because step() -- this function's caller --
  # holds its step name in a local called `name`, and bash's dynamic scoping lets
  # an undeclared assignment in a callee overwrite it.
  local patch_name p
  cd "$BASE_DIR/sglang"
  for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch \
                    sglang-pull_weights.patch sglang-deterministic.patch; do
    p="$patch_dir/$patch_name"
    [ -f "$p" ] || continue
    if git apply --check "$p" 2>/dev/null; then
      git apply "$p"; echo "    applied $patch_name"
    elif git apply --reverse --check "$p" 2>/dev/null; then
      echo "    $patch_name already applied"
    else
      echo "!!! $patch_name does not apply cleanly" >&2; exit 1
    fi
  done

  cd "$BASE_DIR/Megatron-LM"
  for patch_name in megatron.patch megatron-sglang-aligned.patch; do
    p="$patch_dir/$patch_name"
    if [ ! -f "$p" ]; then
      [ "$patch_name" = "megatron.patch" ] && { echo "!!! missing $p" >&2; exit 1; }
      continue
    fi
    if git apply --reverse --check "$p" 2>/dev/null; then
      echo "    $patch_name already applied"
    else
      git update-index --refresh || true
      git apply "$p" --3way || { echo "!!! $patch_name does not apply cleanly" >&2; exit 1; }
      git grep -n '^<<<<<<< ' -- . && { echo "!!! $patch_name left conflicts" >&2; exit 1; }
      echo "    applied $patch_name"
    fi
  done
}

smoke_test() {
  cd "$BASE_DIR"
  python - <<'PYEOF'
import glob, importlib, os, sys, sysconfig
from importlib.metadata import PackageNotFoundError, version as dist_version

# __version__ is not universal here (sglang, slime and torch_memory_saver expose
# none), so compare against installed distribution metadata instead.
DIST = {"megatron.core": "megatron-core", "flash_attn": "flash-attn"}


def ver(mod):
    try:
        return dist_version(DIST.get(mod, mod))
    except PackageNotFoundError:
        return getattr(sys.modules.get(mod), "__version__", "(unknown)")


print("python          ", sys.version.split()[0])
print("executable      ", sys.executable)
ok = True
for mod, want in [
    ("torch", "2.11.0+cu129"),
    ("sglang", "0.5.15.post1"),
    ("megatron.core", "0.16.0rc0"),
    ("slime", None),
    ("transformer_engine", "2.16.1"),
    ("flash_attn", None),   # dist version carries the +cu12torch2.11 build tag
    ("transformers", None),
    ("ray", None),
    ("torch_memory_saver", None),
    ("sglang_router", None),
]:
    try:
        importlib.import_module(mod)
        got = ver(mod)
        flag = ""
        if want and got != want:
            flag = "   !! expected " + want
            ok = False
        print("%-20s %s%s" % (mod, got, flag))
    except Exception as e:
        print("%-20s FAILED: %r" % (mod, e))
        ok = False

import torch
print("torch.version.cuda", torch.version.cuda)

# flash-attn's compiled half: an ABI mismatch against torch surfaces here, not
# at install time.
try:
    import flash_attn_2_cuda  # noqa: F401
    print("flash_attn cuda ext ok")
except Exception as e:
    print("flash_attn cuda ext FAILED: %r" % (e,))
    ok = False

# --colocate depends on torch_memory_saver's native hook; a python-only build
# installs fine and only fails later, inside SGLang.
sp = sysconfig.get_paths()["purelib"]
hooks = [os.path.basename(h)
         for h in glob.glob(os.path.join(sp, "torch_memory_saver_hook_mode_*.so"))]
print("tms native hooks   ", hooks or "MISSING")
if not hooks:
    ok = False

sys.exit(0 if ok else 1)
PYEOF
}

check_rm() {
  [ -f "$CODING_DIR/rl/impossible_lcb_rm.py" ] || { echo "Reward module is not present in this stack layer; skipping reward import"; return 0; }
  PYTHONPATH="$CODING_DIR" python -c \
    'import rl.impossible_lcb_rm as m; print("custom_rm resolved:", m.custom_rm)'
}

# ---------------------------------------------------------------------------
if [ "$CHECK_ONLY" -eq 1 ]; then
  set +u; conda activate "$ENV_PREFIX"; set -u
  smoke_test && check_rm
  exit $?
fi

echo ">>> env      $ENV_PREFIX"
echo ">>> deps     $BASE_DIR"
echo ">>> slime    $SLIME_DIR"
echo ">>> jobs     $MAX_JOBS   arch $TORCH_CUDA_ARCH_LIST"

step 00_create_env      create_env
step 00b_micromamba     fetch_micromamba
# conda's activate.d hooks are third-party shell code; several read variables
# unguarded, so -u is lifted just across activation.
set +u; conda activate "$ENV_PREFIX"; set -u
export CUDA_HOME="$CONDA_PREFIX"
step 01_cuda            install_cuda
step 02_clone_sglang    clone_sglang
step 02b_protoc         fetch_protoc
step 03_install_sglang  install_sglang
step 04_cuda_python     fix_cuda_python
step 05_torch           fix_torch
step 06_sgl_kernels     fix_sgl_kernels
step 06b_fi_jit_cache   install_flashinfer_jit_cache
step 07_nvidia_libs     fix_nvidia_libs
step 08_flash_attn      install_flash_attn
step 09_te              install_te
step 10_apex            install_apex
step 11_tms             install_tms
step 12_megatron        install_megatron
step 13_slime           install_slime
step 14_pin_runtime     pin_runtime
step 15_patches         apply_patches

echo
echo ">>> smoke test"
smoke_test
check_rm
echo
echo ">>> done. activate with:  conda activate $ENV_PREFIX"
