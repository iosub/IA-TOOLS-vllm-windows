# vLLM Windows AI Coding Agent Instructions

## Project Overview
vLLM is a high-throughput, memory-efficient LLM inference engine with **PagedAttention** for optimal memory management. This Windows-specific fork adds MSVC build support and Windows-compatible kernels.

## Core Architecture Components

### Engine System (`vllm/engine/`)
- **LLMEngine** (`llm_engine.py`): Main orchestrator managing scheduler, executor, and tokenizer
- **V1 Engine** (`vllm/v1/`): New architecture with 1.7x speedup, zero-overhead prefix caching
- Key workflow: Request → Scheduler → Executor → Model Runner → Output Processor

### Scheduling (`vllm/core/scheduler.py`)
- **Preemption modes**: `SWAP` (swap to CPU) vs `RECOMPUTE` (discard blocks)
- **SchedulingBudget**: Controls token/sequence limits per iteration
- Continuous batching with priority-based preemption

### Attention System (`vllm/attention/`)
- **PagedAttention**: Memory blocks allocated on-demand, enables dynamic batching
- Multiple backends: FlashAttention, FlashInfer, custom CUDA kernels
- Backend selection via `AttentionBackend` abstraction

### Model Execution (`vllm/model_executor/`)
- **Parallel strategies**: Tensor, pipeline, data, expert parallelism
- **Quantization**: GPTQ, AWQ, AutoRound, INT4/8, FP8 support
- **Multi-modal**: Vision models (LLaVA), audio processing

### Windows-Specific Patterns

### Build System
```bash
# ALWAYS activate virtual environment first
.\.venv\Scripts\activate.ps1

# Required Visual Studio environment
"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=10

# Use uv for package management
uv pip install torch==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements/build.txt
uv pip install -r requirements/windows.txt
uv pip install . --no-build-isolation
```

### Platform Detection (`vllm/entrypoints/cli/serve.py`)
```python
if platform.system() == "Windows":
    import winloop as uvloop_impl
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # No fork support
    os.environ["USE_LIBUV"] = os.environ.get("USE_LIBUV", "0")  # Disabled by default
```

## Key Development Workflows

### Running Models
```bash
# ALWAYS activate virtual environment first
.\.venv\Scripts\activate.ps1

# Install dependencies with uv first
uv pip install vllm

# Serve a model (CLI entry point)
vllm serve "model-name" --tensor-parallel-size 2 --max-model-len 4096

# Offline inference (Python)
from vllm import LLM, SamplingParams
llm = LLM(model="model-name")
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

### Testing Patterns
- Use `@pytest.mark.skip_v1` for V0-only tests
- Multimodal tests require `ImageAsset`, `AudioAsset`, `VideoAsset` from `vllm.assets`
- Windows process spawning affects distributed tests

### CUDA Kernel Development (`csrc/`)
- Custom kernels in `.cu` files, bound via `torch_bindings.cpp`
- Windows MSVC compatibility requires specific template handling
- Use `dispatch_utils.h` for type dispatching

## Configuration Patterns

### Engine Configuration
```python
# Multi-modal setup
limit_mm_per_prompt={"image": 4}
mm_processor_cache_gb=4  # Cache preprocessed inputs

# Memory optimization
max_model_len=16384
tensor_parallel_size=2
```

### Environment Variables
- `VLLM_TARGET_DEVICE`: cuda/cpu/rocm
- `VLLM_WORKER_MULTIPROC_METHOD`: spawn (Windows requirement)
- `CUDA_ROOT`/`CUDA_HOME`: CUDA installation path
- `USE_CUDSS=1`: Enable cuDSS kernels (if available)

## Critical File Patterns

### CLI Structure (`vllm/entrypoints/cli/`)
- `main.py`: Route to subcommands (serve, benchmark, collect_env)
- `serve.py`: Production server setup
- Each subcommand implements `CLISubcommand` interface

### Plugin System (`vllm/plugins/`)
- LoRA resolvers in `lora_resolvers/`  
- Entry points defined in `pyproject.toml` under `[project.entry-points]`

### Version Control
- Use `setuptools-scm` for version from git tags
- Windows builds require specific torch+cu126 wheels from PyTorch index

## Local Configuration (Z_IOSU/)

### Build Documentation (`Z_IOSU/Doc/ojo.md`)
Local build configuration for Visual Studio 2022 Professional:
```bash
# ALWAYS activate venv first
.\.venv\Scripts\activate.ps1

"c:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=10

uv pip install torch==2.7.1+cu126 torchaudio==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

### Troubleshooting (`Z_IOSU/Doc/troubleshooting.md`)
Common build issues and solutions, including CMake errors and CUDA version mismatches

### Example Models (`Z_IOSU/ejemplos/`)
- **Vision Models**: `prithivMLmods/Qwen2-VL-OCR-2B-Instruct` for OCR tasks
- **Multimodal**: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` with image support

### Pre-built Wheels (`Z_IOSU/`)
- Local wheel: `vllm-0.10.2+cu124-cp312-cp312-win_amd64.whl`
- Install with: `uv pip install Z_IOSU/vllm-0.10.2+cu124-cp312-cp312-win_amd64.whl`

When working with this codebase, prioritize understanding the scheduler-executor-model_runner pipeline and Windows-specific multiprocessing constraints.