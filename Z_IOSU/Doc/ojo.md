# PASO 1: SIEMPRE activar entorno virtual primero
.\.venv\Scripts\activate.ps1

# PASO 2: Configurar Visual Studio
"c:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
#(replace 10 with your desired cpu threads to use in parallel to speed up compilation)
set MAX_JOBS=10

# PASO 3: Use uv for all package management
uv pip install torch==2.7.1+cu126 torchaudio==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements/build.txt
uv pip install -r requirements/windows.txt

# Install from wheel (recommended)
uv pip install Z_IOSU/vllm-0.10.2+cu124-cp312-cp312-win_amd64.whl

# Or build from source
uv pip install . --no-build-isolation

vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --tokenizer-mode mistral --config-format mistral --load-format mistral \
 --limit-mm-per-prompt '{"image":4}' --max-model-len 16384