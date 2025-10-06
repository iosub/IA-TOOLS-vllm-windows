# PASO 1: 🔥 SIEMPRE activar entorno virtual primero (CADA VEZ)
.\\.venv\\Scripts\\activate.ps1

# ⚠️ IMPORTANTE: El prompt debe mostrar (vllm-windows)
# Si no se ve, reactivar: .\\.venv\\Scripts\\activate.ps1

# PASO 2: Configurar Visual Studio
"c:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
#(replace 10 with your desired cpu threads to use in parallel to speed up compilation)
set MAX_JOBS=10

# PASO 3: Instalación completa con uv (✅ EXITOSO)
# Opción A: Wheel pre-construida (RECOMENDADO)
uv pip install Z_IOSU/vllm-0.10.2+cu124-cp312-cp312-win_amd64.whl --force-reinstall
uv pip install torch==2.7.1+cu126 torchaudio==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126

# PASO 3.5: Instalar Triton compatible con Windows (CRUCIAL)
### 2. Instalar Triton 3.0.0 compatible con Windows (Python 3.12)
```powershell
# URL correcta para Python 3.12 desde madbuda/triton-windows-builds
uv pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp312-cp312-win_amd64.whl
```

**Versiones disponibles de Triton 3.0.0:**
- Python 3.10: `triton-3.0.0-cp310-cp310-win_amd64.whl`
- Python 3.11: `triton-3.0.0-cp311-cp311-win_amd64.whl`  
- Python 3.12: `triton-3.0.0-cp312-cp312-win_amd64.whl`

**Requisitos:** CUDA 12.x (compatible con nuestro CUDA 12.6)

# PASO 4: CRUCIAL - Evitar conflicto de importación
# Renombrar carpeta vllm local para que Python use la wheel instalada
ren vllm vllm_source_temp

# PASO 5: Verificar instalación
python -c "import vllm; print('✅ vLLM:', vllm.__version__)"

# PASO 6: ✅ CONFIGURACIÓN PROBADA QUE FUNCIONA
# Variables de entorno necesarias para Windows
$env:USE_LIBUV="0"
$env:VLLM_ATTENTION_BACKEND="XFORMERS"

# PASO 7: 🎉 MODELOS PROBADOS EXITOSAMENTE
# ✅ GPT-2 (FUNCIONA PERFECTAMENTE) - 456 tokens/s output
python -m vllm.entrypoints.cli.main serve "gpt2" --max-model-len 512 --enforce-eager

# ✅ DialoGPT (RECOMENDADO PARA CHAT)
python -m vllm.entrypoints.cli.main serve "microsoft/DialoGPT-medium" --max-model-len 1024 --enforce-eager

# ❌ MODELOS PROBLEMÁTICOS (requieren kernels Triton)
# python -m vllm.entrypoints.cli.main serve "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" --max-model-len 2048
# ^ FALLA: PermissionError: [WinError 5] Access is denied (Triton compilation)

# PASO 8: 🧪 API OFFLINE (archivo test_gpt2.py)
# Ver Z_IOSU/ejemplos/test_gpt2.py para código completo

# Opción B: Build desde source (solo si necesario)
# uv pip install -r requirements/build.txt
# uv pip install -r requirements/windows.txt
# uv pip install . --no-build-isolation

vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --tokenizer-mode mistral --config-format mistral --load-format mistral \
 --limit-mm-per-prompt '{"image":4}' --max-model-len 16384