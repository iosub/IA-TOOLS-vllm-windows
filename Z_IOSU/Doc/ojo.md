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

# PASO 4: CRUCIAL - Evitar conflicto de importación
# Renombrar carpeta vllm local para que Python use la wheel instalada
ren vllm vllm_source_temp

# PASO 5: Verificar instalación
python -c "import vllm; print('✅ vLLM:', vllm.__version__)"

# PASO 6: Ejecutar vLLM (opciones alternativas si 'vllm' no funciona)
# Opción A: Comando directo
vllm --help

# Opción B: Si 'vllm' no se encuentra
python -m vllm.entrypoints.cli.main --help

# Opción C: Servir modelo
vllm serve "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" --max-model-len 2048
# O:
python -m vllm.entrypoints.cli.main serve "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" --max-model-len 2048

# Opción B: Build desde source (solo si necesario)
# uv pip install -r requirements/build.txt
# uv pip install -r requirements/windows.txt
# uv pip install . --no-build-isolation

vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --tokenizer-mode mistral --config-format mistral --load-format mistral \
 --limit-mm-per-prompt '{"image":4}' --max-model-len 16384