# vLLM Windows Build Troubleshooting

## Error CMake común: returned non-zero exit status 1

### Problema
```
subprocess.CalledProcessError: Command ['cmake', ...] returned non-zero exit status 1
```

### Soluciones

#### 1. Verificar configuración de Visual Studio
Asegúrate de ejecutar el entorno de Visual Studio **antes** de iniciar la build:
```cmd
"c:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
```

#### 2. Variables de entorno necesarias
```cmd
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=10
```

#### 3. Limpiar build anterior
```cmd
rmdir /s /q build
rmdir /s /q .deps
```

#### 4. Verificar versiones de CUDA
**Problema común**: Error muestra CUDA v13.0, pero torch 2.7.1+cu126 espera CUDA 12.6

**Solución inmediata**: Usar wheel pre-construida (paso 5)

**Solución a largo plazo**:
- Instalar CUDA Toolkit 12.6 para match con torch
- O cambiar a torch version compatible con CUDA 13.0
- Verificar con: `python -c "import torch; print(torch.version.cuda)"`

#### 5. Uso de wheel pre-construido (recomendado) ✅
**IMPORTANTE**: Activar venv primero
```cmd
.\.venv\Scripts\activate.ps1
uv pip install Z_IOSU/vllm-0.10.2+cu124-cp312-cp312-win_amd64.whl --force-reinstall
uv pip install torch==2.7.1+cu126 torchaudio==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```
**Nota**: Este workflow instala exitosamente vLLM 0.10.2+cu124 con todas las dependencias compatibles.

#### 6. Build desde source con uv
```cmd
uv pip install torch==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements/build.txt
uv pip install -r requirements/windows.txt
uv pip install . --no-build-isolation
```

### Pasos de troubleshooting recomendados:

1. **ACTIVAR VENV** (CRÍTICO): `.\venv\Scripts\activate.ps1`
2. **Usar wheel pre-construido PRIMERO** (✅ EXITOSO)
3. **🔥 RENOMBRAR CARPETA vllm LOCAL** (CRÍTICO): `ren vllm vllm_source_temp`
4. **Limpiar entorno** (solo si build desde source)
5. **Configurar Visual Studio** (solo si build desde source)
6. **Verificar CUDA compatibility** (solo si build desde source)

**IMPORTANTE**: El paso 3 es OBLIGATORIO para evitar conflictos de importación.

#### 7. Error: ModuleNotFoundError: No module named 'vllm._C' 🔥
**Problema crítico**: Python importa el código fuente local en lugar de la wheel instalada
**Síntoma**: `vllm serve` falla con error de módulo _C faltante

**Solución OBLIGATORIA**:
```cmd
ren vllm vllm_source_temp
python -c "import vllm; print('Path correcto:', vllm.__file__)"
vllm --help
```

#### 8. Error: RuntimeError: Unknown runtime environment
**Problema**: Al intentar `uv pip install . --no-build-isolation` después de instalar wheel
**Solución**: Este error es normal si ya tienes vLLM instalado de la wheel. **NO necesitas compilar desde source.**

### Comandos de diagnóstico
```cmd
cmake --version
ninja --version
nvcc --version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### ✅ Verificar instalación exitosa
```cmd
python -c "import vllm; print('✅ vLLM instalado correctamente:', vllm.__version__)"
vllm --help
vllm serve "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" --help
```

### 🎉 INSTALACIÓN Y EJECUCIÓN EXITOSA
La wheel pre-construida funciona perfectamente:
- ✅ vLLM 0.10.2 instalado y funcionando
- ✅ CLI funcional con todos los comandos  
- ✅ Servidor API iniciado exitosamente
- ✅ Modelo Qwen2-VL-OCR-2B-Instruct cargando
- ✅ CUDA detectado automáticamente
- ✅ Windows compatibility confirmado

**Comando exitoso final**:
```cmd
.\.venv\Scripts\activate.ps1
ren vllm vllm_source_temp
python -m vllm.entrypoints.cli.main serve "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" --max-model-len 2048
```