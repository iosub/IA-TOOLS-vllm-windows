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
uv pip install torch==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install Z_IOSU/vllm-0.10.2+cu124-cp312-cp312-win_amd64.whl --force-reinstall
```
**Nota**: Este comando instala exitosamente vLLM 0.10.2+cu124 evitando problemas de compilación.

#### 6. Build desde source con uv
```cmd
uv pip install torch==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements/build.txt
uv pip install -r requirements/windows.txt
uv pip install . --no-build-isolation
```

### Pasos de troubleshooting recomendados:

1. **ACTIVAR VENV** (CRÍTICO): `.\.venv\Scripts\activate.ps1`
2. **Limpiar entorno**
3. **Configurar Visual Studio**
4. **Verificar CUDA compatibility**
5. **Usar wheel pre-construido si build falla**

### Comandos de diagnóstico
```cmd
cmake --version
ninja --version
nvcc --version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### Verificar instalación exitosa
```cmd
python -c "import vllm; print('vLLM instalado correctamente:', vllm.__version__)"
vllm --help
```