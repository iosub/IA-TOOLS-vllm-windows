#!/usr/bin/env python3
"""
Test de vLLM con modelo GGUF local de Qwen2-VL-OCR
Usando archivo descargado localmente
"""

import os
from pathlib import Path
from vllm import LLM, SamplingParams

# Configurar variables de entorno para Windows
os.environ["USE_LIBUV"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

def test_local_gguf():
    """Test vLLM con archivo GGUF local"""
    
    print("🧪 Probando vLLM con archivo GGUF local...")
    
    # Rutas a los archivos GGUF descargados
    gguf_path = Path("Z_IOSU/modelos/qwen2vl-ocr-q2k.gguf")  # Archivo actual disponible
    mmproj_path = Path("Z_IOSU/modelos/qwen2vl-ocr-mmproj.gguf")  # Archivo de proyección multimodal
    
    if not gguf_path.exists():
        print(f"❌ Archivo GGUF principal no encontrado: {gguf_path}")
        print("💡 Asegúrate de que la descarga se completó correctamente")
        return False
    
    # El archivo mmproj es opcional para pruebas de solo texto
    if mmproj_path.exists():
        print(f"✅ Archivo mmproj encontrado: {mmproj_path}")
    else:
        print(f"⚠️  Archivo mmproj no encontrado: {mmproj_path}")
        print("   Continuaremos con solo capacidades de texto")
    
    print(f"📁 Archivo GGUF encontrado: {gguf_path}")
    print(f"📊 Tamaño del archivo: {gguf_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        print("📋 Cargando modelo GGUF local...")
        
        llm = LLM(
            model=str(gguf_path.absolute()),  # Ruta absoluta al archivo GGUF
            max_model_len=1024,
            enforce_eager=True,
            gpu_memory_utilization=0.7,
            disable_custom_all_reduce=True,
        )
        
        print("✅ ¡Modelo GGUF local cargado exitosamente!")
        
        # Test de generación
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )
        
        # Prompts de prueba (sin imágenes por ahora)
        prompts = [
            "What is OCR technology?",
            "Explain the advantages of quantized models",
            "How does vision-language model work?"
        ]
        
        print("💬 Generando respuestas...")
        outputs = llm.generate(prompts, sampling_params)
        
        for i, output in enumerate(outputs, 1):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n📝 Test {i}:")
            print(f"Prompt: {prompt}")
            print(f"Respuesta: {generated_text}")
            print("-" * 70)
        
        print("🎉 ¡Test GGUF local completado exitosamente!")
        print("✅ vLLM puede usar archivos GGUF locales en Windows!")
        return True
        
    except Exception as e:
        print(f"❌ Error durante el test GGUF local: {e}")
        
        # Análisis del error
        error_str = str(e)
        if "Triton" in error_str:
            print("⚠️  Mismo problema de kernels Triton que el modelo original")
            print("   El formato GGUF no resuelve el problema de kernels específicos")
        elif "GGUF" in error_str:
            print("⚠️  Problema específico con el formato GGUF")
        elif "config" in error_str.lower():
            print("⚠️  Falta archivo de configuración")
            print("   Los modelos GGUF podrían necesitar archivos adicionales")
        else:
            print(f"⚠️  Error inesperado: {error_str}")
        
        import traceback
        traceback.print_exc()
        return False

def check_file_status():
    """Verificar estado del archivo descargado"""
    gguf_path = Path("Z_IOSU/modelos/qwen2vl-ocr-q2k.gguf")
    if gguf_path.exists():
        size_mb = gguf_path.stat().st_size / (1024*1024)
        print(f"📁 Estado del archivo: {size_mb:.1f} MB")
        if size_mb < 600:  # El archivo Q2_K debería ser ~676MB
            print("⚠️  La descarga podría estar incompleta")
            return False
        return True
    else:
        print("❌ Archivo GGUF no encontrado")
        return False

if __name__ == "__main__":
    if check_file_status():
        test_local_gguf()
    else:
        print("💡 Espera a que termine la descarga y vuelve a ejecutar")