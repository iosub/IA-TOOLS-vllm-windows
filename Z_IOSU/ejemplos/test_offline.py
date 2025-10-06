#!/usr/bin/env python3
"""
Test script para probar vLLM offline con Qwen2-VL-OCR-2B-Instruct
usando configuraciones compatibles con Windows
"""

import os
import torch
from vllm import LLM, SamplingParams

# Configurar variables de entorno antes de importar vLLM
os.environ["USE_LIBUV"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["DISABLE_CUSTOM_ALL_REDUCE"] = "true"

def test_vllm_offline():
    """Test vLLM with offline API"""
    
    print("🚀 Iniciando vLLM con API offline...")
    
    try:
        # Crear instancia LLM con configuración Windows-compatible
        print("📋 Configurando modelo...")
        llm = LLM(
            model="prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
            max_model_len=1024,  # Reducido para evitar problemas de memoria
            enforce_eager=True,  # Deshabilitar CUDA graphs
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            disable_custom_all_reduce=True,
        )
        
        print("✅ Modelo cargado exitosamente!")
        
        # Configurar sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )
        
        # Test prompt simple de texto
        prompts = [
            "¿Cuál es la capital de España?",
            "Explica qué es la inteligencia artificial en una frase",
        ]
        
        print("💬 Generando respuestas...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Mostrar resultados
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n📝 Prompt: {prompt}")
            print(f"🤖 Respuesta: {generated_text}")
            print("-" * 50)
            
        print("🎉 Test completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vllm_offline()