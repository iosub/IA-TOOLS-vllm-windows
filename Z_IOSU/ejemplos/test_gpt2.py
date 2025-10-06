#!/usr/bin/env python3
"""
Test funcional de vLLM Windows con GPT-2
Modelo compatible que no requiere kernels Triton customizados
"""

import os
import torch
from vllm import LLM, SamplingParams

# Configurar variables de entorno para Windows
os.environ["USE_LIBUV"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

def test_gpt2():
    """Test vLLM con GPT-2 que debería funcionar en Windows"""
    
    print("🚀 Probando vLLM Windows con GPT-2...")
    print(f"🔧 Backend de atención: {os.environ.get('VLLM_ATTENTION_BACKEND', 'DEFAULT')}")
    print(f"🔧 USE_LIBUV: {os.environ.get('USE_LIBUV', 'DEFAULT')}")
    
    try:
        # Crear instancia LLM con GPT-2 (modelo simple y compatible)
        print("📋 Cargando GPT-2...")
        llm = LLM(
            model="gpt2",  # Modelo simple sin kernels Triton customizados
            max_model_len=512,  # Longitud corta para prueba
            enforce_eager=True,  # Deshabilitar CUDA graphs
            gpu_memory_utilization=0.7,
            disable_custom_all_reduce=True,
        )
        
        print("✅ GPT-2 cargado exitosamente!")
        
        # Configurar sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=50  # Respuestas cortas para prueba
        )
        
        # Test prompts
        prompts = [
            "The future of artificial intelligence is",
            "Windows operating system is known for",
            "Python programming language allows developers to"
        ]
        
        print("💬 Generando respuestas...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Mostrar resultados
        for i, output in enumerate(outputs, 1):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n📝 Test {i}:")
            print(f"Prompt: {prompt}")
            print(f"Respuesta: {generated_text}")
            print("-" * 70)
            
        print("🎉 ¡Test de GPT-2 completado exitosamente!")
        print("✅ vLLM Windows está funcionando correctamente")
        
    except Exception as e:
        print(f"❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpt2()