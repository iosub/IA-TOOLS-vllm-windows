#!/usr/bin/env python3
"""
Test script para probar vLLM offline con modelo multimodal
usando configuraciones compatibles con Windows
"""

import os
import torch
from vllm import LLM, SamplingParams

# Configurar variables de entorno específicas para Windows
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"
# Usar attention backend compatible
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

def test_vllm_text_model():
    """Test vLLM with a text generation model"""
    
    print("🚀 Iniciando vLLM con modelo de texto...")
    
    try:
        # Verificar CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA no disponible, usando CPU")
        
        # Crear instancia LLM con modelo de texto bien soportado
        print("📋 Configurando modelo Llama2...")
        llm = LLM(
            model="meta-llama/Llama-2-7b-chat-hf",  # Modelo de texto bien soportado
            max_model_len=1024,  
            enforce_eager=True,  # Deshabilitar CUDA graphs
            gpu_memory_utilization=0.7,
            disable_custom_all_reduce=True,
            disable_log_stats=True,
            trust_remote_code=True,
        )
        
        print("✅ Modelo cargado exitosamente!")
        
        # Configurar sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )
        
        # Test prompts en español
        prompts = [
            "¿Cuál es la capital de España?",
            "Explica qué es la inteligencia artificial en una frase.",
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
    test_vllm_text_model()