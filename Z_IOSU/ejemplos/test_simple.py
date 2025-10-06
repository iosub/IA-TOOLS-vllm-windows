#!/usr/bin/env python3
"""
Test script simple para probar vLLM offline con un modelo básico
usando configuraciones compatibles con Windows
"""

import os
import torch
from vllm import LLM, SamplingParams

# Configurar variables de entorno específicas para Windows
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"

def test_vllm_simple():
    """Test vLLM with a simple, well-supported model"""
    
    print("🚀 Iniciando vLLM con modelo simple...")
    
    try:
        # Verificar CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA no disponible, usando CPU")
        
        # Crear instancia LLM con modelo simple
        print("📋 Configurando modelo GPT-2...")
        llm = LLM(
            model="gpt2",  # Modelo simple y bien soportado
            max_model_len=256,  # Muy pequeño para test
            enforce_eager=True,  # Deshabilitar CUDA graphs
            gpu_memory_utilization=0.5,
            disable_custom_all_reduce=True,
            disable_log_stats=True,
        )
        
        print("✅ Modelo cargado exitosamente!")
        
        # Configurar sampling parameters conservadores
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=50  # Pocos tokens para test rápido
        )
        
        # Test prompt muy simple
        prompts = [
            "Once upon a time",
            "The weather today is",
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
    test_vllm_simple()