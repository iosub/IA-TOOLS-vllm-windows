#!/usr/bin/env python3
"""
Test funcional de vLLM en Windows con un modelo compatible
que NO requiere kernels Triton customizados
"""

import os
from vllm import LLM, SamplingParams

# Configurar variables de entorno para Windows
os.environ["USE_LIBUV"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

def test_compatible_model():
    """Test con modelo GPT-2 que es compatible con Windows"""
    
    print("🚀 Iniciando vLLM con modelo GPT-2 (compatible con Windows)...")
    
    try:
        # Usar GPT-2 que NO requiere kernels Triton customizados
        print("📋 Configurando modelo GPT-2...")
        llm = LLM(
            model="gpt2",
            max_model_len=512,  # Tamaño pequeño para test
            enforce_eager=True,  # Deshabilitar CUDA graphs  
            gpu_memory_utilization=0.7,
            disable_custom_all_reduce=True,
        )
        
        print("✅ Modelo GPT-2 cargado exitosamente!")
        
        # Configurar sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50
        )
        
        # Test prompts simples
        prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "Once upon a time in a digital realm,",
        ]
        
        print("💬 Generando respuestas...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Mostrar resultados
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n📝 Prompt: {prompt}")
            print(f"🤖 Respuesta: {generated_text}")
            print("-" * 80)
            
        print("🎉 Test completado exitosamente! vLLM funciona en Windows con GPT-2")
        
    except Exception as e:
        print(f"❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_compatible_model()