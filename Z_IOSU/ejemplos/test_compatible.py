#!/usr/bin/env python3
"""
Test script para probar vLLM offline con modelos pequeños y compatibles
usando configuraciones compatibles con Windows
"""

import os
import torch
from vllm import LLM, SamplingParams

# Configurar variables de entorno específicas para Windows
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"

def test_with_small_model():
    """Test vLLM with small compatible models"""
    
    print("🚀 Iniciando vLLM con modelo pequeño compatible...")
    
    try:
        # Verificar CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"💾 Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA no disponible, usando CPU")
        
        # Probar con diferentes modelos hasta encontrar uno que funcione
        models_to_try = [
            ("microsoft/DialoGPT-small", "Modelo de conversación pequeño"),
            ("distilgpt2", "GPT-2 destilado (más pequeño)"),
            ("gpt2", "GPT-2 estándar"),
        ]
        
        for model_name, description in models_to_try:
            print(f"\n📋 Probando {description}: {model_name}")
            
            try:
                llm = LLM(
                    model=model_name,
                    max_model_len=512,  
                    enforce_eager=True,  
                    gpu_memory_utilization=0.6,
                    disable_custom_all_reduce=True,
                    disable_log_stats=True,
                    trust_remote_code=False,  # Más seguro
                )
                
                print("✅ Modelo cargado exitosamente!")
                
                # Configurar sampling parameters
                sampling_params = SamplingParams(
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=80
                )
                
                # Test prompts simples
                prompts = [
                    "The capital of France is",
                    "Artificial intelligence is",
                ]
                
                print("💬 Generando respuestas...")
                outputs = llm.generate(prompts, sampling_params)
                
                # Mostrar resultados
                for output in outputs:
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    print(f"\n📝 Prompt: {prompt}")
                    print(f"🤖 Respuesta: {generated_text}")
                    print("-" * 30)
                
                print(f"🎉 Test con {model_name} completado exitosamente!")
                break  # Si funciona, salir del loop
                
            except Exception as model_error:
                print(f"❌ Error con {model_name}: {model_error}")
                continue
                
        else:
            print("❌ Ningún modelo funcionó correctamente")
            
    except Exception as e:
        print(f"❌ Error general durante el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_small_model()