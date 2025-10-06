#!/usr/bin/env python3
"""
Test script para probar vLLM con modelos más grandes
Probando progresivamente hasta encontrar el límite óptimo para RTX 3090
"""

import os
import torch
from vllm import LLM, SamplingParams

# Configurar variables de entorno específicas para Windows
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"

def test_larger_models():
    """Test vLLM with progressively larger models"""
    
    print("🚀 Probando vLLM con modelos más grandes...")
    print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Lista de modelos de diferentes tamaños (ordenados de menor a mayor)
    models_to_try = [
        {
            "name": "microsoft/DialoGPT-large", 
            "description": "DialoGPT Large (~1.5GB)",
            "max_len": 1024,
            "memory_util": 0.7
        },
        {
            "name": "EleutherAI/gpt-neo-1.3B",
            "description": "GPT-Neo 1.3B (~5GB)",
            "max_len": 2048,
            "memory_util": 0.75
        },
        {
            "name": "EleutherAI/gpt-neo-2.7B",
            "description": "GPT-Neo 2.7B (~10GB)",
            "max_len": 2048,
            "memory_util": 0.8
        },
        {
            "name": "microsoft/DialoGPT-large",
            "description": "DialoGPT Large con más contexto",
            "max_len": 2048,
            "memory_util": 0.75
        }
    ]
    
    successful_models = []
    
    for model_config in models_to_try:
        model_name = model_config["name"]
        description = model_config["description"]
        max_len = model_config["max_len"]
        memory_util = model_config["memory_util"]
        
        print(f"\n{'='*60}")
        print(f"📋 Probando: {description}")
        print(f"🏷️  Modelo: {model_name}")
        print(f"📏 Max length: {max_len}")
        print(f"💾 GPU utilization: {memory_util}")
        print("="*60)
        
        try:
            # Limpiar memoria GPU antes de cada modelo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            llm = LLM(
                model=model_name,
                max_model_len=max_len,
                enforce_eager=True,
                gpu_memory_utilization=memory_util,
                disable_custom_all_reduce=True,
                disable_log_stats=True,
                trust_remote_code=True,
            )
            
            print("✅ Modelo cargado exitosamente!")
            
            # Configurar sampling parameters
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.9,
                max_tokens=150  # Más tokens para modelos grandes
            )
            
            # Test prompts más elaborados para modelos grandes
            prompts = [
                "Write a short story about artificial intelligence:",
                "Explain the concept of machine learning in simple terms:",
                "What are the benefits and risks of AI technology?",
            ]
            
            print("💬 Generando respuestas...")
            outputs = llm.generate(prompts, sampling_params)
            
            # Mostrar resultados
            print("\n🎯 RESULTADOS:")
            for i, output in enumerate(outputs, 1):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"\n📝 Prompt {i}: {prompt}")
                print(f"🤖 Respuesta: {generated_text}")
                print("-" * 50)
            
            # Estadísticas de memoria
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"\n📊 Memoria GPU utilizada: {memory_allocated:.2f} GB")
                print(f"📊 Memoria GPU reservada: {memory_reserved:.2f} GB")
            
            successful_models.append({
                "name": model_name,
                "description": description,
                "max_len": max_len,
                "memory_gb": memory_allocated if torch.cuda.is_available() else "N/A"
            })
            
            print(f"🎉 Test con {model_name} completado exitosamente!")
            
            # Limpieza explícita
            del llm
            torch.cuda.empty_cache()
            
        except Exception as model_error:
            print(f"❌ Error con {model_name}: {str(model_error)[:200]}...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE MODELOS EXITOSOS:")
    print("="*60)
    
    if successful_models:
        for model in successful_models:
            print(f"✅ {model['description']}")
            print(f"   📄 Modelo: {model['name']}")
            print(f"   📏 Max length: {model['max_len']}")
            print(f"   💾 Memoria: {model['memory_gb']} GB")
            print()
    else:
        print("❌ Ningún modelo grande funcionó correctamente")
        print("💡 Sugerencia: Usar modelos más pequeños o ajustar configuración")
    
    print("🏁 Test de modelos grandes completado!")

if __name__ == "__main__":
    test_larger_models()