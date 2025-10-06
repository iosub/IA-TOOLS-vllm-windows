#!/usr/bin/env python3
"""
Test de compatibilidad de vLLM con modelos GGUF
Probando Qwen2-VL-OCR-2B-Instruct en formato GGUF
"""

import os
from vllm import LLM, SamplingParams

# Configurar variables de entorno para Windows
os.environ["USE_LIBUV"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

def test_gguf_model():
    """Test vLLM con modelo GGUF de Qwen2-VL-OCR"""
    
    print("🧪 Probando vLLM con modelo GGUF...")
    print("📦 Modelo: mradermacher/Qwen2-VL-OCR-2B-Instruct-i1-GGUF")
    print(f"🔧 Backend: {os.environ.get('VLLM_ATTENTION_BACKEND', 'DEFAULT')}")
    
    try:
        # Intentar cargar modelo GGUF
        # Nota: vLLM podría no soportar GGUF directamente
        print("📋 Intentando cargar modelo GGUF...")
        
        # Probamos con diferentes variantes para ver cual funciona
        model_variants = [
            "mradermacher/Qwen2-VL-OCR-2B-Instruct-i1-GGUF",
            # Si no funciona, podríamos intentar especificar un archivo específico
            # "mradermacher/Qwen2-VL-OCR-2B-Instruct-i1-GGUF/Qwen2-VL-OCR-2B-Instruct.i1-Q4_K_M.gguf"
        ]
        
        for model_name in model_variants:
            try:
                print(f"🔄 Probando variante: {model_name}")
                
                llm = LLM(
                    model=model_name,
                    max_model_len=1024,  # Tamaño conservador
                    enforce_eager=True,
                    gpu_memory_utilization=0.7,
                    disable_custom_all_reduce=True,
                    trust_remote_code=True,  # Podría ser necesario
                )
                
                print("✅ ¡Modelo GGUF cargado exitosamente!")
                
                # Test de generación
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=100
                )
                
                prompts = [
                    "Describe what you see in this image: [No image provided, just respond with text]",
                    "What is OCR and how does it work?",
                    "Explain the benefits of using quantized models"
                ]
                
                print("💬 Generando respuestas...")
                outputs = llm.generate(prompts, sampling_params)
                
                for i, output in enumerate(outputs, 1):
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    print(f"\n📝 Test {i}:")
                    print(f"Prompt: {prompt[:50]}...")
                    print(f"Respuesta: {generated_text}")
                    print("-" * 70)
                
                print("🎉 ¡Test GGUF completado exitosamente!")
                return True
                
            except Exception as e:
                print(f"❌ Error con {model_name}: {e}")
                continue
        
        print("❌ Ninguna variante del modelo GGUF funcionó")
        return False
        
    except Exception as e:
        print(f"❌ Error general durante el test GGUF: {e}")
        print("\n🔍 Análisis del error:")
        if "GGUF" in str(e) or "gguf" in str(e):
            print("   • vLLM podría no soportar archivos GGUF directamente")
            print("   • Los modelos GGUF son típicamente para llama.cpp/Ollama")
        elif "Triton" in str(e):
            print("   • Mismo problema de kernels Triton que el modelo original")
        else:
            print(f"   • Error inesperado: {e}")
        
        import traceback
        traceback.print_exc()
        return False

def suggest_alternatives():
    """Sugerir alternativas si GGUF no funciona"""
    print("\n💡 ALTERNATIVAS RECOMENDADAS:")
    print("1. 🔄 Usar el modelo original HF Transformers (si solucionamos Triton)")
    print("2. 🔧 Probar con llama.cpp + GGUF (fuera de vLLM)")
    print("3. 🐳 Usar Ollama que soporta GGUF nativamente")
    print("4. ✅ Continuar con GPT-2 que sabemos que funciona")

if __name__ == "__main__":
    success = test_gguf_model()
    if not success:
        suggest_alternatives()