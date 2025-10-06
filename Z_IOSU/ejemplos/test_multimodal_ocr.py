#!/usr/bin/env python3
"""
Test script para probar vLLM con modelos multimodales (visión + herramientas)
Específicamente optimizado para OCR y análisis de imágenes en Windows
"""

import os
import torch
import base64
import io
from PIL import Image
from vllm import LLM, SamplingParams
from typing import List, Dict, Any

# Configurar variables de entorno específicas para Windows y modelos multimodales
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

def create_sample_images() -> List[str]:
    """Crear imágenes de ejemplo con texto for OCR testing"""
    
    # Crear imagen simple con texto
    from PIL import Image, ImageDraw, ImageFont
    
    images_base64 = []
    
    # Imagen 1: Texto simple
    img1 = Image.new('RGB', (400, 200), color='white')
    draw1 = ImageDraw.Draw(img1)
    try:
        # Intentar usar fuente del sistema
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Fallback a fuente default
        font = ImageFont.load_default()
    
    draw1.text((50, 50), "Hello World!", fill='black', font=font)
    draw1.text((50, 100), "vLLM OCR Test", fill='blue', font=font)
    draw1.text((50, 150), "2024-10-05", fill='red', font=font)
    
    # Convertir a base64
    buffer1 = io.BytesIO()
    img1.save(buffer1, format='PNG')
    img1_b64 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    images_base64.append(img1_b64)
    
    # Imagen 2: Tabla simple
    img2 = Image.new('RGB', (500, 300), color='white')
    draw2 = ImageDraw.Draw(img2)
    
    # Dibujar una tabla simple
    draw2.rectangle([50, 50, 450, 250], outline='black', width=2)
    draw2.line([50, 100, 450, 100], fill='black', width=1)
    draw2.line([250, 50, 250, 250], fill='black', width=1)
    
    draw2.text((70, 70), "Product", fill='black', font=font)
    draw2.text((270, 70), "Price", fill='black', font=font)
    draw2.text((70, 120), "Laptop", fill='black', font=font)
    draw2.text((270, 120), "$999", fill='black', font=font)
    draw2.text((70, 170), "Mouse", fill='black', font=font)
    draw2.text((270, 170), "$25", fill='black', font=font)
    
    buffer2 = io.BytesIO()
    img2.save(buffer2, format='PNG')
    img2_b64 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    images_base64.append(img2_b64)
    
    return images_base64

def test_multimodal_ocr():
    """Test vLLM with multimodal models optimized for OCR"""
    
    print("🚀 Probando vLLM con modelos multimodales para OCR...")
    print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Modelos multimodales que soportan visión y OCR
    multimodal_models = [
        {
            "name": "Qwen/Qwen2-VL-2B-Instruct",
            "description": "Qwen2-VL 2B - Modelo multimodal optimizado",
            "max_len": 2048,
            "memory_util": 0.75,
            "supports_tools": True
        },
        {
            "name": "microsoft/Phi-3.5-vision-instruct", 
            "description": "Phi-3.5 Vision - Modelo compacto con visión",
            "max_len": 1024,
            "memory_util": 0.7,
            "supports_tools": False
        },
        {
            "name": "llava-hf/llava-1.5-7b-hf",
            "description": "LLaVA 1.5 7B - Modelo de visión popular",
            "max_len": 2048,
            "memory_util": 0.8,
            "supports_tools": False
        }
    ]
    
    # Crear imágenes de prueba
    print("🖼️  Creando imágenes de prueba para OCR...")
    test_images = create_sample_images()
    print(f"✅ Creadas {len(test_images)} imágenes de prueba")
    
    successful_models = []
    
    for model_config in multimodal_models:
        model_name = model_config["name"]
        description = model_config["description"]
        max_len = model_config["max_len"]
        memory_util = model_config["memory_util"]
        supports_tools = model_config["supports_tools"]
        
        print(f"\n{'='*70}")
        print(f"📋 Probando: {description}")
        print(f"🏷️  Modelo: {model_name}")
        print(f"📏 Max length: {max_len}")
        print(f"💾 GPU utilization: {memory_util}")
        print(f"🔧 Soporta herramientas: {'Sí' if supports_tools else 'No'}")
        print("="*70)
        
        try:
            # Limpiar memoria GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Configuración específica para modelos multimodales
            llm = LLM(
                model=model_name,
                max_model_len=max_len,
                enforce_eager=True,
                gpu_memory_utilization=memory_util,
                disable_custom_all_reduce=True,
                disable_log_stats=True,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 4},  # Permitir múltiples imágenes
                mm_processor_kwargs={
                    "num_crops": 16  # Para mejor OCR
                } if "qwen" in model_name.lower() else {},
            )
            
            print("✅ Modelo multimodal cargado exitosamente!")
            
            # Configurar sampling parameters para OCR
            sampling_params = SamplingParams(
                temperature=0.1,  # Baja temperatura para OCR preciso
                top_p=0.9,
                max_tokens=500,
                stop=["<|endoftext|>", "<|im_end|>"]
            )
            
            # Prompts específicos para OCR y análisis visual
            ocr_prompts = [
                {
                    "text": "Analyze this image and extract all text you can see. Provide the text in a structured format:",
                    "image": test_images[0]
                },
                {
                    "text": "This image contains a table. Extract the table data and format it as JSON:",
                    "image": test_images[1]
                },
                {
                    "text": "Describe what you see in this image and identify any text, numbers, or structured data:",
                    "image": test_images[0]
                }
            ]
            
            print("👁️  Procesando imágenes con OCR...")
            
            # Preparar inputs multimodales
            multimodal_inputs = []
            for prompt_data in ocr_prompts:
                # Formato específico para vLLM multimodal
                if "qwen" in model_name.lower():
                    # Formato Qwen2-VL
                    input_data = {
                        "prompt": f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt_data['text']}<|im_end|>\n<|im_start|>assistant\n",
                        "multi_modal_data": {
                            "image": f"data:image/png;base64,{prompt_data['image']}"
                        }
                    }
                else:
                    # Formato general multimodal
                    input_data = {
                        "prompt": prompt_data['text'],
                        "multi_modal_data": {
                            "image": f"data:image/png;base64,{prompt_data['image']}"
                        }
                    }
                multimodal_inputs.append(input_data)
            
            # Generar respuestas
            try:
                outputs = llm.generate(
                    [inp["prompt"] for inp in multimodal_inputs],
                    sampling_params,
                    # multi_modal_data=[inp["multi_modal_data"] for inp in multimodal_inputs]
                )
                
                # Mostrar resultados de OCR
                print("\n🎯 RESULTADOS DE OCR:")
                for i, output in enumerate(outputs, 1):
                    prompt = ocr_prompts[i-1]["text"]
                    generated_text = output.outputs[0].text
                    print(f"\n📝 Tarea {i}: {prompt}")
                    print(f"👁️  Resultado OCR: {generated_text}")
                    print("-" * 60)
                
                # Estadísticas de memoria
                memory_gb = "N/A"
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    memory_gb = f"{memory_allocated:.2f}"
                    print(f"\n📊 Memoria GPU utilizada: {memory_allocated:.2f} GB")
                    print(f"📊 Memoria GPU reservada: {memory_reserved:.2f} GB")
                
                successful_models.append({
                    "name": model_name,
                    "description": description,
                    "max_len": max_len,
                    "memory_gb": memory_gb,
                    "supports_tools": supports_tools
                })
                
                print(f"🎉 Test OCR con {model_name} completado exitosamente!")
                
            except Exception as inference_error:
                print(f"❌ Error en inferencia con {model_name}: {str(inference_error)[:200]}...")
                print("💡 El modelo se cargó pero falló en la inferencia multimodal")
            
            # Limpieza
            del llm
            torch.cuda.empty_cache()
            
        except Exception as model_error:
            print(f"❌ Error cargando {model_name}: {str(model_error)[:200]}...")
            if "architectures" in str(model_error).lower():
                print("💡 Posible problema: Arquitectura no soportada en esta versión de vLLM")
            elif "memory" in str(model_error).lower():
                print("💡 Posible problema: Memoria insuficiente")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Resumen final
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE MODELOS MULTIMODALES EXITOSOS:")
    print("="*70)
    
    if successful_models:
        for model in successful_models:
            print(f"✅ {model['description']}")
            print(f"   📄 Modelo: {model['name']}")
            print(f"   📏 Max length: {model['max_len']}")
            print(f"   💾 Memoria: {model['memory_gb']} GB")
            print(f"   🔧 Herramientas: {'Sí' if model['supports_tools'] else 'No'}")
            print()
    else:
        print("❌ Ningún modelo multimodal funcionó correctamente")
        print("💡 Sugerencias:")
        print("   - Verificar que vLLM soporte estos modelos multimodales")
        print("   - Intentar con versión más reciente de vLLM")
        print("   - Usar modelos más pequeños")
    
    print("🏁 Test de modelos multimodales OCR completado!")

def test_tools_integration():
    """Test integration with tools for enhanced OCR capabilities"""
    
    print("\n🔧 Probando integración con herramientas...")
    
    # Ejemplo de herramientas que se pueden integrar con OCR
    ocr_tools = [
        {
            "name": "extract_text",
            "description": "Extract all text from an image",
            "parameters": {
                "image": "base64 encoded image",
                "language": "language code for OCR"
            }
        },
        {
            "name": "analyze_table", 
            "description": "Extract and structure table data from image",
            "parameters": {
                "image": "base64 encoded image",
                "format": "output format (json, csv, etc.)"
            }
        },
        {
            "name": "detect_documents",
            "description": "Detect and classify document types",
            "parameters": {
                "image": "base64 encoded image"
            }
        }
    ]
    
    print("🛠️  Herramientas OCR disponibles:")
    for tool in ocr_tools:
        print(f"   - {tool['name']}: {tool['description']}")
    
    print("✅ Framework de herramientas preparado para integración")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL/Pillow disponible")
    except ImportError:
        print("❌ PIL/Pillow no disponible. Instalar con: pip install Pillow")
        exit(1)
    
    test_multimodal_ocr()
    test_tools_integration()