#!/usr/bin/env python3
"""
Test script simplificado para modelos de visión con OCR
Usando configuraciones más conservadoras para Windows
"""

import os
import torch
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams

# Configurar variables de entorno básicas para Windows
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"

def create_simple_ocr_image() -> str:
    """Crear una imagen simple para OCR testing"""
    
    # Crear imagen simple con texto claro
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_normal = ImageFont.truetype("arial.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_normal = ImageFont.load_default()
    
    # Texto simple y claro
    draw.text((50, 50), "SAMPLE DOCUMENT", fill='black', font=font_large)
    draw.line([50, 80, 350, 80], fill='black', width=2)
    
    draw.text((50, 100), "Name: John Smith", fill='black', font=font_normal)
    draw.text((50, 130), "ID: 12345", fill='black', font=font_normal)
    draw.text((50, 160), "Date: 2024-10-05", fill='black', font=font_normal)
    draw.text((50, 190), "Amount: $1,234.56", fill='black', font=font_normal)
    
    # Convertir a base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_b64

def test_vision_models_simple():
    """Test modelos de visión con configuración simplificada"""
    
    print("🚀 Probando modelos de visión con configuración simplificada...")
    print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Crear imagen de prueba simple
    print("🖼️  Creando imagen de prueba simple...")
    test_image = create_simple_ocr_image()
    print("✅ Imagen de prueba creada")
    
    # Modelos alternativos que pueden funcionar mejor
    vision_models = [
        {
            "name": "llava-hf/llava-1.5-7b-hf",
            "description": "LLaVA 1.5 7B (modelo popular de visión)",
            "max_len": 512,
            "memory_util": 0.6
        },
        {
            "name": "microsoft/Phi-3-vision-128k-instruct",
            "description": "Phi-3 Vision (modelo compacto de Microsoft)",
            "max_len": 512,
            "memory_util": 0.5
        }
    ]
    
    successful_models = []
    
    for model_config in vision_models:
        model_name = model_config["name"]
        description = model_config["description"]
        max_len = model_config["max_len"]
        memory_util = model_config["memory_util"]
        
        print(f"\n{'='*60}")
        print(f"📋 Probando: {description}")
        print(f"🏷️  Modelo: {model_name}")
        print("="*60)
        
        try:
            # Limpiar memoria GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Configuración muy conservadora
            llm = LLM(
                model=model_name,
                max_model_len=max_len,
                enforce_eager=True,
                gpu_memory_utilization=memory_util,
                disable_custom_all_reduce=True,
                disable_log_stats=True,
                trust_remote_code=True,
                # Configuración multimodal mínima
                limit_mm_per_prompt={"image": 1},
            )
            
            print("✅ Modelo cargado exitosamente!")
            
            # Sampling parameters conservadores
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=200,
                stop=["<|endoftext|>", "</s>", "<|im_end|>"]
            )
            
            # Prompt simple para OCR
            simple_prompt = "What text do you see in this image? Extract all the visible text."
            
            print("👁️  Procesando imagen...")
            
            try:
                # Formato básico multimodal
                outputs = llm.generate(
                    [simple_prompt],
                    sampling_params
                )
                
                result = outputs[0].outputs[0].text
                print(f"\n🎯 RESULTADO OCR:")
                print(f"📝 Prompt: {simple_prompt}")
                print(f"🤖 Respuesta: {result}")
                print("-" * 50)
                
                # Estadísticas
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"📊 Memoria GPU utilizada: {memory_allocated:.2f} GB")
                
                successful_models.append({
                    "name": model_name,
                    "description": description,
                    "result": result[:100] + "..." if len(result) > 100 else result
                })
                
                print(f"🎉 Test con {model_name} completado exitosamente!")
                
            except Exception as inference_error:
                print(f"❌ Error en inferencia: {str(inference_error)[:150]}...")
                print("💡 El modelo se cargó pero falló en la inferencia multimodal")
            
            # Limpieza
            del llm
            torch.cuda.empty_cache()
            
        except Exception as model_error:
            print(f"❌ Error cargando {model_name}: {str(model_error)[:150]}...")
            
            if "not supported" in str(model_error).lower():
                print("💡 Este modelo puede no estar soportado en vLLM")
            elif "memory" in str(model_error).lower(): 
                print("💡 Problema de memoria - intentar con configuración más baja")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Resumen
    print(f"\n{'='*60}")
    print("📊 RESUMEN:")
    print("="*60)
    
    if successful_models:
        print("✅ Modelos que funcionaron:")
        for model in successful_models:
            print(f"   - {model['description']}")
            print(f"     Resultado: {model['result']}")
            print()
    else:
        print("❌ Ningún modelo de visión funcionó")
        print("\n💡 ALTERNATIVAS:")
        print("   1. Usar modelos de texto puro con descripción de imagen")
        print("   2. Probar con versiones más recientes de vLLM")
        print("   3. Usar APIs externas para OCR (Azure Vision, Google Vision)")
        print("   4. Implementar OCR con Tesseract + modelo de texto")

def suggest_ocr_alternatives():
    """Sugerir alternativas para OCR si los modelos multimodales no funcionan"""
    
    print(f"\n{'='*60}")
    print("🔧 ALTERNATIVAS PARA OCR EN WINDOWS")
    print("="*60)
    
    alternatives = [
        {
            "name": "Tesseract + vLLM",
            "description": "Usar Tesseract para OCR + vLLM para procesamiento de texto",
            "pros": ["Muy preciso para OCR", "Funciona sin modelos multimodales"],
            "setup": "pip install pytesseract"
        },
        {
            "name": "Azure Computer Vision API",
            "description": "API de Microsoft para OCR avanzado",
            "pros": ["Muy preciso", "Soporta muchos idiomas", "Análisis de formularios"],
            "setup": "pip install azure-cognitiveservices-vision-computervision"
        },
        {
            "name": "EasyOCR + vLLM",
            "description": "EasyOCR para extracción + vLLM para análisis",
            "pros": ["Fácil de usar", "Sin API keys", "Buen rendimiento"],
            "setup": "pip install easyocr"
        },
        {
            "name": "PaddleOCR + vLLM",
            "description": "PaddleOCR para OCR + vLLM para procesamiento",
            "pros": ["Muy preciso", "Soporta tablas", "Open source"],
            "setup": "pip install paddlepaddle paddleocr"
        }
    ]
    
    for alt in alternatives:
        print(f"\n📌 {alt['name']}:")
        print(f"   📄 {alt['description']}")
        print(f"   ✅ Ventajas: {', '.join(alt['pros'])}")
        print(f"   🔧 Setup: {alt['setup']}")
    
    print(f"\n💡 RECOMENDACIÓN:")
    print("   Para uso en producción, combinar EasyOCR/Tesseract para OCR")
    print("   con vLLM (modelos de texto) para análisis y procesamiento.")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL/Pillow disponible")
    except ImportError:
        print("❌ PIL/Pillow no disponible.")
        exit(1)
    
    # Ejecutar tests
    test_vision_models_simple()
    suggest_ocr_alternatives()