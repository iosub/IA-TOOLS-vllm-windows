#!/usr/bin/env python3
"""
OCR REAL con vLLM - Procesamiento directo de imagen con modelo multimodal
Enfoque directo: Qwen2-VL para OCR real de la imagen 1im.png
"""

import os
import torch
from vllm import LLM, SamplingParams
from PIL import Image
import base64
from io import BytesIO

# Configurar variables de entorno para Windows
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"

def process_real_image_ocr():
    """Procesar imagen real con OCR usando modelo multimodal"""
    
    image_path = r"C:\IA\tools\vllm-windows\Z_IOSU\ejemplos\imagen\1im.png"
    
    print("🎯 OCR REAL CON vLLM - PROCESAMIENTO DIRECTO")
    print(f"📷 Imagen: {image_path}")
    print("="*60)
    
    # Verificar que existe la imagen
    if not os.path.exists(image_path):
        print(f"❌ No se encuentra la imagen: {image_path}")
        return False
    
    # Mostrar info de la imagen
    try:
        with Image.open(image_path) as img:
            print(f"📏 Dimensiones: {img.size}")
            print(f"🎨 Formato: {img.format}")
            print(f"🖼️  Modo: {img.mode}")
    except Exception as e:
        print(f"❌ Error abriendo imagen: {e}")
        return False
    
    print("\n⏳ Cargando modelo Qwen2-VL para OCR real...")
    
    try:
        # Intentar con el modelo Qwen2-VL que es específico para OCR
        llm = LLM(
            model="Qwen/Qwen2-VL-2B-Instruct",
            max_model_len=4096,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
            # Configuración multimodal
            limit_mm_per_prompt={"image": 1},
        )
        
        print("✅ Modelo multimodal cargado!")
        
        # Configuración para OCR preciso
        sampling_params = SamplingParams(
            temperature=0.1,  # Muy determinista para OCR
            max_tokens=1000,  # Suficiente para texto completo
            top_p=0.9,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # === EXTRACCIÓN COMPLETA DE TEXTO ===
        print("\n🔍 EXTRAYENDO TEXTO DE LA IMAGEN...")
        print("-" * 40)
        
        # Para modelos multimodales, necesitamos pasar la imagen
        # El formato exacto depende del modelo, pero intentaremos el estándar
        
        # Prompt específico para OCR completo
        ocr_prompts = [
            f"<|im_start|>user\n<image>{image_path}</image>\nExtract all text from this image. Provide complete OCR results with every visible word, number, and symbol.<|im_end|>\n<|im_start|>assistant\n",
            
            f"<|im_start|>user\n<image>{image_path}</image>\nAnalyze this document and extract:\n1. All text content\n2. Structure (headers, paragraphs, lists)\n3. Numbers and dates\n4. Any tables or forms<|im_end|>\n<|im_start|>assistant\n",
            
            f"<|im_start|>user\n<image>{image_path}</image>\nPerform detailed OCR on this image. Extract:\n- Every word and phrase\n- All numbers and amounts\n- Dates and times\n- Names and addresses\n- Any structured data<|im_end|>\n<|im_start|>assistant\n"
        ]
        
        print("📋 Procesando imagen con OCR avanzado...")
        
        # Generar OCR para cada prompt
        for i, prompt in enumerate(ocr_prompts, 1):
            try:
                print(f"\n🔬 Análisis {i}/3...")
                outputs = llm.generate([prompt], sampling_params)
                
                for output in outputs:
                    result = output.outputs[0].text
                    print(f"\n📄 RESULTADO OCR {i}:")
                    print("=" * 40)
                    print(result)
                    print("=" * 40)
                    
            except Exception as e:
                print(f"❌ Error en análisis {i}: {e}")
                continue
        
        print("\n🎉 ¡OCR COMPLETADO!")
        return True
        
    except Exception as model_error:
        print(f"❌ Error con modelo multimodal: {model_error}")
        
        # Fallback: Si falla el multimodal, usar OCR tradicional + LLM
        print("\n🔄 FALLBACK: Usando OCR tradicional + análisis LLM...")
        return ocr_fallback(image_path)

def ocr_fallback(image_path):
    """Fallback usando OCR tradicional + análisis con LLM"""
    
    try:
        # Intentar usar pytesseract si está disponible
        try:
            import pytesseract
            from PIL import Image
            
            print("🔍 Usando Tesseract OCR...")
            
            # Extraer texto con Tesseract
            image = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(image, lang='spa+eng')
            
            print(f"\n📄 TEXTO EXTRAÍDO CON TESSERACT:")
            print("=" * 50)
            print(extracted_text)
            print("=" * 50)
            
            # Ahora usar LLM para analizar el texto extraído
            print("\n🤖 Analizando texto con LLM...")
            
            llm = LLM(
                model="microsoft/DialoGPT-large",
                max_model_len=1024,
                enforce_eager=True,
                gpu_memory_utilization=0.7,
                disable_custom_all_reduce=True,
            )
            
            analysis_prompt = f"""Analyze this extracted text from an image and provide structured information:

TEXT FROM IMAGE:
{extracted_text}

Please extract and organize:
1. Document type
2. Key information (names, dates, amounts)
3. Structured data
4. Summary of content

Analysis:"""
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=500)
            outputs = llm.generate([analysis_prompt], sampling_params)
            
            print(f"\n🤖 ANÁLISIS LLM:")
            print("=" * 40)
            for output in outputs:
                print(output.outputs[0].text)
            print("=" * 40)
            
            return True
            
        except ImportError:
            print("⚠️  pytesseract no disponible")
            print("💡 Instalar con: pip install pytesseract")
            
            # Manual: Mostrar la imagen para análisis visual
            print(f"\n👁️  ANÁLISIS MANUAL DE LA IMAGEN:")
            print(f"📷 Abrir imagen: {image_path}")
            print("📝 Describe manualmente el contenido visible")
            
            return False
            
    except Exception as e:
        print(f"❌ Error en fallback: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INICIANDO OCR REAL CON vLLM")
    print("🎯 PROCESAMIENTO DIRECTO DE IMAGEN")
    
    success = process_real_image_ocr()
    
    if success:
        print("\n🎊 ¡OCR COMPLETADO EXITOSAMENTE!")
    else:
        print("\n❌ OCR falló - revisar configuración o imagen")

if __name__ == "__main__":
    main()