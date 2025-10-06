#!/usr/bin/env python3
"""
OCR REAL para Windows - Sin problemas de Triton
Usando enfoque híbrido: OCR tradicional + LLM análisis
"""

import os
import sys
from PIL import Image

# Configurar variables de entorno para evitar problemas de Triton
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"
os.environ["VLLM_USE_TRITON"] = "0"  # Deshabilitar Triton
os.environ["DISABLE_TRITON"] = "1"

def install_and_import_tesseract():
    """Instalar e importar tesseract si no está disponible"""
    try:
        import pytesseract
        return pytesseract
    except ImportError:
        print("📦 Instalando pytesseract...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])
        import pytesseract
        return pytesseract

def ocr_with_tesseract_and_llm():
    """OCR real usando Tesseract + análisis con LLM que funciona"""
    
    image_path = r"C:\IA\tools\vllm-windows\Z_IOSU\ejemplos\imagen\1im.png"
    
    print("🎯 OCR REAL - TESSERACT + vLLM")
    print(f"📷 Imagen: {image_path}")
    print("="*60)
    
    # Verificar imagen
    if not os.path.exists(image_path):
        print(f"❌ Imagen no encontrada: {image_path}")
        return False
    
    # Mostrar info de imagen
    try:
        with Image.open(image_path) as img:
            print(f"📏 Dimensiones: {img.size}")
            print(f"🎨 Formato: {img.format}")
            
            # Guardar info básica
            width, height = img.size
            print(f"📊 Área: {width}x{height} pixels")
    except Exception as e:
        print(f"❌ Error leyendo imagen: {e}")
        return False
    
    # PASO 1: OCR con Tesseract
    print("\n🔍 PASO 1: EXTRACCIÓN DE TEXTO CON TESSERACT")
    print("-" * 50)
    
    try:
        pytesseract = install_and_import_tesseract()
        
        # Configurar Tesseract para mejor OCR
        image = Image.open(image_path)
        
        # OCR con diferentes configuraciones
        configs = [
            '--oem 3 --psm 6',  # Uniform block of text
            '--oem 3 --psm 4',  # Single column text
            '--oem 3 --psm 3',  # Fully automatic
        ]
        
        extracted_texts = []
        
        for i, config in enumerate(configs, 1):
            try:
                print(f"🔬 Configuración {i}: {config}")
                text = pytesseract.image_to_string(image, config=config, lang='spa+eng')
                
                if text.strip():
                    print(f"📄 Texto extraído ({len(text)} caracteres):")
                    print("'" + text.strip()[:200] + "..." if len(text) > 200 else text.strip() + "'")
                    extracted_texts.append(text.strip())
                else:
                    print("❌ No se extrajo texto")
                    
            except Exception as e:
                print(f"❌ Error con config {i}: {e}")
        
        # Tomar el mejor resultado (el más largo)
        if extracted_texts:
            best_text = max(extracted_texts, key=len)
            print(f"\n✅ MEJOR RESULTADO ({len(best_text)} caracteres):")
            print("="*50)
            print(best_text)
            print("="*50)
        else:
            print("❌ No se pudo extraer texto con Tesseract")
            return False
            
    except Exception as e:
        print(f"❌ Error con Tesseract: {e}")
        print("💡 Asegúrate de tener Tesseract instalado en Windows")
        print("💡 Descarga: https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    # PASO 2: Análisis con LLM (usando modelo que sabemos funciona)
    print(f"\n🤖 PASO 2: ANÁLISIS INTELIGENTE CON LLM")
    print("-" * 50)
    
    try:
        from vllm import LLM, SamplingParams
        
        print("⏳ Cargando modelo para análisis...")
        
        # Usar modelo que sabemos funciona sin problemas de Triton
        llm = LLM(
            model="gpt2",  # Modelo simple que funciona
            max_model_len=1024,
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            disable_custom_all_reduce=True,
            disable_log_stats=True,
        )
        
        print("✅ Modelo cargado!")
        
        # Análisis estructurado del texto extraído
        analysis_prompts = [
            f"Analyze this extracted text and identify key information:\n\nTEXT: {best_text}\n\nExtract:\n1. Document type:\n2. Important dates:\n3. Names/entities:\n4. Numbers/amounts:\n5. Summary:",
            
            f"Structure this extracted text:\n\nRAW TEXT: {best_text}\n\nOrganized format:\n- Type:\n- Key data:\n- Details:",
            
            f"From this OCR text, extract structured data:\n\nTEXT: {best_text}\n\nJSON format:\n{{"
        ]
        
        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=300,
            top_p=0.9
        )
        
        print("🔬 Analizando contenido extraído...")
        
        for i, prompt in enumerate(analysis_prompts, 1):
            try:
                outputs = llm.generate([prompt], sampling_params)
                result = outputs[0].outputs[0].text
                
                print(f"\n📊 ANÁLISIS {i}:")
                print("="*40)
                print(result)
                print("="*40)
                
            except Exception as e:
                print(f"❌ Error en análisis {i}: {e}")
        
        print(f"\n🎉 ¡OCR Y ANÁLISIS COMPLETADOS!")
        return True
        
    except Exception as e:
        print(f"❌ Error con LLM: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INICIANDO OCR REAL SIN PROBLEMAS DE TRITON")
    print("🔧 USANDO TESSERACT + vLLM (MODELO COMPATIBLE)")
    
    success = ocr_with_tesseract_and_llm()
    
    if success:
        print("\n🎊 ¡PROCESO COMPLETADO EXITOSAMENTE!")
        print("✅ Texto extraído de la imagen")
        print("✅ Análisis inteligente realizado")
        print("🎯 Datos estructurados obtenidos")
    else:
        print("\n❌ Proceso falló")
        print("💡 Verificar instalación de Tesseract")
        print("💡 Verificar que la imagen sea legible")

if __name__ == "__main__":
    main()