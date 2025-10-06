#!/usr/bin/env python3
"""
Test script específico para Qwen2-VL-OCR-2B-Instruct
Optimizado para OCR con herramientas en Windows
"""

import os
import torch
import base64
import io
import json
from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams
from typing import List, Dict, Any

# Configurar variables de entorno específicas para Qwen2-VL en Windows
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"
# Desactivar problemas conocidos con multimodal en Windows
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

def create_ocr_test_images() -> List[Dict[str, str]]:
    """Crear imágenes específicas para testing OCR avanzado"""
    
    images_data = []
    
    # Imagen 1: Documento con múltiples elementos
    img1 = Image.new('RGB', (600, 400), color='white')
    draw1 = ImageDraw.Draw(img1)
    
    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_normal = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Título
    draw1.text((50, 30), "INVOICE #INV-2024-001", fill='black', font=font_title)
    draw1.line([50, 70, 550, 70], fill='black', width=2)
    
    # Información del documento
    draw1.text((50, 90), "Date: October 5, 2024", fill='black', font=font_normal)
    draw1.text((50, 120), "Customer: John Doe", fill='black', font=font_normal)
    draw1.text((50, 150), "Address: 123 Main St, City, 12345", fill='black', font=font_normal)
    
    # Tabla de productos
    draw1.rectangle([50, 200, 550, 350], outline='black', width=1)
    draw1.line([50, 230, 550, 230], fill='black', width=1)
    draw1.line([250, 200, 250, 350], fill='black', width=1)
    draw1.line([350, 200, 350, 350], fill='black', width=1)
    draw1.line([450, 200, 450, 350], fill='black', width=1)
    
    # Headers
    draw1.text((60, 210), "Product", fill='black', font=font_normal)
    draw1.text((260, 210), "Qty", fill='black', font=font_normal)
    draw1.text((360, 210), "Price", fill='black', font=font_normal)
    draw1.text((460, 210), "Total", fill='black', font=font_normal)
    
    # Items
    draw1.text((60, 250), "Laptop Computer", fill='black', font=font_small)
    draw1.text((260, 250), "2", fill='black', font=font_small)
    draw1.text((360, 250), "$999.00", fill='black', font=font_small)
    draw1.text((460, 250), "$1,998.00", fill='black', font=font_small)
    
    draw1.text((60, 280), "Wireless Mouse", fill='black', font=font_small)
    draw1.text((260, 280), "3", fill='black', font=font_small)
    draw1.text((360, 280), "$25.00", fill='black', font=font_small)
    draw1.text((460, 280), "$75.00", fill='black', font=font_small)
    
    # Total
    draw1.text((360, 320), "TOTAL:", fill='black', font=font_normal)
    draw1.text((460, 320), "$2,073.00", fill='black', font=font_normal)
    
    buffer1 = io.BytesIO()
    img1.save(buffer1, format='PNG')
    img1_b64 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    
    images_data.append({
        "name": "invoice_document",
        "base64": img1_b64,
        "description": "Documento de factura con tabla estructurada"
    })
    
    # Imagen 2: Formulario con campos
    img2 = Image.new('RGB', (500, 600), color='white')
    draw2 = ImageDraw.Draw(img2)
    
    # Título del formulario
    draw2.text((50, 30), "APPLICATION FORM", fill='black', font=font_title)
    draw2.line([50, 70, 450, 70], fill='black', width=2)
    
    # Campos del formulario
    fields = [
        ("Name:", "Maria Garcia", 100),
        ("Email:", "maria.garcia@email.com", 140),
        ("Phone:", "+1 (555) 123-4567", 180),
        ("Date of Birth:", "15/03/1990", 220),
        ("Address:", "456 Oak Avenue", 260),
        ("City:", "Springfield", 300),
        ("ZIP Code:", "54321", 340),
        ("Occupation:", "Software Engineer", 380)
    ]
    
    for field, value, y_pos in fields:
        draw2.text((50, y_pos), field, fill='black', font=font_normal)
        draw2.rectangle([150, y_pos-5, 400, y_pos+20], outline='gray', width=1)
        draw2.text((160, y_pos), value, fill='blue', font=font_normal)
    
    # Checkbox section
    draw2.text((50, 450), "□ I agree to terms and conditions", fill='black', font=font_normal)
    draw2.text((50, 480), "☑ Subscribe to newsletter", fill='black', font=font_normal)
    
    # Signature
    draw2.text((50, 520), "Signature: ___________________", fill='black', font=font_normal)
    draw2.text((50, 550), "Date: October 5, 2024", fill='black', font=font_normal)
    
    buffer2 = io.BytesIO()
    img2.save(buffer2, format='PNG')
    img2_b64 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    
    images_data.append({
        "name": "application_form",
        "base64": img2_b64,
        "description": "Formulario de aplicación con campos estructurados"
    })
    
    return images_data

def test_qwen2vl_ocr():
    """Test específico para Qwen2-VL-OCR-2B-Instruct"""
    
    print("🚀 Probando Qwen2-VL-OCR-2B-Instruct para OCR avanzado...")
    print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Crear imágenes de prueba OCR
    print("🖼️  Creando imágenes de prueba especializadas para OCR...")
    test_images = create_ocr_test_images()
    print(f"✅ Creadas {len(test_images)} imágenes de prueba")
    
    try:
        # Limpiar memoria GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n📋 Cargando Qwen2-VL-OCR-2B-Instruct...")
        
        # Configuración específica para Qwen2-VL
        llm = LLM(
            model="prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
            max_model_len=1024,  # Conservador para estabilidad
            enforce_eager=True,  # Esencial para Windows
            gpu_memory_utilization=0.75,
            disable_custom_all_reduce=True,
            disable_log_stats=True,
            trust_remote_code=True,
            # Configuraciones específicas para multimodal
            limit_mm_per_prompt={"image": 1},  # Una imagen por prompt
            mm_processor_kwargs={
                "num_crops": 16,  # Mejor OCR con más crops
            },
        )
        
        print("✅ Qwen2-VL cargado exitosamente!")
        
        # Sampling parameters optimizados para OCR
        sampling_params = SamplingParams(
            temperature=0.05,  # Muy baja para OCR preciso
            top_p=0.8,
            max_tokens=800,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # Prompts especializados para OCR con Qwen2-VL
        ocr_tasks = [
            {
                "name": "text_extraction",
                "prompt": "Extract all text from this image. Organize the text maintaining its structure and format. Include all numbers, dates, and special characters exactly as they appear.",
                "image_idx": 0
            },
            {
                "name": "structured_data_extraction", 
                "prompt": "Analyze this document and extract the structured data as JSON. Include all table data, fields, and values. Format the response as valid JSON.",
                "image_idx": 0
            },
            {
                "name": "form_field_extraction",
                "prompt": "This is a form with various fields. Extract all field names and their corresponding values. Present the information in a clear, organized format.",
                "image_idx": 1
            },
            {
                "name": "document_analysis",
                "prompt": "Analyze this document and provide: 1) Document type, 2) Key information extracted, 3) Any important numbers or dates, 4) Overall structure description.",
                "image_idx": 0
            }
        ]
        
        print("👁️  Ejecutando tareas de OCR especializadas...")
        
        results = []
        
        for task in ocr_tasks:
            print(f"\n🔍 Ejecutando: {task['name']}")
            
            # Preparar prompt en formato Qwen2-VL
            image_data = test_images[task['image_idx']]
            
            # Formato específico para Qwen2-VL
            qwen_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{task['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            
            try:
                # Generar respuesta
                outputs = llm.generate(
                    [qwen_prompt],
                    sampling_params
                )
                
                result = {
                    "task": task['name'],
                    "prompt": task['prompt'],
                    "image": image_data['description'],
                    "response": outputs[0].outputs[0].text
                }
                results.append(result)
                
                print(f"✅ {task['name']} completado")
                
            except Exception as task_error:
                print(f"❌ Error en {task['name']}: {str(task_error)[:100]}...")
                continue
        
        # Mostrar todos los resultados
        print(f"\n{'='*80}")
        print("📊 RESULTADOS DE OCR CON QWEN2-VL")
        print("="*80)
        
        for result in results:
            print(f"\n🎯 TAREA: {result['task'].upper()}")
            print(f"📋 Prompt: {result['prompt']}")
            print(f"🖼️  Imagen: {result['image']}")
            print(f"🤖 Resultado OCR:")
            print("-" * 60)
            print(result['response'])
            print("-" * 60)
        
        # Estadísticas finales
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n📊 Memoria GPU utilizada: {memory_allocated:.2f} GB")
            print(f"📊 Memoria GPU reservada: {memory_reserved:.2f} GB")
        
        print(f"\n🎉 Test de OCR con Qwen2-VL completado exitosamente!")
        print(f"✅ Procesadas {len(results)} tareas de OCR")
        
        # Limpieza
        del llm
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error con Qwen2-VL: {str(e)}")
        
        if "architectures" in str(e).lower():
            print("\n💡 SOLUCIÓN POSIBLE:")
            print("   - El modelo Qwen2-VL puede no estar soportado en esta versión de vLLM")
            print("   - Intentar con: Qwen/Qwen2-VL-2B-Instruct (versión oficial)")
            print("   - Verificar compatibilidad de vLLM con modelos multimodales")
        
        elif "memory" in str(e).lower():
            print("\n💡 SOLUCIÓN POSIBLE:")
            print("   - Reducir gpu_memory_utilization a 0.6")
            print("   - Reducir max_model_len a 512")
            print("   - Cerrar otras aplicaciones que usen GPU")
        
        else:
            print("\n💡 SOLUCIONES POSIBLES:")
            print("   - Verificar que el modelo esté disponible en HuggingFace")
            print("   - Intentar con enforce_eager=True")
            print("   - Revisar logs detallados para más información")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def create_ocr_tools_framework():
    """Crear framework de herramientas para OCR"""
    
    print(f"\n{'='*60}")
    print("🔧 FRAMEWORK DE HERRAMIENTAS OCR")
    print("="*60)
    
    ocr_tools = {
        "text_extraction": {
            "name": "extract_all_text",
            "description": "Extrae todo el texto de una imagen manteniendo formato",
            "input": "imagen (base64)",
            "output": "texto estructurado"
        },
        "table_extraction": {
            "name": "extract_table_data", 
            "description": "Extrae datos de tablas y los convierte a JSON/CSV",
            "input": "imagen con tabla",
            "output": "datos estructurados"
        },
        "form_processing": {
            "name": "process_form_fields",
            "description": "Procesa formularios y extrae campos y valores",
            "input": "imagen de formulario",
            "output": "campos estructurados"
        },
        "document_classification": {
            "name": "classify_document",
            "description": "Clasifica tipo de documento y extrae metadatos",
            "input": "imagen de documento",
            "output": "clasificación y metadatos"
        },
        "handwriting_recognition": {
            "name": "recognize_handwriting",
            "description": "Reconoce texto manuscrito",
            "input": "imagen con texto manuscrito",
            "output": "texto digitalizado"
        }
    }
    
    print("🛠️  Herramientas OCR disponibles:")
    for tool_id, tool_info in ocr_tools.items():
        print(f"\n   📌 {tool_info['name']}:")
        print(f"      📄 {tool_info['description']}")
        print(f"      ⬅️  Input: {tool_info['input']}")
        print(f"      ➡️  Output: {tool_info['output']}")
    
    print(f"\n✅ Framework OCR preparado con {len(ocr_tools)} herramientas")
    return ocr_tools

if __name__ == "__main__":
    # Verificar dependencias
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL/Pillow disponible")
    except ImportError:
        print("❌ PIL/Pillow no disponible.")
        print("💡 Instalar con: pip install Pillow")
        exit(1)
    
    # Ejecutar tests
    test_qwen2vl_ocr()
    create_ocr_tools_framework()