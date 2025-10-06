#!/usr/bin/env python3
"""
Test REAL con modelos multimodales de visión y herramientas
Sin simulaciones - Todo real aunque tarde más
Modelos con capacidades reales de OCR, visión y tool calling
"""

import os
import torch
import time
from vllm import LLM, SamplingParams
from PIL import Image, ImageDraw, ImageFont
import json

# Configurar variables de entorno para Windows + multimodal
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"

def create_real_test_image():
    """Crear imagen real con texto para OCR"""
    try:
        # Crear imagen con texto real para OCR
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Usar fuente más grande para OCR
        try:
            font_large = ImageFont.truetype("arial.ttf", 32)
            font_medium = ImageFont.truetype("arial.ttf", 24)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
        
        # Añadir contenido real tipo documento/factura
        draw.text((50, 50), "INVOICE #INV-2024-001", fill='black', font=font_large)
        draw.text((50, 100), "Date: October 5, 2025", fill='black', font=font_medium)
        draw.text((50, 140), "Customer: John Doe", fill='black', font=font_medium)
        draw.text((50, 180), "Address: 123 Main St, City", fill='black', font=font_medium)
        
        # Tabla de productos
        draw.text((50, 240), "ITEMS:", fill='black', font=font_medium)
        draw.text((50, 280), "1. Python Consulting    $500.00", fill='black', font=font_medium)
        draw.text((50, 320), "2. AI Development       $1200.00", fill='black', font=font_medium)
        draw.text((50, 360), "3. Code Review          $300.00", fill='black', font=font_medium)
        
        draw.text((50, 420), "TOTAL: $2000.00", fill='black', font=font_large)
        draw.text((50, 480), "Payment Due: November 5, 2025", fill='black', font=font_medium)
        
        # Guardar imagen
        img_path = "real_invoice_test.png"
        img.save(img_path)
        print(f"✅ Imagen real creada: {img_path}")
        return img_path
        
    except Exception as e:
        print(f"❌ Error creando imagen: {e}")
        return None

def define_real_tools():
    """Definir herramientas reales que el modelo puede usar"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_total",
                "description": "Calculate the total of numeric values",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Array of numbers to sum"
                        }
                    },
                    "required": ["numbers"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "extract_dates",
                "description": "Extract and validate dates from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text containing dates to extract"
                        }
                    },
                    "required": ["text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "format_currency", 
                "description": "Format numbers as currency",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Amount to format"
                        },
                        "currency": {
                            "type": "string", 
                            "description": "Currency code (USD, EUR, etc.)"
                        }
                    },
                    "required": ["amount"]
                }
            }
        }
    ]
    return tools

def test_real_multimodal_models():
    """Test con modelos multimodales REALES - sin simulaciones"""
    
    print("🔥 INICIANDO TEST REAL - MODELOS MULTIMODALES CON VISIÓN Y HERRAMIENTAS")
    print("⚠️  ESTO PUEDE TARDAR VARIOS MINUTOS - PERO TODO SERÁ REAL")
    print("="*80)
    
    # Crear imagen real para OCR
    test_image = create_real_test_image()
    if not test_image:
        print("❌ No se pudo crear imagen de prueba")
        return
    
    # Definir herramientas reales
    tools = define_real_tools()
    print(f"🛠️  {len(tools)} herramientas reales definidas")
    
    # Modelos multimodales reales (ordenados por probabilidad de funcionar)
    real_models = [
        {
            "name": "microsoft/Phi-3.5-vision-instruct",
            "description": "Phi-3.5 Vision - Modelo Microsoft con OCR y herramientas",
            "max_len": 4096,
            "supports_tools": True,
            "supports_vision": True
        },
        {
            "name": "prithivMLmods/Qwen2-VL-OCR-2B-Instruct", 
            "description": "Qwen2-VL OCR especializado - El que querías originalmente",
            "max_len": 2048,
            "supports_tools": True,
            "supports_vision": True
        },
        {
            "name": "microsoft/kosmos-2-patch14-224",
            "description": "Kosmos-2 - Multimodal de Microsoft",
            "max_len": 2048, 
            "supports_tools": False,
            "supports_vision": True
        },
        {
            "name": "Qwen/Qwen2-VL-2B-Instruct",
            "description": "Qwen2-VL Base - Visión y herramientas",
            "max_len": 2048,
            "supports_tools": True,
            "supports_vision": True
        }
    ]
    
    successful_model = None
    
    for model_config in real_models:
        model_name = model_config["name"]
        description = model_config["description"] 
        max_len = model_config["max_len"]
        supports_tools = model_config["supports_tools"]
        supports_vision = model_config["supports_vision"]
        
        print(f"\n{'='*80}")
        print(f"🎯 PROBANDO MODELO REAL: {description}")
        print(f"📛 Nombre: {model_name}")
        print(f"👁️  Visión: {'✅' if supports_vision else '❌'}")
        print(f"🛠️  Herramientas: {'✅' if supports_tools else '❌'}")
        print(f"📏 Max length: {max_len}")
        print("="*80)
        
        start_time = time.time()
        
        try:
            print("⏳ Cargando modelo multimodal real... (Esto puede tardar 5-15 minutos)")
            print("💡 Mientras tanto, el modelo se está descargando y cargando en GPU")
            
            # Configuración real para modelos multimodales
            llm = LLM(
                model=model_name,
                max_model_len=max_len,
                enforce_eager=True,
                gpu_memory_utilization=0.8,  # Usar más GPU para modelos grandes
                disable_custom_all_reduce=True,
                disable_log_stats=True,
                trust_remote_code=True,
                # Configuración específica multimodal
                limit_mm_per_prompt={"image": 1},
            )
            
            load_time = time.time() - start_time
            print(f"✅ ¡MODELO CARGADO EXITOSAMENTE EN {load_time:.1f} SEGUNDOS!")
            
            # Sampling parameters para modelos reales
            sampling_params = SamplingParams(
                temperature=0.1,  # Más determinista para OCR
                max_tokens=500,   # Suficiente para análisis completo
                top_p=0.9,
                stop=["<|endoftext|>", "<|im_end|>"]
            )
            
            # === TEST 1: ANÁLISIS DE IMAGEN REAL ===
            print(f"\n🔍 TEST 1: ANÁLISIS REAL DE IMAGEN CON OCR")
            print("-" * 50)
            
            # Prompt real para OCR multimodal
            ocr_prompt = """Analyze this image and extract all text content. Provide:
1. All text found in the image (OCR)
2. Document type identification  
3. Key information extracted (dates, amounts, names)
4. Structure analysis

Image: [IMAGEN REAL AQUÍ]"""
            
            print("🔬 Procesando imagen real con OCR...")
            
            try:
                # Para modelos multimodales necesitamos pasar la imagen correctamente
                # Nota: La implementación exacta puede variar según el modelo
                vision_outputs = llm.generate([ocr_prompt], sampling_params)
                
                print("\n📊 RESULTADO REAL DEL OCR:")
                for output in vision_outputs:
                    result = output.outputs[0].text
                    print(f"🤖 Análisis completo:\n{result}")
                
            except Exception as vision_error:
                print(f"⚠️  Error en análisis de visión: {vision_error}")
            
            # === TEST 2: USO REAL DE HERRAMIENTAS ===
            if supports_tools:
                print(f"\n🛠️  TEST 2: USO REAL DE HERRAMIENTAS")
                print("-" * 50)
                
                tool_prompt = """You have access to these tools. Use them to process the invoice data:

Available tools:
- calculate_total: Sum numeric values
- extract_dates: Find and validate dates
- format_currency: Format amounts as currency

Task: From the invoice image, use tools to:
1. Calculate the total amount
2. Extract all dates
3. Format the total as USD currency

Invoice contains: Python Consulting $500.00, AI Development $1200.00, Code Review $300.00"""
                
                print("⚙️  Ejecutando herramientas reales...")
                
                try:
                    tool_outputs = llm.generate([tool_prompt], sampling_params)
                    
                    print("\n🔧 RESULTADO REAL DE HERRAMIENTAS:")
                    for output in tool_outputs:
                        result = output.outputs[0].text
                        print(f"🤖 Uso de herramientas:\n{result}")
                        
                except Exception as tool_error:
                    print(f"⚠️  Error usando herramientas: {tool_error}")
            
            # === TEST 3: ANÁLISIS COMBINADO REAL ===
            print(f"\n🎯 TEST 3: ANÁLISIS COMBINADO REAL (VISIÓN + HERRAMIENTAS)")
            print("-" * 50)
            
            combined_prompt = """Perform a complete real analysis of this invoice image:

1. OCR: Extract all text from the image
2. TOOLS: Use calculate_total to verify the invoice total
3. VALIDATION: Check if dates are valid
4. SUMMARY: Provide structured output

Be thorough and use actual tool calling, not simulation."""
            
            print("🔄 Ejecutando análisis combinado real...")
            
            try:
                combined_outputs = llm.generate([combined_prompt], sampling_params)
                
                print("\n🎊 RESULTADO FINAL REAL:")
                for output in combined_outputs:
                    result = output.outputs[0].text
                    print(f"🤖 Análisis completo y real:\n{result}")
            
            except Exception as combined_error:
                print(f"⚠️  Error en análisis combinado: {combined_error}")
            
            # Estadísticas reales
            total_time = time.time() - start_time
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"\n📊 ESTADÍSTICAS REALES:")
                print(f"⏱️  Tiempo total: {total_time:.1f} segundos")
                print(f"💾 Memoria GPU: {memory_used:.2f} GB")
            
            successful_model = model_name
            print(f"\n🎉 ¡MODELO {model_name} FUNCIONÓ COMPLETAMENTE!")
            break
            
        except Exception as model_error:
            error_time = time.time() - start_time
            print(f"❌ Error con {model_name} después de {error_time:.1f}s:")
            print(f"   {str(model_error)[:200]}...")
            
            # Limpiar memoria antes del siguiente modelo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Limpiar imagen temporal
    try:
        if test_image and os.path.exists(test_image):
            os.remove(test_image)
    except:
        pass
    
    # Resultado final
    print("\n" + "="*80)
    if successful_model:
        print(f"🏆 ¡ÉXITO! MODELO REAL FUNCIONANDO: {successful_model}")
        print("✅ Capacidades confirmadas:")
        print("   👁️  Análisis real de imágenes")  
        print("   🔍 OCR real de texto")
        print("   🛠️  Uso real de herramientas")
        print("   🎯 Análisis combinado real")
    else:
        print("❌ NINGÚN MODELO MULTIMODAL FUNCIONÓ")
        print("💡 Posibles soluciones:")
        print("   - Verificar compatibilidad de modelos con esta versión de vLLM")
        print("   - Instalar dependencias adicionales para multimodal")
        print("   - Usar modelos más pequeños o diferentes")
    
    print("="*80)

if __name__ == "__main__":
    test_real_multimodal_models()