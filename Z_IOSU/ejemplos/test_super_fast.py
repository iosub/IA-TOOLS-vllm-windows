#!/usr/bin/env python3
"""
Test súper rápido - Usando GPT-2 (que ya funciona) para simular OCR y herramientas
Carga en segundos, no minutos
"""

import os
import torch
import time
from vllm import LLM, SamplingParams

# Configurar variables de entorno para Windows
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["USE_LIBUV"] = "0"

def test_super_fast():
    """Test súper rápido usando GPT-2 que ya sabemos que funciona"""
    
    print("⚡ TEST SÚPER RÁPIDO - OCR Y HERRAMIENTAS SIMULADAS")
    print("="*60)
    
    start_time = time.time()
    
    try:
        print("⏳ Cargando GPT-2 (debería ser rápido)...")
        
        # Usar el modelo más pequeño y rápido
        llm = LLM(
            model="gpt2",
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.3,  # Muy conservador
            disable_custom_all_reduce=True,
            disable_log_stats=True,
        )
        
        load_time = time.time() - start_time
        print(f"✅ Modelo cargado en {load_time:.1f} segundos!")
        
        # Configuración rápida
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=60,  # Corto para velocidad
            top_p=0.9
        )
        
        # === SIMULACIÓN DE OCR ===
        print("\n🔍 SIMULANDO CAPACIDADES DE OCR:")
        ocr_prompts = [
            "This image contains the text: 'Invoice #12345'",
            "OCR extracted: 'Total Amount: $150.00'",
            "Document text found: 'Dear Customer'"
        ]
        
        gen_start = time.time()
        ocr_outputs = llm.generate(ocr_prompts, sampling_params)
        gen_time = time.time() - gen_start
        
        for i, output in enumerate(ocr_outputs, 1):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n📄 OCR {i}: {prompt}")
            print(f"🤖 Procesado: {generated_text[:80]}...")
        
        print(f"⚡ OCR generado en {gen_time:.1f} segundos")
        
        # === SIMULACIÓN DE HERRAMIENTAS ===
        print("\n🛠️  SIMULANDO HERRAMIENTAS:")
        tool_prompts = [
            "Using calculator tool: 123 + 456 =",
            "Search tool found: 'Python programming tutorial'",
            "File tool opened: 'document.pdf' contains"
        ]
        
        tool_start = time.time()
        tool_outputs = llm.generate(tool_prompts, sampling_params)
        tool_time = time.time() - tool_start
        
        for i, output in enumerate(tool_outputs, 1):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n🔧 Tool {i}: {prompt}")
            print(f"🤖 Resultado: {generated_text[:80]}...")
        
        print(f"⚡ Herramientas simuladas en {tool_time:.1f} segundos")
        
        # === SIMULACIÓN DE ANÁLISIS MULTIMODAL ===
        print("\n👁️  SIMULANDO ANÁLISIS DE IMAGEN:")
        vision_prompts = [
            "Analyzing image: I can see text that says",
            "Image contains: A document with the heading",
            "Visual analysis shows: The image has text"
        ]
        
        vision_start = time.time()
        vision_outputs = llm.generate(vision_prompts, sampling_params)
        vision_time = time.time() - vision_start
        
        for i, output in enumerate(vision_outputs, 1):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n👁️  Visión {i}: {prompt}")
            print(f"🤖 Análisis: {generated_text[:80]}...")
        
        print(f"⚡ Análisis visual en {vision_time:.1f} segundos")
        
        # Estadísticas finales
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS DEL TEST RÁPIDO:")
        print("="*60)
        print(f"⏱️  Tiempo total: {total_time:.1f} segundos")
        print(f"🚀 Tiempo de carga: {load_time:.1f} segundos")
        print(f"💬 Tiempo de generación: {gen_time + tool_time + vision_time:.1f} segundos")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 Memoria GPU usada: {memory_used:.2f} GB")
        
        print("\n🎯 PRÓXIMOS PASOS PARA OCR REAL:")
        print("1. Usar modelos especializados como TrOCR o PaddleOCR")
        print("2. Integrar con bibliotecas de OCR existentes")
        print("3. Usar modelos multimodales como Qwen2-VL (cuando termine de cargar)")
        print("4. Implementar pipeline de preprocesamiento de imágenes")
        
        print("\n✅ Test súper rápido completado!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_super_fast()