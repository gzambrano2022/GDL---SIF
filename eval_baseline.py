import argparse
import csv
import glob
import os
import time
import torch
from PIL import Image

# Importar las funciones de los módulos del proyecto (alineado con main_pipeline.py)
from modulos.m1_extraction import extraer_producto
from modulos.m2_depth import estimar_profundidad
from modulos.m3_mask import generar_mascara_espacial
from modulos.m4_generate import insertar_producto
from utils.memory import liberar_memoria


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación del pipeline de inserción virtual de productos (ControlNet Depth + Inpainting)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/inputs/valid",
        help="Directorio con imágenes de validación (por defecto: data/inputs/valid).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/outputs/eval_results",
        help="Directorio de salida para resultados y métricas (por defecto: data/outputs/eval_results).",
    )
    parser.add_argument(
        "--prompt_mascara",
        type=str,
        default="floor . ground . carpet",
        help="Prompt textual para detección de superficie en M3 (por defecto: floor . ground . carpet).",
    )
    return parser.parse_args()


def evaluar_baseline(
    input_dir="data/inputs/valid",
    output_dir="data/outputs/eval_results",
    prompt_mascara="floor . ground . carpet"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar todos los pares en la carpeta de validación
    items = sorted(glob.glob(os.path.join(input_dir, "*_item.jpg")))
    rooms = sorted(glob.glob(os.path.join(input_dir, "*_room.jpg")))
    
    if not items or len(items) != len(rooms):
        print("Error: No se encontraron pares completos en el input_dir.")
        return

    csv_path = os.path.join(output_dir, "metricas_baseline.csv")
    
    print(f"Iniciando evaluación de {len(items)} pares...")
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Par", "Tiempo_Total_Seg", "VRAM_Peak_MB", "Estado"])

        for i, (item_path, room_path) in enumerate(zip(items, rooms)):
            pair_name = os.path.basename(item_path).split('_')[0]
            print(f"\n--- Procesando {pair_name} ({i+1}/{len(items)}) ---")
            
            # Reiniciar métricas de GPU antes de cada iteración
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            # Rutas temporales y finales
            tmp_item = os.path.join(output_dir, f"{pair_name}_nobg.png")
            tmp_depth = os.path.join(output_dir, f"{pair_name}_depth.png")
            tmp_mask = os.path.join(output_dir, f"{pair_name}_mask.png")
            final_out = os.path.join(output_dir, f"{pair_name}_final.png")
            
            try:
                # --- Módulo 1: Extracción del producto ---
                producto_extraido = extraer_producto(
                    input_path=item_path,
                    output_path=tmp_item
                )
                liberar_memoria()

                # --- Módulo 2: Estimación de profundidad ---
                profundidad_generada = estimar_profundidad(
                    input_path=room_path,
                    output_path=tmp_depth
                )
                liberar_memoria()
                
                # --- Módulo 3: Enmascaramiento espacial ---
                mascara_generada = generar_mascara_espacial(
                    habitacion_path=room_path,
                    output_path=tmp_mask,
                    text_prompt=prompt_mascara
                )
                liberar_memoria()
                
                # --- Módulo 4: Inserción y generación final con ControlNet Depth ---
                insertar_producto(
                    habitacion_path=room_path,
                    mascara_path=mascara_generada,
                    producto_path=producto_extraido,
                    profundidad_path=profundidad_generada,
                    output_path=final_out
                )
                liberar_memoria()
                
                estado = "Exito"
            except Exception as e:
                print(f"Error en {pair_name}: {e}")
                estado = "Fallo"
            
            # Capturar métricas finales de la iteración
            end_time = time.time()
            total_time = round(end_time - start_time, 2)
            peak_vram = (
                round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                if torch.cuda.is_available()
                else 0.0
            )
            
            writer.writerow([pair_name, total_time, peak_vram, estado])
            print(f"Completado en {total_time}s | Peak VRAM: {peak_vram} MB")

    print(f"\nEvaluación finalizada. Métricas guardadas en {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluar_baseline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt_mascara=args.prompt_mascara
    )