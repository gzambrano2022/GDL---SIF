import os
import gc
import torch
from PIL import Image
from transformers import pipeline

def estimar_profundidad(input_path: str, output_path: str) -> str:
    print("\n=== MÓDULO 2: Estimación de Profundidad (Depth Anything Small) ===")

    device = 0 if torch.cuda.is_available() else -1

    # Pipeline oficial de Hugging Face (muy estable)
    pipe = pipeline(
        task="depth-estimation",
        model="LiheYoung/depth-anything-small-hf",
        device=device
    )

    # Cargar imagen de la habitación
    image = Image.open(input_path).convert("RGB")
    print(f"Imagen cargada: {image.size}")

    # Inferencia
    result = pipe(image)
    depth_image = result["depth"]   # ya viene como PIL Image

    # Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    depth_image.save(output_path)
    print(f"Mapa de profundidad guardado en: {output_path}")

    # Liberar memoria
    del pipe, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("→ VRAM liberada")

    return output_path