import os
import gc
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def generar_mascara_espacial(
    habitacion_path: str,
    output_path: str,
    text_prompt: str = "floor . ground . carpet . rug",
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    margen_inferior: float = 0.50
) -> str:
    """
    Módulo 3: Enmascaramiento Espacial
    1. Intenta detectar el suelo con Grounding DINO + MobileSAM
    2. Si falla → activa Bounding Box de emergencia (plan de contingencia)
    """
    print("\n=== MÓDULO 3: Enmascaramiento Espacial ===")

    img = Image.open(habitacion_path).convert("RGB")
    w, h = img.size
    print(f"Imagen: {w}x{h}")

    mask = None
    metodo_usado = "bbox_emergencia"

    # -------------------------------------------------
    # INTENTO 1: Grounding DINO + MobileSAM
    # -------------------------------------------------
    try:
        print("Intentando detección automática (Grounding DINO + MobileSAM)...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Grounding DINO ---
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        inputs = processor(images=img, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[img.size[::-1]]
            )[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(h, w)]
            )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        # Liberar Grounding DINO
        del model, processor, inputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(boxes) == 0:
            raise ValueError("No se detectó ninguna región de suelo")

        print(f"Detectadas {len(boxes)} posibles regiones de suelo")

        # --- MobileSAM ---
        from mobile_sam import sam_model_registry, SamPredictor

        sam_checkpoint = "weights/mobile_sam.pt"
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"No se encontró {sam_checkpoint}. Descárgalo primero.")

        mobile_sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()

        predictor = SamPredictor(mobile_sam)
        predictor.set_image(np.array(img))

        # Usamos la caja con mayor score
        best_idx = np.argmax(scores)
        best_box = boxes[best_idx]
        print(f"Mejor caja (score={scores[best_idx]:.3f}): {best_box}")

        masks, _, _ = predictor.predict(
            box=best_box,
            multimask_output=False
        )

        mask_np = masks[0].astype(np.uint8) * 255
        mask = Image.fromarray(mask_np, mode="L")
        metodo_usado = "grounding_dino + mobile_sam"

        # Liberar MobileSAM
        del mobile_sam, predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Detección automática exitosa")

    except Exception as e:
        print(f"Falló la detección automática: {e}")
        print("Activando plan de contingencia → Bounding Box")

        # -------------------------------------------------
        # PLAN DE CONTINGENCIA: Bounding Box
        # -------------------------------------------------
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        x1 = int(w * 0.15)
        x2 = int(w * 0.85)
        y1 = int(h * margen_inferior)
        y2 = int(h * 0.98)

        draw.rectangle([x1, y1, x2, y2], fill=255)
        print(f"Bounding Box de emergencia: ({x1}, {y1}) → ({x2}, {y2})")

    # Guardar máscara
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask.save(output_path)
    print(f"Máscara guardada en: {output_path}")
    print(f"Método usado: {metodo_usado}")

    return output_path