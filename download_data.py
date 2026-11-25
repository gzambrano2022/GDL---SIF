import argparse
import os
import random
import glob
from PIL import Image

def descargar_y_preparar_dataset(output_dir: str, max_pairs: int):
    splits = ["train", "valid", "test"]
    directorios = {split: os.path.join(output_dir, split) for split in splits}
    for dir_path in directorios.values():
        os.makedirs(dir_path, exist_ok=True)

    print("📥 Descargando dataset de habitaciones (Hugging Face)...")
    try:
        from datasets import load_dataset
        rooms_ds = load_dataset("wunderkinds07/empty-room", split="train")
    except Exception as e:
        raise RuntimeError(f"Error cargando Hugging Face: {e}")

    print("📥 Descargando dataset de sofás (Kaggle)...")
    try:
        import kaggle
        temp_kaggle_dir = "temp_sofas"
        os.makedirs(temp_kaggle_dir, exist_ok=True)
        kaggle.api.dataset_download_files("sunnykusawa/sofas", path=temp_kaggle_dir, unzip=True)
        
        sofa_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            sofa_files.extend(glob.glob(os.path.join(temp_kaggle_dir, '**', ext), recursive=True))
            
        if not sofa_files:
            raise ValueError("Kaggle no descargó imágenes. Verifica tu ~/.kaggle/access_token o la variable KAGGLE_API_TOKEN.")
    except Exception as e:
        raise RuntimeError(f"Error descargando desde Kaggle: {e}")

    total_disponible = min(len(rooms_ds), len(sofa_files))
    if max_pairs and max_pairs < total_disponible:
        total_disponible = max_pairs

    print(f"\n📊 Total de pares viables a generar: {total_disponible}")
    
    indices = list(range(total_disponible))
    random.shuffle(indices)

    train_cut = int(0.7 * total_disponible)
    valid_cut = int(0.9 * total_disponible)

    print(f"Distribuido en: Train ({train_cut}), Valid ({valid_cut - train_cut}), Test ({total_disponible - valid_cut})\n")

    exitosos = 0
    for i, idx in enumerate(indices):
        if i < train_cut:
            split_actual = "train"
        elif i < valid_cut:
            split_actual = "valid"
        else:
            split_actual = "test"
            
        target_dir = directorios[split_actual]
        pair_id = f"pair{i+1:04d}"
        
        item_path = os.path.join(target_dir, f"{pair_id}_item.jpg")
        room_path = os.path.join(target_dir, f"{pair_id}_room.jpg")

        try:
            row = rooms_ds[idx]
            
            # Autodetección priorizando Before_Image (sensible a mayúsculas)
            img_col = None
            for col_name in ['Before_Image', 'before_image', 'image', 'img']:
                if col_name in row:
                    img_col = col_name
                    break
            
            if not img_col:
                raise KeyError(f"No se encontró columna de imagen. Columnas disponibles: {list(row.keys())}")
            
            room_img = row[img_col]
            if room_img.mode != "RGB":
                room_img = room_img.convert("RGB")
            room_img.save(room_path, format="JPEG", quality=90)

            sofa_img = Image.open(sofa_files[idx]).convert("RGB")
            sofa_img.save(item_path, format="JPEG", quality=90)

            exitosos += 1
            if i % 10 == 0 or i == total_disponible - 1:
                print(f"[{i+1}/{total_disponible}] Pares procesados correctamente...")

        except Exception as e:
            print(f"  ✗ Error procesando el par {pair_id}: {e}")

    print("=" * 50)
    print(f"✅ Construcción finalizada: {exitosos}/{total_disponible} pares distribuidos en '{output_dir}'.")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/inputs")
    parser.add_argument("--max_pairs", type=int, default=None)
    args = parser.parse_args()
    descargar_y_preparar_dataset(args.output_dir, args.max_pairs)