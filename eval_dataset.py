import argparse
import os
import re
import sys
from main_pipeline import ejecutar_pipeline

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def descubrir_pares(eval_dir: str):
    """
    Identifica pares de imágenes (producto, habitación) en el directorio de evaluación.
    Soporta dos esquemas organizativos:
      1. Subcarpetas por par: data/inputs/eval/<id_par>/{producto, habitacion}.*
      2. Archivos planos con nombres emparejados:
         data/inputs/eval/{id}_producto.* y data/inputs/eval/{id}_habitacion.*
         o producto_{id}.* y habitacion_{id}.*
    Retorna una lista de tuplas: (id_par, ruta_producto, ruta_habitacion)
    """
    pares = []
    if not os.path.exists(eval_dir):
        return pares

    entries = sorted(os.listdir(eval_dir))

    # Esquema 1: Subcarpetas por par
    subdirs = [e for e in entries if os.path.isdir(os.path.join(eval_dir, e))]
    for subdir in subdirs:
        s_path = os.path.join(eval_dir, subdir)
        files = [f for f in os.listdir(s_path) if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
        prod_file, hab_file = None, None

        for f in files:
            name_lower = f.lower()
            if any(k in name_lower for k in ["producto", "product", "prod", "item"]):
                prod_file = os.path.join(s_path, f)
            elif any(k in name_lower for k in ["habitacion", "room", "scene", "hab", "bg", "fondo"]):
                hab_file = os.path.join(s_path, f)

        # Si no se identificó por nombre pero hay exactamente 2 imágenes, se emparejan
        if not (prod_file and hab_file) and len(files) == 2:
            prod_file = os.path.join(s_path, files[0])
            hab_file = os.path.join(s_path, files[1])

        if prod_file and hab_file:
            pares.append((subdir, prod_file, hab_file))

    # Esquema 2: Archivos planos en eval_dir
    flat_files = [f for f in entries if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
    prod_map = {}
    hab_map = {}

    for f in flat_files:
        base, _ = os.path.splitext(f)
        # Patrones: <id>_producto o <id>-producto o <id>_item
        m_prod_suf = re.match(r"^(.*?)[_-]?(producto|product|prod|item)$", base, re.I)
        if m_prod_suf:
            key = m_prod_suf.group(1).strip("_-")
            prod_map[key] = os.path.join(eval_dir, f)
            continue

        # Patrones: producto_<id> o item_<id>
        m_prod_pref = re.match(r"^(producto|product|prod|item)[_-]?(.*?)$", base, re.I)
        if m_prod_pref:
            key = m_prod_pref.group(2).strip("_-")
            prod_map[key] = os.path.join(eval_dir, f)
            continue

        # Patrones: <id>_habitacion o <id>-habitacion
        m_hab_suf = re.match(r"^(.*?)[_-]?(habitacion|room|scene|hab|bg|fondo)$", base, re.I)
        if m_hab_suf:
            key = m_hab_suf.group(1).strip("_-")
            hab_map[key] = os.path.join(eval_dir, f)
            continue

        # Patrones: habitacion_<id> o habitacion-<id>
        m_hab_pref = re.match(r"^(habitacion|room|scene|hab|bg|fondo)[_-]?(.*?)$", base, re.I)
        if m_hab_pref:
            key = m_hab_pref.group(2).strip("_-")
            hab_map[key] = os.path.join(eval_dir, f)
            continue

    # Unir claves coincidentes
    for key in sorted(set(prod_map.keys()) & set(hab_map.keys())):
        pair_id = key if key else "item"
        # Evitar duplicados si ya vino de una subcarpeta
        if not any(p[0] == pair_id for p in pares):
            pares.append((pair_id, prod_map[key], hab_map[key]))

    return pares


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script de evaluación por lotes para el pipeline virtual-shopping."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/inputs/eval",
        help="Directorio con los pares de evaluación (default: data/inputs/eval).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/outputs/eval",
        help="Directorio donde se guardarán los resultados (default: data/outputs/eval).",
    )
    parser.add_argument(
        "--prompt_mascara",
        type=str,
        default="floor . ground . carpet",
        help="Prompt textual para detección de superficie en M3.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("INICIANDO EVALUACIÓN POR LOTES (eval_dataset.py)")
    print(f"Carpeta de entrada: {args.input_dir}")
    print(f"Carpeta de salida : {args.output_dir}")
    print("=" * 60)

    if not os.path.exists(args.input_dir):
        print(f"[ERROR] La carpeta de entrada no existe: {args.input_dir}")
        sys.exit(1)

    pares = descubrir_pares(args.input_dir)
    if not pares:
        print(f"[AVISO] No se encontraron pares de imágenes válidos en {args.input_dir}.")
        print("Asegúrate de estructurar las imágenes como:")
        print("  - Subcarpetas: data/inputs/eval/<id>/producto.jpg y habitacion.jpg")
        print("  - Archivos planos: <id>_producto.jpg y <id>_habitacion.jpg")
        sys.exit(0)

    print(f"Se encontraron {len(pares)} par(es) para procesar.\n")
    os.makedirs(args.output_dir, exist_ok=True)

    exitos = 0
    errores = []

    for idx, (pair_id, prod_path, hab_path) in enumerate(pares, start=1):
        print("-" * 50)
        print(f"[{idx}/{len(pares)}] Procesando par: '{pair_id}'")
        print(f"  • Producto  : {prod_path}")
        print(f"  • Habitación: {hab_path}")

        # Directorio específico para este par en el output
        pair_out_dir = os.path.join(args.output_dir, pair_id)
        os.makedirs(pair_out_dir, exist_ok=True)

        final_path = os.path.join(pair_out_dir, f"{pair_id}_resultado_final.png")
        prod_out = os.path.join(pair_out_dir, f"{pair_id}_producto_sin_fondo.png")
        depth_out = os.path.join(pair_out_dir, f"{pair_id}_profundidad.png")
        mask_out = os.path.join(pair_out_dir, f"{pair_id}_mascara.png")

        try:
            resultado = ejecutar_pipeline(
                producto_in=prod_path,
                habitacion_in=hab_path,
                output_final=final_path,
                output_producto=prod_out,
                output_profundidad=depth_out,
                output_mascara=mask_out,
                prompt_mascara=args.prompt_mascara,
            )
            print(f"[OK] Par '{pair_id}' completado exitosamente -> {resultado}")
            exitos += 1

        except Exception as e:
            print(f"[ERROR] Falló el procesamiento del par '{pair_id}': {e}")
            errores.append((pair_id, str(e)))

    print("\n" + "=" * 60)
    print("RESUMEN DE EVALUACIÓN")
    print(f"Total pares: {len(pares)}")
    print(f"Procesados con éxito: {exitos}")
    print(f"Fallidos: {len(errores)}")

    if errores:
        print("\nDetalle de fallos:")
        for pid, err in errores:
            print(f"  - [{pid}]: {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()
