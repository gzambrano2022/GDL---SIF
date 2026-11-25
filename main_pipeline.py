import argparse
import os
from modulos.m1_extraction import extraer_producto
from modulos.m2_depth import estimar_profundidad
from modulos.m3_mask import generar_mascara_espacial
from modulos.m4_generate import insertar_producto
from utils.memory import liberar_memoria

# ----------------------------------------------------------------------------
# Configuración del pipeline (edítala aquí; los valores de M4 están en m4_generate.py)
# ----------------------------------------------------------------------------
DIR_SALIDA = "data/outputs"                # salidas: DIR_SALIDA/<split>/<par>/...
PROMPT_MASCARA = "floor . ground . carpet" # prompt de detección de suelo (M3)
REUSAR_INTERMEDIOS = True                  # si ya existen producto sin fondo, profundidad y máscara de ese par, omite M1-M3
MODOS = ["base", "sin_controlnet", "controlnet", "comparar"]


def parse_args():
    parser = argparse.ArgumentParser(description="Inserción virtual de un producto en una habitación.")
    parser.add_argument("--producto", type=str, default="data/inputs/producto.jpg",
                        help="Ruta a la imagen del producto.")
    parser.add_argument("--habitacion", type=str, default="data/inputs/habitacion.jpg",
                        help="Ruta a la imagen de la habitación.")
    parser.add_argument("--modo", type=str, default="sin_controlnet", choices=MODOS,
                        help="base: compositing+sombra sin difusión | sin_controlnet | controlnet | comparar (ambos).")
    return parser.parse_args()


def carpeta_salida(habitacion_in: str) -> str:
    """data/inputs/valid/pair0041_room.jpg -> data/outputs/valid/pair0041"""
    ruta = os.path.normpath(habitacion_in)
    split = os.path.basename(os.path.dirname(ruta))
    stem = os.path.splitext(os.path.basename(ruta))[0]
    par = stem[:-5] if stem.endswith("_room") else stem
    return os.path.join(DIR_SALIDA, split, par)


def ejecutar_pipeline(producto_in: str, habitacion_in: str, modo: str = "sin_controlnet") -> str:
    """Ejecuta M1 -> M2 -> M3 -> M4 para un par habitación-producto y retorna la ruta del resultado."""
    if not os.path.exists(producto_in):
        raise FileNotFoundError(f"No se encontró la imagen del producto: {producto_in}")
    if not os.path.exists(habitacion_in):
        raise FileNotFoundError(f"No se encontró la imagen de la habitación: {habitacion_in}")

    carpeta = carpeta_salida(habitacion_in)
    os.makedirs(carpeta, exist_ok=True)
    out_producto = os.path.join(carpeta, "producto_sin_fondo.png")
    out_profundidad = os.path.join(carpeta, "profundidad.png")
    out_mascara = os.path.join(carpeta, "mascara.png")
    out_final = os.path.join(carpeta, f"resultado_{modo}.png")

    # --- M1: extracción del producto ---
    if REUSAR_INTERMEDIOS and os.path.exists(out_producto):
        print(f"[M1 omitido] Reutilizando {out_producto}")
    else:
        extraer_producto(input_path=producto_in, output_path=out_producto)
        liberar_memoria()

    # --- M2: profundidad (la usa M4 en los modos con ControlNet) ---
    if REUSAR_INTERMEDIOS and os.path.exists(out_profundidad):
        print(f"[M2 omitido] Reutilizando {out_profundidad}")
    else:
        estimar_profundidad(input_path=habitacion_in, output_path=out_profundidad)
        liberar_memoria()

    # --- M3: máscara de suelo ---
    if REUSAR_INTERMEDIOS and os.path.exists(out_mascara):
        print(f"[M3 omitido] Reutilizando {out_mascara}")
    else:
        generar_mascara_espacial(habitacion_path=habitacion_in, output_path=out_mascara, text_prompt=PROMPT_MASCARA)
        liberar_memoria()

    # --- M4: inserción (el resto de parámetros vive en la configuración global de m4_generate.py) ---
    resultado = insertar_producto(
        habitacion_path=habitacion_in,
        mascara_path=out_mascara,
        producto_path=out_producto,
        profundidad_path=out_profundidad,
        output_path=out_final,
        modo=modo,
    )
    liberar_memoria()
    return resultado


def main():
    args = parse_args()
    print(f"Iniciando pipeline (M1 -> M2 -> M3 -> M4) en modo '{args.modo}'...\n")
    resultado = ejecutar_pipeline(args.producto, args.habitacion, args.modo)
    print("\nPipeline terminado con éxito.")
    print(f"Resultado final guardado en: {resultado}")


if __name__ == "__main__":
    main()