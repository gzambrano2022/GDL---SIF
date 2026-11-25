from modulos.m4_generate import barrido_parametros

resultados = barrido_parametros(
    habitacion_path="/home/gznov/virtual-shopping/data/inputs/habitacion.jpg",
    mascara_path="data/outputs/mascara.png",
    producto_path="data/outputs/producto_sin_fondo.png",
    output_dir="data/outputs/barrido_sd",

    con_controlnet=False,

    strengths=(0.25, 0.35, 0.45),
    guidance_scales=(5.0, 6.5),

    num_steps=35,
    seed=42,

    # Configuración de la escena
    iluminacion="auto",
    ancho_rel=0.46,
    alto_max_rel=0.40,
    base_rel=0.76,

    anillo_px=6,
    borde_editable_px=1,
)

for ruta in resultados:
    print(ruta)