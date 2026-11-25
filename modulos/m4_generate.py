"""
Módulo 4: inserción local del producto.

Idea central: la máscara del suelo (M3) SOLO sirve para decidir dónde apoyar el producto
y dónde puede caer la sombra. Lo que se le pasa al modelo de difusión (mask_image) es una
máscara de integración pequeña alrededor del producto, que excluye el producto.

Flujo:
  1. recortar_producto_transparente()  -> quita el borde transparente que deja M1
  2. calcular_colocacion()             -> apoya la base del producto en el suelo (no lo centra)
  3. estimar_iluminacion()             -> de dónde viene la luz y con qué confianza
     crear_sombra_iluminada()          -> sombra proyectada por filas con falso 3D (profundidad de M2) + contacto fijo;
                                          con confianza de luz baja: cenital, solo contacto
     (crear_sombra_base() queda como modo manual)
  3b. armonizar_producto()             -> nitidez, grano, contraste, color, brillo y bordes del sofá según el cuarto
  4. crear_mascara_integracion()       -> anillo + sombra, SIN el interior del producto
  5. generar()                         -> inpainting sobre un recorte (ROI); el producto se re-pega intacto
  6. modos: "base" -> "sin_controlnet" -> "controlnet" / "comparar"
"""
import os
import json
import itertools

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
)

from utils.memory import liberar_memoria

# El repo "runwayml/stable-diffusion-inpainting" fue retirado del Hub; este es el espejo oficial.
INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
CONTROLNET_MODEL = "lllyasviel/control_v11f1p_sd15_depth"
WORK_SIZE = 512

# Prompt de INTEGRACIÓN (el producto no se toca; solo se pide continuidad, sombra y luz coherente)
PROMPT_DEFAULT = (
    "same room, floor and wall surfaces continuing naturally around the sofa, "
    "soft contact shadow, ambient occlusion, matching room lighting, color temperature "
    "and perspective, seamless integration, photorealistic photo"
)
NEGATIVE_DEFAULT = (
    "redesign, new sofa, different sofa, changed sofa, extra furniture, duplicated object, "
    "floating object, cutout edges, white halo, blurry, distorted, low quality, text, watermark"
)


# ============================================================================
# CONFIGURACIÓN GLOBAL
# Edita estos valores aquí: main_pipeline.py ya no los pasa, así que mandan estos.
# Las funciones los usan como valores por defecto; un argumento explícito los sobrescribe.
# ============================================================================

# --- Flujo ------------------------------------------------------------------
MODO_DEFAULT = "sin_controlnet"   # "base" | "sin_controlnet" | "controlnet" | "comparar"
PREFERIDO = "sin_controlnet"      # cuál queda como resultado final en modo "comparar"
RECALCULAR_PROFUNDIDAD = False    # False: usa el mapa de M2 (cuarto vacío) | True: lo recalcula con el sofá puesto

# --- Colocación -------------------------------------------------------------
ANCHO_REL = None        # ancho del sofá = ANCHO_REL x ancho de la zona de suelo. None = escala automática (perspectiva, ver abajo)
ALTO_MAX_REL = 0.58     # el alto del sofá no supera ALTO_MAX_REL x alto de la imagen
POS_X_REL = 0.50        # 0 = borde izquierdo del suelo, 0.5 = centro, 1 = borde derecho
BASE_REL = 0.84         # línea de apoyo: 0 = parte alta del suelo (lejos), 1 = parte baja (cerca de la cámara)
ANCHO_OBJETIVO = None   # ancho fijo en px (ignora ANCHO_REL)
OFFSET = (0, 0)         # ajuste fino (dx, dy) en px
# Escala automática (ANCHO_REL = None y ANCHO_OBJETIVO = None): tamaño real del sofá + perspectiva de la foto
ALTURA_PRODUCTO_M = 0.85         # alto típico del sofá en metros (respaldo incluido). Sube/baja para agrandar/achicar todo
ANCHO_PRODUCTO_M_MIN = 0.80      # el ancho real que se deduce de la proporción del recorte se limita a este rango (m)
ANCHO_PRODUCTO_M_MAX = 2.40
ALTURA_CAMARA_M = 1.40           # altura típica de la cámara sobre el suelo en fotos de interiores (m)
HORIZONTE_REL_PRIOR = 0.45       # fila del horizonte (x alto) cuando no se puede medir en la máscara de suelo
ESCALA_AUTO_FACTOR = 1.0         # multiplicador global del ancho automático (1.15 = 15 % más grande)
ESCALA_CABE_EN_SUELO = 0.92      # el sofá no supera este factor x el ancho del suelo medido en su línea de apoyo
ESCALA_ANCHO_MIN_REL = 0.12      # límites del ancho automático, como fracción del ancho de la foto
ESCALA_ANCHO_MAX_REL = 0.80

# --- Iluminación y sombra ---------------------------------------------------
SOMBRA = True
ILUMINACION = "auto"             # "auto": estima la luz y deriva la sombra | "manual": usa los valores de más abajo
ESCALA_SOMBRA = 1.4         # antes 1.0     # multiplica la intensidad de la sombra PROYECTADA en modo auto (0.7 más suave, 1.3 más marcada)
FACTOR_PERSPECTIVA = 0.35        # compresión de la profundidad del suelo en la imagen
AZIMUT_LUZ_DEG = None            # fuerza la dirección de la luz (0=derecha, 90=fondo, 180=izquierda, 270=cámara)
ELEVACION_LUZ_DEG = 45.0  # antes NONE        # fuerza la elevación (alta = sombra corta, baja = larga)
PROMPT_CON_LUZ = True            # añade la dirección de la luz al prompt si la confianza >= 0.5
SOMBRA_PROYECTADA_RANGO = (0.18, 0.45) # antes 0.06, 0.32   # intensidad de la sombra proyectada: (luz difusa, luz dura)
ELEVACION_RANGO_DEG = (35.0, 80.0)       # elevación estimada: (luz dura, luz difusa)
SOMBRA_LONGITUD_MAX_REL = 0.60           # largo máximo de la sombra = este factor x ancho del sofá
SOMBRA_HUELLA_REL = 0.16                 # profundidad de la huella del sofá (x su alto)
# Sombra de contacto (patas/base): independiente de la dirección y la fuerza de la luz
SOMBRA_CONTACTO_FIJA = 0.85              # oscuridad en el centro de la sombra de contacto (0-1)
SOMBRA_CONTACTO_RADIO_MIN = 2            # radio mínimo alrededor de los puntos más bajos (px)
SOMBRA_CONTACTO_RADIO_MAX = 5            # radio máximo (px)
SOMBRA_CONTACTO_RADIO_REL = 0.008        # radio = esto x ancho del sofá, entre el mínimo y el máximo
# Falso 3D con el mapa de profundidad de M2
PROFUNDIDAD_CLARO_ES_CERCA = True        # Depth Anything: más claro = más cerca (invierte si tu mapa es al revés)
SOMBRA_PROF_INFLUENCIA = 1.2   # antes 0.60         # las filas más lejanas proyectan hasta (1 + esto) veces más lejos
SOMBRA_PROF_REF = 0.18         # antes 0.25          # diferencia de profundidad (0-1) que cuenta como "totalmente más lejos"
# Estimación de luz poco fiable
CONFIANZA_MIN_LUZ = 0.40                 # por debajo: luz cenital suave, sin sombra proyectada, solo contacto
SOMBRA_CENITAL_AO = 0.0                  # 0 = solo contacto; >0 añade una oclusión ambiental suave bajo el sofá
PISTA_SUELO_PESO = 0.70                  # peso de la pista del suelo (gradiente de luminancia) frente a la de brillos
PISTA_SUELO_BORDE_REL = 0.20             # analiza solo el 20 % exterior de la máscara de suelo (None = todo el suelo)
# Solo ILUMINACION = "manual":
INTENSIDAD_CONTACTO = 0.21
INTENSIDAD_PROYECTADA = 0.06
DIRECCION_LUZ = 0.0

# --- Máscara de integración (lo único que el modelo de difusión puede editar) ----------
ANILLO_PX = None                 # ancho del anillo en px de la imagen; None = automático
ANILLO_AUTO_REL = 0.03           # automático: ANILLO_AUTO_REL x lado mayor del sofá ...
ANILLO_AUTO_MIN = 8              # ... con este mínimo en px
BORDE_EDITABLE_PX = 3            # banda del borde del sofá que el modelo puede retocar (0 = sofá totalmente protegido)
INCLUIR_SOMBRA_EN_MASCARA = True # la zona de sombra también es editable
MARGEN_ROI = 0.35                # contexto alrededor de la zona editable al recortar para el modelo

# --- Armonización del sofá con el cuarto ------------------------------------
ARMONIZAR = True
ARMONIZAR_FUERZA = 1.0           # 0 = sin cambios, 1 = completa, >1 más agresiva
ARMONIZAR_PESOS = dict(nitidez=1.0, ruido=1.0, contraste=1.0, color=1.0, brillo=1.0, bordes=1.0)  # 0 apaga un paso
ARMONIZAR_TOPES = dict(
    nitidez_tolerancia=1.10,     # el sofá puede tener hasta 10 % más detalle fino que el cuarto
    desenfoque_max_px=3.0,       # desenfoque máximo aplicable
    ruido_max=3.5,               # grano máximo añadido (escala 0-255)
    contraste_min=0.80,          # reduce el contraste hasta -20 %
    nivel_negro_max=0.06,        # sube el nivel de negro hasta este valor
    saturacion_min=0.85,         # desatura hasta -15 %
    tinte_max=0.08,              # balance de blancos hacia la luz ambiente, hasta ±8 %
    brillo_local=0.15,           # ajuste por luminosidad local del suelo, ±15 %
    gradiente_luz=0.12,          # amplitud máxima del gradiente de luz sobre el sofá
    luz_envolvente=0.35,         # mezcla de color del fondo en el borde del sofá
)

# --- Difusión (ajuste fino: al final) ---------------------------------------
STRENGTH = 0.30
GUIDANCE_SCALE = 7.0
NUM_STEPS = 25
CONTROLNET_SCALE = 0.55
SEED = 42


# ----------------------------------------------------------------------------
# Utilidades de imagen
# ----------------------------------------------------------------------------
def letterbox_image(image: Image.Image, target_size: int = 512, fill_value=0):
    """(Se mantiene por compatibilidad con otros módulos; el Módulo 4 ya no la usa.)"""
    w, h = image.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resample = Image.Resampling.NEAREST if image.mode == "L" else Image.Resampling.LANCZOS
    resized = image.resize((new_w, new_h), resample)
    pad_color = fill_value if image.mode != "RGB" else (fill_value,) * 3
    padded = Image.new(image.mode, (target_size, target_size), pad_color)
    pad_left, pad_top = (target_size - new_w) // 2, (target_size - new_h) // 2
    padded.paste(resized, (pad_left, pad_top))
    return padded, (pad_left, pad_top, new_w, new_h)


def unletterbox_image(padded_image: Image.Image, padding_info: tuple, orig_size: tuple) -> Image.Image:
    """(Se mantiene por compatibilidad.)"""
    pad_left, pad_top, new_w, new_h = padding_info
    cropped = padded_image.crop((pad_left, pad_top, pad_left + new_w, pad_top + new_h))
    return cropped.resize(orig_size, Image.Resampling.LANCZOS)


def _blur(mask: Image.Image, sigma: float) -> Image.Image:
    # float() explícito: Pillow >= 12 falla si el radio es un escalar de NumPy (np.float64, etc.)
    sigma = float(sigma)
    return mask.filter(ImageFilter.GaussianBlur(sigma)) if sigma > 0 else mask


def _dilatar(mask: Image.Image, px: float) -> Image.Image:
    """Dilatación aproximada (px píxeles): desenfoque gaussiano + umbral al 2%."""
    if px <= 0:
        return mask
    return _blur(mask, px / 2.05).point(lambda v: 255 if v > 5 else 0)


def _erosionar(mask: Image.Image, px: float) -> Image.Image:
    """Erosión aproximada (px píxeles): desenfoque gaussiano + umbral al 98%."""
    if px <= 0:
        return mask
    return _blur(mask, px / 2.05).point(lambda v: 255 if v > 250 else 0)


# ----------------------------------------------------------------------------
# Punto 2: recortar el producto transparente
# ----------------------------------------------------------------------------
def recortar_producto_transparente(
    producto: Image.Image,
    umbral_alpha: int = 10,
    erosion_alpha_px: int = 1,
    padding: int = 0,
) -> Image.Image:
    """
    M1 devuelve el producto con el tamaño de la foto original: el sofá ocupa una parte y el
    resto es transparente. Si se escala/posiciona ese lienzo completo, la base real del sofá
    no queda donde se espera. Esta función:
      - pone alpha=0 a los valores residuales (< umbral_alpha) que deja RMBG,
      - erosiona el alpha 'erosion_alpha_px' px para quitar el halo del fondo original,
      - recorta al bounding box de los píxeles opacos (+ padding).
    """
    producto = producto.convert("RGBA")
    alpha = producto.getchannel("A").point(lambda v: 0 if v < umbral_alpha else v)
    for _ in range(max(erosion_alpha_px, 0)):
        alpha = alpha.filter(ImageFilter.MinFilter(3))
    producto.putalpha(alpha)

    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("El producto no tiene píxeles opacos (canal alfa vacío)")
    l, t, r, b = bbox
    l, t = max(l - padding, 0), max(t - padding, 0)
    r, b = min(r + padding, producto.width), min(b + padding, producto.height)
    recorte = producto.crop((l, t, r, b))
    print(f"Producto recortado: {producto.size} -> {recorte.size}")

    if recorte.getchannel("A").getextrema()[0] == 255:
        print("AVISO: el producto no tiene transparencia; se preservará todo el rectángulo.")
    return recorte


# ----------------------------------------------------------------------------
# Punto 3: colocación apoyada (no centrada)
# ----------------------------------------------------------------------------
def calcular_colocacion(
    floor_bbox,
    img_size,
    producto: Image.Image,
    ancho_rel: float = ANCHO_REL,
    alto_max_rel: float = ALTO_MAX_REL,
    pos_x_rel: float = POS_X_REL,
    base_rel: float = BASE_REL,
    ancho_objetivo: int = ANCHO_OBJETIVO,
    offset=OFFSET,
):
    """
    Apoya la BASE del producto sobre una línea del suelo; no centra el producto en la máscara.
      - ancho: 'ancho_rel' x ancho de la zona de suelo (o 'ancho_objetivo' en px).
      - tope de alto: 'alto_max_rel' x alto de la imagen, y nunca sale por arriba de la foto.
      - pos_x_rel: 0 = borde izquierdo del suelo, 0.5 = centro, 1 = borde derecho.
      - base_rel: altura de la línea de apoyo dentro del suelo (0 = parte alta/lejos,
        1 = parte baja/cerca de la cámara). Más bajo = sofá más lejano.
      - offset: ajuste fino (dx, dy) en px.
    Retorna (producto_redimensionado, (paste_x, paste_y)).
    """
    W, H = img_size
    x1, y1, x2, y2 = floor_bbox
    box_w, box_h = x2 - x1 + 1, y2 - y1 + 1
    pw, ph = producto.size

    base_y = int(round(y1 + base_rel * box_h)) + offset[1]
    x_centro = int(round(x1 + pos_x_rel * box_w)) + offset[0]

    s = ancho_objetivo / pw if ancho_objetivo else (ancho_rel * box_w) / pw
    s = min(s, (alto_max_rel * H) / ph, max(base_y, 1) / ph)
    new_w, new_h = max(int(round(pw * s)), 1), max(int(round(ph * s)), 1)
    producto = producto.resize((new_w, new_h), Image.Resampling.LANCZOS)

    paste_x = x_centro - new_w // 2
    paste_y = base_y - new_h
    return producto, (paste_x, paste_y)


def estimar_horizonte(mascara_suelo: Image.Image, img_size) -> dict:
    """
    Fila de la línea del horizonte (donde convergen las paredes laterales), estimada con la propia
    máscara de suelo: en un cuarto visto de frente, el ancho del suelo crece linealmente hacia
    abajo, ancho(y) = k*(y - y_horizonte). Se ajusta esa recta con las filas cuyo borde izquierdo y
    derecho no están cortados por el marco de la foto y se despeja y_horizonte.
    Si el ajuste no es fiable (máscara rectangular de emergencia, vista en ángulo, poco suelo)
    se mezcla con, o cae a, HORIZONTE_REL_PRIOR x alto de la imagen.
    Retorna dict(y_h, confianza, metodo, ...).
    """
    W, H = img_size
    prior = float(np.clip(HORIZONTE_REL_PRIOR, 0.05, 0.90)) * H
    res = dict(y_h=prior, y_h_ajuste=None, confianza=0.0, metodo="prior", filas_usadas=0, r2=None)
    suelo = np.asarray(mascara_suelo.convert("L").resize((W, H), Image.Resampling.NEAREST)) > 128

    filas = np.flatnonzero(suelo.sum(axis=1) > 0.05 * W)
    if len(filas) < 30:
        return res
    sub = suelo[filas]
    xl = np.argmax(sub, axis=1)
    xr = (W - 1) - np.argmax(sub[:, ::-1], axis=1)
    ancho = (xr - xl + 1).astype(np.float64)
    ok = (xl > 3) & (xr < W - 4) & (ancho > 0.10 * W)  # bordes que son de la habitación, no del marco
    y, w = filas[ok].astype(np.float64), ancho[ok]
    if len(y) < 25 or (y.max() - y.min()) < 0.08 * H:
        return res

    for _ in range(4):  # ajuste robusto: descarta filas atípicas (muebles, huecos de la máscara)
        k, c = np.polyfit(y, w, 1)
        r = w - (k * y + c)
        mad = np.median(np.abs(r - np.median(r))) + 1e-6
        keep = np.abs(r) <= max(3 * 1.4826 * mad, 2.0)
        if keep.sum() < 20 or keep.all():
            break
        y, w = y[keep], w[keep]
    k, c = np.polyfit(y, w, 1)
    r2 = float(np.clip(1 - ((w - (k * y + c)) ** 2).sum() / (((w - w.mean()) ** 2).sum() + 1e-9), 0, 1))
    res.update(filas_usadas=int(len(y)), r2=round(r2, 3))
    if k < 0.05:  # el suelo no se ensancha hacia abajo: no hay convergencia medible
        return res

    y_ajuste = float(-c / k)
    conf = r2 * float(np.clip(((y.max() - y.min()) / H) / 0.25, 0, 1))
    if not (0.05 * H <= y_ajuste <= 0.75 * H) or y_ajuste > filas.min() + 0.02 * H:
        conf *= 0.3  # incompatible con la geometría (el horizonte debe quedar sobre el suelo)
    res.update(y_h_ajuste=round(y_ajuste, 1), confianza=round(conf, 3))
    if conf >= 0.35:
        res.update(y_h=float(np.clip(conf * y_ajuste + (1 - conf) * prior, 0.05 * H, 0.75 * H)),
                   metodo="ajuste del suelo")
    return res


def estimar_escala_perspectiva(
    mascara_suelo: Image.Image,
    floor_bbox,
    img_size,
    producto: Image.Image,
    base_rel: float = BASE_REL,
    offset=OFFSET,
):
    """
    Ancho del sofá en píxeles a partir de su tamaño REAL y la perspectiva de la foto.
    Para una cámara a altura h_cam sobre el suelo, un objeto apoyado en la fila y se ve a
        px_por_metro = (y - y_horizonte) / h_cam
    (Hoiem et al., "Putting Objects in Perspective"). Con eso:
        alto_px  = ALTURA_PRODUCTO_M x px_por_metro
        ancho_px = (ALTURA_PRODUCTO_M x proporción del recorte) x px_por_metro   [ancho limitado a un rango real]
    Así el sofá se achica al alejarse y crece al acercarse, sin depender de qué tan grande salió la
    máscara de suelo. Si el suelo en la línea de apoyo está medido (no cortado por el marco), el sofá
    tampoco puede ser más ancho que ESCALA_CABE_EN_SUELO x ese ancho.
    Retorna (ancho_px, info).
    """
    W, H = img_size
    x1, y1, x2, y2 = floor_bbox
    base_y = float(np.clip(round(y1 + base_rel * (y2 - y1 + 1)) + offset[1], 0, H - 1))

    hor = estimar_horizonte(mascara_suelo, img_size)
    dist = max(base_y - hor["y_h"], 0.08 * H)           # evita escalas absurdas si la base queda sobre el horizonte
    px_por_m = dist / ALTURA_CAMARA_M

    pw, ph = producto.size
    ancho_m = float(np.clip(ALTURA_PRODUCTO_M * pw / max(ph, 1), ANCHO_PRODUCTO_M_MIN, ANCHO_PRODUCTO_M_MAX))
    ancho_px = ancho_m * px_por_m * ESCALA_AUTO_FACTOR

    limite_suelo = None
    suelo = np.asarray(mascara_suelo.convert("L").resize((W, H), Image.Resampling.NEAREST)) > 128
    xs = np.flatnonzero(suelo[int(base_y)])
    if len(xs) >= 8 and xs[0] > 3 and xs[-1] < W - 4:
        limite_suelo = ESCALA_CABE_EN_SUELO * float(xs[-1] - xs[0] + 1)
        ancho_px = min(ancho_px, limite_suelo)
    ancho_px = float(np.clip(ancho_px, ESCALA_ANCHO_MIN_REL * W, ESCALA_ANCHO_MAX_REL * W))

    info = dict(
        ancho_px=round(ancho_px, 1), ancho_m=round(ancho_m, 2), px_por_metro=round(px_por_m, 1),
        base_y=int(base_y), horizonte_y=round(float(hor["y_h"]), 1), horizonte_rel=round(float(hor["y_h"]) / H, 3),
        horizonte_metodo=hor["metodo"], horizonte_confianza=hor["confianza"],
        horizonte_ajuste_y=hor["y_h_ajuste"], limite_por_suelo=None if limite_suelo is None else round(limite_suelo, 1),
    )
    return ancho_px, info


def guardar_diagnostico_escala(habitacion: Image.Image, info: dict, caja, ruta: str):
    """Habitación con el horizonte estimado (cian), la línea de apoyo (amarillo) y el sofá previsto (verde)."""
    img = habitacion.copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    ancho = max(2, W // 400)
    d.line((0, info["horizonte_y"], W, info["horizonte_y"]), fill=(0, 220, 255), width=ancho)
    d.line((0, info["base_y"], W, info["base_y"]), fill=(255, 220, 0), width=ancho)
    px, py, nw, nh = caja
    d.rectangle((px, py, px + nw, py + nh), outline=(0, 255, 0), width=ancho)
    d.rectangle((0, 0, W, 18), fill=(0, 0, 0))
    d.text((6, 4), (f"horizonte {info['horizonte_rel']:.2f}H [{info['horizonte_metodo']}, conf {info['horizonte_confianza']:.2f}] | "
                    f"{info['px_por_metro']:.0f} px/m | sofa {info['ancho_m']:.2f} m = {info['ancho_px']:.0f} px"), fill=(255, 255, 255))
    img.save(ruta)


def verificar_apoyo(mascara_suelo: Image.Image, prod_alpha: Image.Image, pos) -> float:
    """Fracción de la base del producto que realmente descansa sobre la máscara de suelo."""
    px, py = pos
    nw, nh = prod_alpha.size
    banda = max(int(nh * 0.04), 2)
    base_alpha = np.array(prod_alpha)[nh - banda:, :] > 127
    W, H = mascara_suelo.size
    ys, xs = np.nonzero(base_alpha)
    ys, xs = ys + (py + nh - banda), xs + px
    dentro = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if dentro.sum() == 0:
        return 0.0
    suelo = np.array(mascara_suelo) > 128
    return float(suelo[ys[dentro], xs[dentro]].mean())


# ----------------------------------------------------------------------------
# Punto 4: sombra base antes del inpainting
# ----------------------------------------------------------------------------
def crear_sombra_base(
    size,
    producto_alpha: Image.Image,
    pos,
    zona_suelo: Image.Image = None,
    contacto: float = 0.55,
    proyectada: float = 0.30,
    direccion_luz: float = 0.0,
) -> Image.Image:
    """
    Mapa de sombra (L, 0-255, ya incluye la intensidad):
      - contacto: elipse fina y oscura en la base (oclusión ambiental)
      - proyectada: silueta aplastada y desenfocada, desplazada según 'direccion_luz' (-1..1)
    Se limita a 'zona_suelo' para que no caiga sobre paredes.
    """
    W, H = size
    px, py = pos
    nw, nh = producto_alpha.size
    y_base = py + nh

    alto = max(int(nh * 0.22), 2)
    sil = producto_alpha.resize((nw, alto), Image.Resampling.BILINEAR)
    sil = sil.point(lambda v: int(v * proyectada))
    capa_proy = Image.new("L", (W, H), 0)
    capa_proy.paste(sil, (px + int(direccion_luz * nw * 0.15), y_base - int(alto * 0.7)))
    capa_proy = _blur(capa_proy, max(nh * 0.04, 2))

    capa_cont = Image.new("L", (W, H), 0)
    ImageDraw.Draw(capa_cont).ellipse(
        (px + nw * 0.03, y_base - nh * 0.030, px + nw * 0.97, y_base + nh * 0.035),
        fill=int(255 * contacto),
    )
    capa_cont = _blur(capa_cont, max(nh * 0.015, 1.5))

    sombra = ImageChops.lighter(capa_proy, capa_cont)
    if zona_suelo is not None:
        sombra = ImageChops.multiply(sombra, zona_suelo)
    return sombra


def aplicar_sombra(room: Image.Image, sombra: Image.Image) -> Image.Image:
    arr = np.asarray(room, dtype=np.float32)
    k = (np.asarray(sombra, dtype=np.float32) / 255.0)[..., None]
    return Image.fromarray(np.clip(arr * (1.0 - k), 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------
# Iluminación: estimar de dónde viene la luz y derivar la sombra de esa estimación
# ----------------------------------------------------------------------------
def _luminancia(img: Image.Image, lado_max: int = 192) -> np.ndarray:
    w, h = img.size
    s = min(1.0, lado_max / max(w, h))
    small = img.convert("RGB").resize(
        (max(int(w * s), 8), max(int(h * s), 8)), Image.Resampling.BILINEAR
    )
    a = np.asarray(small, dtype=np.float32) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _unitario(v):
    n = float(np.hypot(v[0], v[1]))
    return (np.array(v, dtype=np.float32) / n) if n > 1e-6 else np.zeros(2, dtype=np.float32)


def estimar_iluminacion(
    habitacion: Image.Image,
    mascara_suelo: Image.Image = None,
    azimut_deg: float = None,
    elevacion_deg: float = None,
) -> dict:
    """
    Estima la luz dominante de la habitación (vacía) con dos pistas independientes:
      A) masa brillante: centroide de las zonas más luminosas (ventana, lámpara, reflejos)
      B) gradiente del suelo: plano de mejor ajuste a la luminancia del suelo (aclara hacia la luz)
    Si ambas apuntan al mismo lado, la confianza sube; si se contradicen, se cancelan y baja.

    Retorna un dict con:
      direccion      (lx, ly) unitario HACIA la luz en el plano de la imagen (x: derecha, y: abajo)
      azimut_deg     0=luz desde la derecha, 90=desde el fondo, 180=izquierda, 270=desde la cámara
      elevacion_deg  35-80: alta = sombra corta; baja = sombra larga
      confianza      0-1: qué tan fiable es la dirección
      fuerza         0-1: qué tan direccional/dura es la luz (confianza x contraste)
      difusion       0-1: 1 = luz difusa (penumbra ancha)
    Es una heurística de una sola imagen, no un modelo físico: por eso devuelve la confianza
    y la sombra se atenúa y acorta cuando es baja.
    """
    L = _luminancia(habitacion)
    h, w = L.shape
    Lb = np.asarray(
        Image.fromarray((L * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(1.0, w / 80))),
        dtype=np.float32,
    ) / 255.0
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

    # Pista A: masa brillante
    sA, uA = 0.0, np.zeros(2, dtype=np.float32)
    peso = np.clip(Lb - np.percentile(Lb, 90), 0, None)
    if peso.sum() > 1e-6:
        tot = peso.sum()
        cx, cy = float((peso * X).sum() / tot), float((peso * Y).sum() / tot)
        sx = float(np.sqrt((peso * (X - cx) ** 2).sum() / tot))
        sy = float(np.sqrt((peso * (Y - cy) ** 2).sum() / tot))
        compacidad = 1.0 - min(np.hypot(sx, sy) / 0.8, 1.0)
        vA = (cx, 0.5 * cy)  # la altura en la imagen dice poco sobre fondo/frente
        sA = min(np.hypot(*vA) * 1.5, 1.0) * (0.4 + 0.6 * compacidad)
        uA = _unitario(vA)

    # Pista B: gradiente de luminancia sobre el suelo
    if mascara_suelo is not None:
        m = np.asarray(mascara_suelo.convert("L").resize((w, h), Image.Resampling.NEAREST)) > 128
    else:
        m = np.zeros((h, w), dtype=bool)
    if m.sum() < 200:
        m = Y > 0.0  # sin suelo fiable: mitad inferior de la imagen
    elif PISTA_SUELO_BORDE_REL:
        # Solo el borde exterior del suelo: más palanca para el plano y menos textura local
        radio = float(np.sqrt(m.sum() / np.pi))
        nucleo = _gauss_np(m.astype(np.float32), max(0.5, PISTA_SUELO_BORDE_REL * radio / 2.05)) > 0.98
        if (m & ~nucleo).sum() >= 200:
            m = m & ~nucleo
    x, y, z = X[m], Y[m], Lb[m]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    r2 = 1.0 - ((z - A @ coef) ** 2).sum() / (((z - z.mean()) ** 2).sum() + 1e-9)
    grad_rel = float(np.hypot(coef[0], coef[1]) / (z.mean() + 1e-6))
    sB = float(np.clip(grad_rel * 2.5, 0, 1) * np.clip(r2 * 2.0, 0, 1)) * PISTA_SUELO_PESO
    uB = _unitario((float(coef[0]), 0.5 * float(coef[1])))

    # Combinación: pistas alineadas suman, contradictorias se cancelan
    V = sA * uA + sB * uB
    suma = sA + sB
    mag = float(np.hypot(V[0], V[1]))
    if mag > 1e-3 and suma > 1e-3:
        direccion = (float(V[0] / mag), float(V[1] / mag))
        confianza = (mag / suma) * min(1.0, suma)
    else:
        direccion, confianza = (0.0, -1.0), 0.0  # sin información: luz cenital

    # Dureza: contraste global y zonas sobreexpuestas (luz dura) vs. imagen plana (luz difusa)
    p5, p95 = np.percentile(Lb, [5, 95])
    contraste = float((p95 - p5) / (p95 + p5 + 1e-6))
    sobreexp = float((L > 0.92).mean())
    duro = 0.6 * min(contraste / 0.6, 1.0) + 0.4 * min(sobreexp / 0.05, 1.0)
    fuerza = float(confianza * (0.35 + 0.65 * duro))

    fuente, cenital = "estimada", False
    if azimut_deg is not None:
        a = np.radians(azimut_deg)
        direccion = (float(np.cos(a)), float(-np.sin(a)))
        confianza, fuerza, fuente = 1.0, max(fuerza, 0.6), "manual"
    elif confianza < CONFIANZA_MIN_LUZ:
        # Dirección poco fiable: luz cenital suave (sin sombra lateral ni gradiente sobre el sofá)
        direccion, cenital, fuente = (0.0, 0.0), True, "cenital (confianza baja)"
    elev_min, elev_max = ELEVACION_RANGO_DEG
    elev = float(elevacion_deg) if elevacion_deg is not None else elev_max - (elev_max - elev_min) * fuerza
    if cenital and elevacion_deg is None:
        elev = 89.0
    elev = float(np.clip(elev, 10.0, 89.0))

    return dict(
        direccion=direccion,
        azimut_deg=float(np.degrees(np.arctan2(-direccion[1], direccion[0])) % 360),
        elevacion_deg=elev,
        confianza=float(confianza),
        fuerza=float(fuerza),
        difusion=float(np.clip(1.0 - fuerza, 0.1, 1.0)),
        pista_brillo=float(sA), pista_suelo=float(sB), r2_suelo=float(r2),
        contraste=contraste, sobreexpuesto=sobreexp, fuente=fuente, cenital=cenital,
    )


def _lado_luz(luz: dict):
    """Nombre del lado de donde viene la luz (es, en)."""
    lx, ly = luz["direccion"]
    partes_es, partes_en = [], []
    if abs(lx) > 0.35:
        partes_es.append("la izquierda" if lx < 0 else "la derecha")
        partes_en.append("left" if lx < 0 else "right")
    if abs(ly) > 0.6 and not partes_es:
        partes_es.append("el fondo" if ly < 0 else "el frente")
        partes_en.append("behind" if ly < 0 else "front")
    return (" y ".join(partes_es) or "arriba"), (" ".join(partes_en) or "above")


def describir_iluminacion(luz: dict) -> str:
    if luz.get("cenital"):
        return (f"luz cenital suave: confianza {luz['confianza']:.2f} < {CONFIANZA_MIN_LUZ}, "
                f"sin sombra proyectada (solo contacto) [{luz['fuente']}]")
    es, _ = _lado_luz(luz)
    return (
        f"luz desde {es} (azimut {luz['azimut_deg']:.0f}°, elevación {luz['elevacion_deg']:.0f}°), "
        f"confianza {luz['confianza']:.2f}, fuerza {luz['fuerza']:.2f}, difusión {luz['difusion']:.2f} "
        f"[{luz['fuente']}]"
    )


def sufijo_prompt_luz(luz: dict, confianza_min: float = 0.5) -> str:
    """Pista de dirección para el prompt, solo si la estimación es fiable."""
    if luz["confianza"] < confianza_min:
        return ""
    lx, ly = luz["direccion"]
    _, en = _lado_luz(luz)
    if en == "above":
        return "overhead light, soft shadow directly under the furniture"
    opuesto = "left" if lx > 0.35 else "right" if lx < -0.35 else ("front" if ly < 0 else "back")
    return f"light coming from the {en}, soft shadow falling toward the {opuesto}"


def guardar_diagnostico_iluminacion(habitacion: Image.Image, luz: dict, ruta: str):
    """Guarda la habitación con una flecha hacia la luz y el resumen de la estimación."""
    img = habitacion.copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    cx, cy = W // 2, H // 2
    largo = 0.25 * min(W, H) * (0.4 + 0.6 * luz["confianza"])
    ex, ey = cx + luz["direccion"][0] * largo, cy + luz["direccion"][1] * largo
    ancho = max(2, W // 250)
    d.line((cx, cy, ex, ey), fill=(255, 220, 0), width=ancho)
    ang = np.arctan2(ey - cy, ex - cx)
    for da in (2.6, -2.6):
        d.line((ex, ey, ex + 0.12 * largo * np.cos(ang + da), ey + 0.12 * largo * np.sin(ang + da)),
               fill=(255, 220, 0), width=ancho)
    d.rectangle((0, 0, W, 18), fill=(0, 0, 0))
    d.text((6, 4), describir_iluminacion(luz).encode("ascii", "replace").decode(), fill=(255, 255, 255))
    img.save(ruta)


def normalizar_profundidad(prof: Image.Image, size) -> np.ndarray:
    """
    Mapa de profundidad de M2 -> float 0.0 (atrás) .. 1.0 (adelante), normalizado con los percentiles
    2-98 del mapa COMPLETO (así las diferencias entre filas del sofá conservan su magnitud real).
    Depth Anything entrega más claro = más cerca; si tu mapa viniera invertido,
    pon PROFUNDIDAD_CLARO_ES_CERCA = False.
    """
    d = np.asarray(prof.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    lo, hi = np.percentile(d, [2, 98])
    d = np.clip((d - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return d if PROFUNDIDAD_CLARO_ES_CERCA else 1.0 - d


def recortar_profundidad(prof_norm: np.ndarray, pos, size) -> np.ndarray:
    """Recorte (nh, nw) del mapa normalizado bajo el producto (los bordes se replican si sale de la imagen)."""
    px, py = pos
    nw, nh = size
    H, W = prof_norm.shape
    ys = np.clip(np.arange(py, py + nh), 0, H - 1)
    xs = np.clip(np.arange(px, px + nw), 0, W - 1)
    return prof_norm[np.ix_(ys, xs)]


def _factor_profundidad_filas(prof_roi: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Por cada fila del sofá, cuánto más lejos está que su base: 0 en la base, 1 cuando la diferencia de
    profundidad llega a SOMBRA_PROF_REF. Se promedia solo sobre los píxeles opacos de la fila.
    """
    nh = A.shape[0]
    peso = A.sum(axis=1)
    d = np.where(peso > 1e-3, (prof_roi * A).sum(axis=1) / np.maximum(peso, 1e-3), np.nan)
    idx = np.arange(nh)
    ok = ~np.isnan(d)
    if ok.sum() < 2:
        return np.zeros(nh, dtype=np.float32)
    d = np.interp(idx, idx[ok], d[ok])
    k = max(3, (nh // 40) * 2 + 1)  # suavizado de filas (ventana impar)
    d = np.convolve(np.pad(d, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    base = d[-max(2, int(0.08 * nh)):].mean()
    return np.clip((base - d) / SOMBRA_PROF_REF, 0.0, 1.0).astype(np.float32)


def crear_sombra_iluminada(
    size,
    producto_alpha: Image.Image,
    pos,
    luz: dict,
    zona_suelo: Image.Image = None,
    escala_intensidad: float = ESCALA_SOMBRA,
    factor_perspectiva: float = FACTOR_PERSPECTIVA,
    huella_rel: float = SOMBRA_HUELLA_REL,
    longitud_max_rel: float = SOMBRA_LONGITUD_MAX_REL,
    profundidad_roi: np.ndarray = None,
):
    """
    Sombra derivada de la estimación de luz. Retorna (mapa L 0-255, info).
      - proyectada: cada fila del sofá se desplaza sobre el suelo según su altura sobre la base
        (a x h en horizontal, b x h en vertical), amplificada por la profundidad de esa fila:
        las filas más lejanas (respaldo) proyectan una sombra más larga que las cercanas (asiento).
        Con profundidad_roi=None solo cuenta la geometría. Se ignora si la luz es cenital (confianza baja).
      - contacto: solo bajo los puntos opacos más bajos (patas/base), radio de 2-5 px y oscuridad fija
        (SOMBRA_CONTACTO_FIJA), independiente de la dirección y la fuerza de la luz.
    profundidad_roi: recorte (nh, nw) del mapa de M2 ya normalizado 0.0 (atrás) .. 1.0 (adelante);
    ver normalizar_profundidad() y recortar_profundidad().
    """
    W, H = size
    px, py = pos
    nw, nh = producto_alpha.size
    y_base = py + nh
    A = np.asarray(producto_alpha, dtype=np.float32) / 255.0
    cenital = bool(luz.get("cenital", False))

    lx, ly = luz["direccion"]
    ux, uy = -lx, -ly
    tan_e = np.tan(np.radians(luz["elevacion_deg"]))
    a = ux / tan_e                       # px horizontales de sombra por px de altura
    b = uy * factor_perspectiva / tan_e  # px verticales (b>0: hacia la cámara; b<0: hacia el fondo)
    largo = nh / tan_e
    tope = longitud_max_rel * nw
    if largo > tope:
        a, b, largo = a * tope / largo, b * tope / largo, tope
    b_min = 0.03  # evita colapsar la silueta a una línea cuando la luz es lateral pura
    if abs(b) < b_min:
        b = b_min if b >= 0 else -b_min

    f = luz["fuerza"]
    p0, p1 = SOMBRA_PROYECTADA_RANGO
    i_proy = min((p0 + (p1 - p0) * f) * escala_intensidad, 0.7)
    dif = luz["difusion"]
    sigma_proy = max(2.0, nh * (0.02 + 0.06 * dif) + 0.05 * largo * dif)

    # --- Sombra proyectada: falso 3D por filas (omitida con luz cenital) ---
    capa_proy = Image.new("L", (W, H), 0)
    if not cenital:
        z = (_factor_profundidad_filas(profundidad_roi, A) if profundidad_roi is not None
             else np.zeros(nh, dtype=np.float32))
        m_row = 1.0 + SOMBRA_PROF_INFLUENCIA * z
        # Filas intermedias interpoladas: así no quedan huecos cuando el desplazamiento vertical es grande
        dy_ext = abs(b) * nh * (1.0 + SOMBRA_PROF_INFLUENCIA)
        n_sub = int(max(nh, 2 * dy_ext + 2))
        h_s = np.linspace(0, nh - 1, n_sub)          # altura sobre la base (px); 0 = fila de la base
        r_s = (nh - 1) - h_s                          # fila fuente (fraccional)
        r0 = np.floor(r_s).astype(int)
        r1 = np.minimum(r0 + 1, nh - 1)
        w1 = (r_s - r0)[:, None].astype(np.float32)
        A_s = A[r0] * (1.0 - w1) + A[r1] * w1
        m_s = np.interp(r_s, np.arange(nh), m_row)
        dx_s, dy_s = a * h_s * m_s, b * h_s * m_s

        x_lo, x_hi = int(np.floor(min(dx_s.min(), 0.0))), int(np.ceil(max(dx_s.max(), 0.0)))
        y_lo, y_hi = int(np.floor(min(dy_s.min(), 0.0))), int(np.ceil(max(dy_s.max(), 0.0)))
        cw, ch = nw + (x_hi - x_lo) + 2, (y_hi - y_lo) + 2
        proy = np.zeros((ch, cw), dtype=np.float32)
        for i in range(n_sub):                        # bucle por filas: cada una a su propio (dx, dy)
            yy, xx = int(round(dy_s[i])) - y_lo, int(round(dx_s[i])) - x_lo
            seg = proy[yy, xx:xx + nw]
            np.maximum(seg, A_s[i], out=seg)

        # Barrido hacia el fondo: la sombra también cubre la profundidad (huella) del producto
        huella = max(int(huella_rel * nh), 2)
        pasos = 5
        local = np.zeros((ch + huella, cw), dtype=np.float32)
        for i in range(pasos + 1):
            s = huella - int(round(huella * i / pasos))
            seg = local[s:s + ch]
            np.maximum(seg, proy, out=seg)
        capa = Image.fromarray(np.clip(local * 255.0 * i_proy, 0, 255).astype(np.uint8))
        capa_proy.paste(capa, (px + x_lo, y_base + y_lo - huella))
        capa_proy = _blur(capa_proy, sigma_proy)

    # --- Contacto: solo bajo los puntos opacos más bajos (patas / base) ---
    radio = int(np.clip(round(SOMBRA_CONTACTO_RADIO_REL * nw), SOMBRA_CONTACTO_RADIO_MIN, SOMBRA_CONTACTO_RADIO_MAX))
    banda = max(2, int(0.02 * nh))
    pie = (A[nh - banda:, :] > 0.5).astype(np.uint8) * 255
    pad = radio + 2
    lienzo = Image.new("L", (nw + 2 * pad, banda + 2 * pad), 0)
    lienzo.paste(Image.fromarray(pie), (pad, pad))
    lienzo = _blur(_dilatar(lienzo, radio), max(0.8, 0.5 * radio))
    lienzo = lienzo.point(lambda v: int(v * SOMBRA_CONTACTO_FIJA))
    capa_cont = Image.new("L", (W, H), 0)
    capa_cont.paste(lienzo, (px - pad, y_base - banda - pad))

    # Luz cenital: opcionalmente una oclusión ambiental simétrica bajo el sofá (apagada por defecto)
    if cenital and SOMBRA_CENITAL_AO > 0:
        ao = Image.new("L", (W, H), 0)
        ImageDraw.Draw(ao).ellipse(
            (px + nw * 0.05, y_base - nh * 0.03, px + nw * 0.95, y_base + nh * 0.06),
            fill=int(255 * SOMBRA_CENITAL_AO),
        )
        capa_cont = ImageChops.lighter(capa_cont, _blur(ao, max(2.0, nh * 0.03)))

    sombra = ImageChops.lighter(capa_proy, capa_cont)
    if zona_suelo is not None:
        sombra = ImageChops.multiply(sombra, zona_suelo)

    info = dict(
        modo="cenital (solo contacto)" if cenital else "proyectada",
        profundidad=profundidad_roi is not None,
        longitud_px=0 if cenital else int(round(largo)),
        intensidad_proyectada=0.0 if cenital else round(i_proy, 3),
        intensidad_contacto=SOMBRA_CONTACTO_FIJA, radio_contacto_px=radio,
        desenfoque_px=0.0 if cenital else round(sigma_proy, 1),
        cae_hacia="-" if cenital else (
            ("derecha" if ux > 0.35 else "izquierda" if ux < -0.35 else "centro")
            + ("/cámara" if uy > 0.35 else "/fondo" if uy < -0.35 else "")),
    )
    return sombra, info


# ----------------------------------------------------------------------------
# Armonización del producto con la habitación
# (nitidez, ruido, contraste, color, brillo y bordes; determinista, no generativa)
# ----------------------------------------------------------------------------
def _gauss_np(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussiana separable en NumPy sobre los ejes 0 y 1 (funciona con (h,w) y (h,w,c))."""
    if sigma <= 0.05:
        return a
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    w = np.exp(-(x ** 2) / (2 * sigma ** 2))
    w /= w.sum()
    out = a.astype(np.float32, copy=False)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i, wi in enumerate(w):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + n)
            acc += wi * p[tuple(sl)]
        out = acc
    return out


def _gauss_masked(Y: np.ndarray, A: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussiana ponderada por alfa: el fondo transparente no contamina los bordes."""
    if sigma <= 0.05:
        return Y
    return _gauss_np(Y * A, sigma) / np.maximum(_gauss_np(A, sigma), 1e-3)


def _lum(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _sat(rgb: np.ndarray) -> np.ndarray:
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    return (mx - mn) / (mx + 1e-6)


def _recorte_central(h: int, w: int, lado: int = 420):
    y0, x0 = max((h - lado) // 2, 0), max((w - lado) // 2, 0)
    return slice(y0, y0 + min(lado, h)), slice(x0, x0 + min(lado, w))


def _ratio_hf(Y: np.ndarray, A=None, m=None, s: float = 0.0) -> float:
    """
    Detalle fino / detalle medio de la luminancia tras desenfocar 's' px. Mide nitidez
    de forma poco dependiente del contenido: más alto = más nítido.
    (Desenfocar s y luego 1 o 3 equivale a una gaussiana de hypot(s, 1) o hypot(s, 3).)
    """
    g = (lambda sig: _gauss_np(Y, sig)) if A is None else (lambda sig: _gauss_masked(Y, A, sig))
    g0, g1, g3 = g(s), g(float(np.hypot(s, 1.0))), g(float(np.hypot(s, 3.0)))
    hf, mid = g0 - g1, g1 - g3
    if m is not None:
        hf, mid = hf[m], mid[m]
    return float(np.sqrt(np.mean(hf ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-6))


def _sigma_ruido(Y255: np.ndarray, m=None) -> float:
    """Desviación del ruido (escala 0-255), estimador de Immerkaer con mediana (robusto a textura)."""
    c = Y255[1:-1, 1:-1]
    N = (Y255[:-2, :-2] - 2 * Y255[:-2, 1:-1] + Y255[:-2, 2:]
         - 2 * Y255[1:-1, :-2] + 4 * c - 2 * Y255[1:-1, 2:]
         + Y255[2:, :-2] - 2 * Y255[2:, 1:-1] + Y255[2:, 2:])
    v = np.abs(N[m[1:-1, 1:-1]] if m is not None else N)
    if v.size < 100:
        return 0.0
    return float(np.median(v) / (0.6745 * 6.0))


def armonizar_producto(
    prod: Image.Image,
    habitacion: Image.Image,
    pos,
    mascara_suelo: Image.Image = None,
    luz: dict = None,
    fuerza: float = 1.0,
    pesos: dict = None,
    seed: int = 0,
):
    """
    Acerca el aspecto del producto (ya escalado y posicionado) al de la foto de la habitación.
    Mide el cuarto junto a donde irá el sofá y ajusta, con topes para no deformar el producto:
      nitidez   : desenfoca solo si el producto tiene más detalle fino que el cuarto (nunca afila)
      ruido     : añade grano hasta igualar el ruido estimado del cuarto (tope: ruido_max)
      contraste : reduce el contraste "de estudio" (tope: contraste_min) y sube un poco el nivel de negro
      color     : desplaza el balance de blancos hacia la luz ambiente (tope: tinte_max) y desatura (tope: saturacion_min)
      brillo    : escala por la luminosidad local del suelo (tope: brillo_local) + gradiente según la luz estimada
      bordes    : suaviza el alfa como el resto de la foto y mezcla color del fondo en el borde
    No usa difusión: el producto sigue siendo el original, solo retocado.
    Pesos y topes de cada paso: ARMONIZAR_PESOS y ARMONIZAR_TOPES (configuración global).
    Retorna (producto_armonizado RGBA, info con las medidas y los ajustes aplicados).
    """
    pesos = {**ARMONIZAR_PESOS, **(pesos or {})}
    k = {n: float(np.clip(v * fuerza, 0.0, 1.5)) for n, v in pesos.items()}
    px, py = pos
    nw, nh = prod.size
    W, H = habitacion.size

    arr = np.asarray(prod.convert("RGBA"), dtype=np.float32) / 255.0
    rgb, A = arr[..., :3].copy(), arr[..., 3].copy()
    info = {}

    # --- Referencias del cuarto ---
    mx, my = int(0.3 * nw), int(0.3 * nh)
    box = (max(px - mx, 0), max(py - my, 0), min(px + nw + mx, W), min(py + nh + my, H))
    ref = np.asarray(habitacion.crop(box), dtype=np.float32) / 255.0   # alrededor del sofá
    Yref = _lum(ref)
    gs = (max(W // 4, 8), max(H // 4, 8))
    glob = np.asarray(habitacion.resize(gs, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    Yg = _lum(glob)

    interior = _erosionar(Image.fromarray(((A > 0.95) * 255).astype(np.uint8)), 3)
    m_int = np.asarray(interior) > 0
    if m_int.sum() < 200:
        m_int = A > 0.5

    # --- 1. Nitidez: que el sofá no tenga más detalle fino que el cuarto ---
    sigma_ap = 0.0
    if k["nitidez"] > 0 and m_int.sum() >= 200:
        sy, sx = _recorte_central(*Yref.shape)
        r_cuarto = _ratio_hf(Yref[sy, sx])
        cy, cx = _recorte_central(nh, nw)
        Yp = _lum(rgb)
        Yc, Ac, mc = Yp[cy, cx], A[cy, cx], m_int[cy, cx]
        if mc.sum() >= 200:
            objetivo = r_cuarto * ARMONIZAR_TOPES["nitidez_tolerancia"]
            r_prod = _ratio_hf(Yc, Ac, mc, 0.0)
            sigma_b = 0.0
            if r_prod > objetivo:
                sigma_b = ARMONIZAR_TOPES["desenfoque_max_px"]
                for s in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0):
                    if s > ARMONIZAR_TOPES["desenfoque_max_px"]:
                        break
                    if _ratio_hf(Yc, Ac, mc, s) <= objetivo:
                        sigma_b = s
                        break
            sigma_ap = sigma_b * k["nitidez"]
            info.update(ratio_detalle_cuarto=round(r_cuarto, 3), ratio_detalle_producto=round(r_prod, 3))
            if sigma_ap > 0.05:
                pm = _gauss_np(rgb * A[..., None], sigma_ap)
                ab = _gauss_np(A, sigma_ap)
                rgb = np.where((ab > 0.02)[..., None], pm / np.maximum(ab, 1e-3)[..., None], rgb)
    info["desenfoque_px"] = round(float(sigma_ap), 2)

    # --- 2. Brillo local del suelo + gradiente de la luz estimada ---
    f_loc = 1.0
    if k["brillo"] > 0:
        if mascara_suelo is not None:
            ms = np.asarray(mascara_suelo.convert("L").resize(gs, Image.Resampling.NEAREST)) > 128
            bx0, bx1 = int(box[0] * gs[0] / W), max(int(np.ceil(box[2] * gs[0] / W)), int(box[0] * gs[0] / W) + 1)
            by0, by1 = int(box[1] * gs[1] / H), max(int(np.ceil(box[3] * gs[1] / H)), int(box[1] * gs[1] / H) + 1)
            loc = ms[by0:by1, bx0:bx1]
            if loc.sum() > 20 and ms.sum() > 50:
                tb = ARMONIZAR_TOPES["brillo_local"]
                f_loc = float(np.clip(Yg[by0:by1, bx0:bx1][loc].mean() / (Yg[ms].mean() + 1e-6), 1.0 - tb, 1.0 + tb))
        rgb = rgb * (f_loc ** k["brillo"])
        if luz is not None:
            amp = ARMONIZAR_TOPES["gradiente_luz"] * luz["fuerza"] * k["brillo"]
            rampa = luz["direccion"][0] * (2.0 * np.linspace(0, 1, nw, dtype=np.float32) - 1.0)
            rgb = rgb * (1.0 + amp * rampa)[None, :, None]
            info["gradiente_luz"] = round(float(amp), 3)
    info["brillo_local"] = round(f_loc ** k["brillo"], 3)
    rgb = np.clip(rgb, 0, 1)

    # --- 3. Contraste "de estudio" + nivel de negro ---
    if k["contraste"] > 0 and m_int.sum() >= 200:
        Yi = _lum(rgb)
        mu = float(Yi[m_int].mean())
        cv_p = float(Yi[m_int].std() / (mu + 1e-6))
        cv_r = float(Yref.std() / (Yref.mean() + 1e-6))
        c = float(np.clip(cv_r / (cv_p + 1e-6), ARMONIZAR_TOPES["contraste_min"], 1.0))  # solo reduce
        c = 1.0 - (1.0 - c) * k["contraste"]
        rgb = rgb + ((mu + (Yi - mu) * c) - Yi)[..., None]
        lift = float(np.clip(0.5 * (np.percentile(Yg, 1.0) - 0.01), 0.0, ARMONIZAR_TOPES["nivel_negro_max"])) * k["contraste"]
        rgb = rgb * (1.0 - lift) + lift
        info.update(contraste=round(c, 3), nivel_negro=round(lift, 3))
    rgb = np.clip(rgb, 0, 1)

    # --- 4. Color: balance hacia la luz ambiente + saturación ---
    if k["color"] > 0:
        neutro = (Yg > np.percentile(Yg, 80)) & (_sat(glob) < 0.30)  # paredes, techo, blancos
        if neutro.sum() > 200:
            t = glob[neutro].mean(axis=0)
            t = t / t.mean()
            gains = np.clip(1.0 + 0.5 * k["color"] * (t - 1.0), 1.0 - ARMONIZAR_TOPES["tinte_max"], 1.0 + ARMONIZAR_TOPES["tinte_max"])
            rgb = rgb * gains[None, None, :]
            info["tinte_rgb"] = [round(float(g), 3) for g in gains]
        if m_int.sum() >= 200:
            sat_p = float(_sat(np.clip(rgb, 0, 1))[m_int].mean())
            sat_r = float(_sat(ref).mean())
            s = float(np.clip(sat_r / (sat_p + 1e-6), ARMONIZAR_TOPES["saturacion_min"], 1.0))  # solo desatura
            s = 1.0 - (1.0 - s) * k["color"]
            Yi = _lum(rgb)[..., None]
            rgb = Yi + (rgb - Yi) * s
            info["saturacion"] = round(s, 3)
    rgb = np.clip(rgb, 0, 1)

    # --- 5. Ruido: igualar el grano del cuarto ---
    if k["ruido"] > 0 and m_int.sum() >= 200:
        sig_cuarto = _sigma_ruido(Yref * 255.0)
        sig_prod = _sigma_ruido(_lum(rgb) * 255.0, m_int)
        extra = float(min(np.sqrt(max(sig_cuarto ** 2 - sig_prod ** 2, 0.0)) * k["ruido"], ARMONIZAR_TOPES["ruido_max"]))
        info.update(ruido_cuarto=round(sig_cuarto, 2), ruido_producto=round(sig_prod, 2), ruido_añadido=round(extra, 2))
        if extra > 0.2:
            rng = np.random.default_rng(seed)
            n = _gauss_np(rng.standard_normal((nh, nw)).astype(np.float32), 0.6)  # grano con leve correlación
            n = n / (n.std() + 1e-6)
            nc = rng.standard_normal((nh, nw, 3)).astype(np.float32) * 0.35        # un poco de ruido cromático
            rgb = rgb + (n[..., None] * 0.9 + nc) * (extra / 255.0)
    rgb = np.clip(rgb, 0, 1)

    # --- 6. Bordes: alfa tan suave como el resto de la foto + luz envolvente ---
    if k["bordes"] > 0:
        s_e = max(0.7, 0.8 * sigma_ap) * k["bordes"]
        A2 = _gauss_np(A, s_e)
        banda = np.clip(1.0 - np.abs(2.0 * A2 - 1.0), 0.0, 1.0)  # máxima justo en el borde
        fondo = habitacion.crop((px, py, px + nw, py + nh)).filter(ImageFilter.GaussianBlur(6.0))
        fondo = np.asarray(fondo, dtype=np.float32) / 255.0
        w_lw = (ARMONIZAR_TOPES["luz_envolvente"] * k["bordes"]) * banda
        rgb = rgb * (1.0 - w_lw[..., None]) + fondo * w_lw[..., None]
        A = np.clip((A2 - 0.08) / 0.92, 0.0, 1.0)  # recorta el halo exterior que deja el desenfoque
        info["borde_px"] = round(float(s_e), 2)

    out = np.concatenate([np.clip(rgb, 0, 1), A[..., None]], axis=-1)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)), info


def resumen_armonizacion(info: dict) -> str:
    partes = [f"desenfoque={info.get('desenfoque_px', 0)}px"]
    if "ruido_añadido" in info:
        partes.append(f"grano +{info['ruido_añadido']} (cuarto {info['ruido_cuarto']}, producto {info['ruido_producto']})")
    if "contraste" in info:
        partes.append(f"contraste x{info['contraste']}")
    if "tinte_rgb" in info:
        partes.append(f"tinte {tuple(info['tinte_rgb'])}")
    if "saturacion" in info:
        partes.append(f"saturación x{info['saturacion']}")
    partes.append(f"brillo local x{info.get('brillo_local', 1.0)}")
    if "borde_px" in info:
        partes.append(f"borde σ={info['borde_px']}px")
    return ", ".join(partes)


# ----------------------------------------------------------------------------
# Puntos 1, 5 y 6: máscara de integración pequeña (el interior del producto queda protegido)
# ----------------------------------------------------------------------------
def crear_mascara_integracion(
    alpha_full: Image.Image,
    sombra: Image.Image,
    anillo_px: int,
    borde_editable_px: int = 0,
    umbral_sombra: int = 10,
    incluir_sombra: bool = True,
) -> Image.Image:
    """
    Blanco = lo que el modelo puede editar: un anillo de 'anillo_px' alrededor del producto
    más la zona de sombra. Negro = todo lo demás, incluido el producto.
    borde_editable_px = 0 -> el producto completo queda protegido (valor para empezar).
    Subirlo a 2-3 permite que el modelo retoque una banda fina del borde para fundir la costura.
    """
    prod_bin = alpha_full.point(lambda v: 255 if v > 127 else 0)
    protegido = _erosionar(prod_bin, borde_editable_px)  # px=0 -> todo el producto
    anillo = ImageChops.subtract(_dilatar(prod_bin, anillo_px), prod_bin)
    zona = anillo
    if incluir_sombra:
        # La zona de sombra también es editable: puede ampliar bastante el área si es grande
        sombra_bin = sombra.point(lambda v: 255 if v > umbral_sombra else 0)
        zona = ImageChops.lighter(anillo, sombra_bin)
    return ImageChops.subtract(zona, protegido)


def calcular_roi(mask_local: Image.Image, producto_box, margen: float = MARGEN_ROI, min_lado: int = 256):
    """ROI cuadrada alrededor de la zona editable (+contexto): sin letterbox y con más resolución."""
    W, H = mask_local.size
    bb = mask_local.getbbox()
    if bb is None:
        raise ValueError("La máscara de integración quedó vacía")
    l, t = min(bb[0], producto_box[0]), min(bb[1], producto_box[1])
    r, b = max(bb[2], producto_box[2]), max(bb[3], producto_box[3])
    lado = int(min(max(max(r - l, b - t) * (1 + 2 * margen), min_lado), W, H))
    cx, cy = (l + r) / 2, (t + b) / 2
    x0 = int(np.clip(round(cx - lado / 2), 0, W - lado))
    y0 = int(np.clip(round(cy - lado / 2), 0, H - lado))
    return (x0, y0, x0 + lado, y0 + lado)


# ----------------------------------------------------------------------------
# Preparación de la escena (sin modelos de difusión)
# ----------------------------------------------------------------------------
def preparar_escena(
    habitacion_path,
    mascara_path,
    producto_path,
    prefix,
    ancho_rel=ANCHO_REL,
    alto_max_rel=ALTO_MAX_REL,
    pos_x_rel=POS_X_REL,
    base_rel=BASE_REL,
    ancho_objetivo=ANCHO_OBJETIVO,
    offset=OFFSET,
    sombra=SOMBRA,
    intensidad_contacto=INTENSIDAD_CONTACTO,
    intensidad_proyectada=INTENSIDAD_PROYECTADA,
    direccion_luz=DIRECCION_LUZ,
    iluminacion=ILUMINACION,
    escala_intensidad=ESCALA_SOMBRA,
    factor_perspectiva=FACTOR_PERSPECTIVA,
    azimut_luz_deg=AZIMUT_LUZ_DEG,
    elevacion_luz_deg=ELEVACION_LUZ_DEG,
    anillo_px=ANILLO_PX,
    borde_editable_px=BORDE_EDITABLE_PX,
    incluir_sombra_en_mascara=INCLUIR_SOMBRA_EN_MASCARA,
    armonizar=ARMONIZAR,
    armonizar_fuerza=ARMONIZAR_FUERZA,
    armonizar_pesos=None,
    margen_roi=MARGEN_ROI,
    profundidad_path=None,
    recalcular_profundidad=RECALCULAR_PROFUNDIDAD,
):
    habitacion = Image.open(habitacion_path).convert("RGB")
    mascara_suelo = Image.open(mascara_path).convert("L")
    producto = Image.open(producto_path).convert("RGBA")
    if mascara_suelo.size != habitacion.size:
        mascara_suelo = mascara_suelo.resize(habitacion.size, Image.Resampling.NEAREST)
    W, H = habitacion.size
    print(f"Habitación: {W}x{H}")

    # La máscara de suelo solo define la zona de apoyo (NO es mask_image)
    ys, xs = np.nonzero(np.array(mascara_suelo) > 128)
    if len(xs) == 0:
        raise ValueError("La máscara de suelo está vacía (toda negra)")
    x1, x2 = np.percentile(xs, [0.5, 99.5]).astype(int)
    y1, y2 = np.percentile(ys, [0.5, 99.5]).astype(int)
    floor_bbox = (int(x1), int(y1), int(x2), int(y2))
    print(f"Zona de suelo: {floor_bbox[:2]} -> {floor_bbox[2:]}")

    # 2. recortar producto  3. apoyar
    producto = recortar_producto_transparente(producto)

    # Escala automática si no se forzó ancho_rel ni ancho_objetivo: tamaño real + perspectiva
    info_escala = None
    if ancho_objetivo is None and ancho_rel is None:
        ancho_objetivo, info_escala = estimar_escala_perspectiva(
            mascara_suelo, floor_bbox, (W, H), producto, base_rel=base_rel, offset=offset,
        )
        print(
            f"Escala automática: sofá {info_escala['ancho_m']} m -> {info_escala['ancho_px']} px "
            f"({info_escala['px_por_metro']} px/m; horizonte {info_escala['horizonte_rel']}H "
            f"[{info_escala['horizonte_metodo']}, conf {info_escala['horizonte_confianza']}]"
            + (f"; tope por suelo {info_escala['limite_por_suelo']} px" if info_escala["limite_por_suelo"] else "")
            + ")"
        )

    prod, (px, py) = calcular_colocacion(
        floor_bbox, (W, H), producto, ancho_rel, alto_max_rel, pos_x_rel, base_rel,
        ancho_objetivo, offset,
    )
    nw, nh = prod.size
    print(f"Producto: {nw}x{nh}, base apoyada en y={py + nh}, x=[{px},{px + nw}]")
    if info_escala is not None:
        guardar_diagnostico_escala(habitacion, info_escala, (px, py, nw, nh), f"{prefix}_escala.png")

    apoyo = verificar_apoyo(mascara_suelo, prod.getchannel("A"), (px, py))
    print(f"Apoyo sobre el suelo detectado: {apoyo:.0%}")
    if apoyo < 0.7:
        print("AVISO: parte de la base queda fuera del suelo. Ajusta base_rel, pos_x_rel u offset.")

    # Iluminación: se estima antes porque la usan la armonización y la sombra
    luz = None
    if iluminacion == "auto" and (sombra or armonizar):
        luz = estimar_iluminacion(habitacion, mascara_suelo, azimut_luz_deg, elevacion_luz_deg)
        print(f"Iluminación: {describir_iluminacion(luz)}")
        guardar_diagnostico_iluminacion(habitacion, luz, f"{prefix}_iluminacion.png")

    # Compositing crudo (referencia para comparar) y armonización del producto
    sin_sombra = habitacion.copy()
    sin_sombra.paste(prod, (px, py), prod)
    info_arm = None
    if armonizar:
        prod, info_arm = armonizar_producto(
            prod, habitacion, (px, py), mascara_suelo, luz, armonizar_fuerza, armonizar_pesos,
        )
        print(f"Armonización: {resumen_armonizacion(info_arm)}")
        with open(f"{prefix}_armonizacion.json", "w", encoding="utf-8") as fh:
            json.dump(info_arm, fh, indent=2, ensure_ascii=False)

    alpha_full = Image.new("L", (W, H), 0)
    alpha_full.paste(prod.getchannel("A"), (px, py))

    # 4. sombra base sobre la habitación (antes de pegar el producto)
    room_sombra = habitacion
    mapa_sombra = Image.new("L", (W, H), 0)
    if sombra:
        zona_suelo = _dilatar(mascara_suelo, max(4, int(0.01 * max(W, H))))
        if iluminacion == "auto":
            prof_roi = None
            if profundidad_path and os.path.exists(profundidad_path):
                prof_norm = normalizar_profundidad(Image.open(profundidad_path), (W, H))
                prof_roi = recortar_profundidad(prof_norm, (px, py), (nw, nh))
            mapa_sombra, info_sombra = crear_sombra_iluminada(
                (W, H), prod.getchannel("A"), (px, py), luz, zona_suelo,
                escala_intensidad, factor_perspectiva, profundidad_roi=prof_roi,
            )
            print(f"Sombra [{info_sombra['modo']}, profundidad={'sí' if info_sombra['profundidad'] else 'no'}]: "
                  f"cae hacia {info_sombra['cae_hacia']}, longitud={info_sombra['longitud_px']}px, "
                  f"intensidad contacto/proyectada={info_sombra['intensidad_contacto']}/"
                  f"{info_sombra['intensidad_proyectada']}, desenfoque={info_sombra['desenfoque_px']}px")
            with open(f"{prefix}_iluminacion.json", "w", encoding="utf-8") as fh:
                json.dump({"luz": luz, "sombra": info_sombra}, fh, indent=2, ensure_ascii=False)
        else:
            mapa_sombra = crear_sombra_base(
                (W, H), prod.getchannel("A"), (px, py), zona_suelo,
                intensidad_contacto, intensidad_proyectada, direccion_luz,
            )
        room_sombra = aplicar_sombra(habitacion, mapa_sombra)
    compo = room_sombra.copy()
    compo.paste(prod, (px, py), prod)

    # 5. máscara de integración pequeña
    if anillo_px is None:
        anillo_px = max(ANILLO_AUTO_MIN, int(ANILLO_AUTO_REL * max(nw, nh)))
    mask_local = crear_mascara_integracion(
        alpha_full, mapa_sombra, anillo_px, borde_editable_px,
        incluir_sombra=incluir_sombra_en_mascara,
    )
    roi = calcular_roi(mask_local, (px, py, px + nw, py + nh), margen_roi)
    print(f"Máscara de integración: anillo={anillo_px}px, borde editable={borde_editable_px}px, ROI={roi}")

    out_dir = os.path.dirname(prefix)
    sin_sombra.save(os.path.join(out_dir, "test_compositing.png"))
    sin_sombra.save(f"{prefix}_00_compositing.png")
    compo.save(f"{prefix}_01_con_sombra.png")
    mask_local.save(f"{prefix}_mascara_integracion.png")
    alpha_full.save(f"{prefix}_producto_alpha.png")
    with open(f"{prefix}_colocacion.json", "w", encoding="utf-8") as fh:
        json.dump(dict(
            imagen=[int(W), int(H)], pos=[int(px), int(py)], size=[int(nw), int(nh)],
            roi=[int(v) for v in roi], floor_bbox=[int(v) for v in floor_bbox],
            apoyo=round(float(apoyo), 3), anillo_px=int(anillo_px), borde_editable_px=int(borde_editable_px),
            escala=info_escala,
        ), fh, indent=2)

    return dict(
        habitacion=habitacion, sin_sombra=sin_sombra, compo=compo, mask_local=mask_local,
        roi=roi, prod=prod, pos=(px, py), prefix=prefix, profundidad=None,
        profundidad_path=profundidad_path, recalcular_profundidad=recalcular_profundidad,
        compo_path=f"{prefix}_01_con_sombra.png", luz=luz, info_armonizacion=info_arm,
        borde_editable_px=borde_editable_px,
    )


# ----------------------------------------------------------------------------
# Profundidad (solo para ControlNet)
# ----------------------------------------------------------------------------
def _asegurar_profundidad(escena):
    """
    Por defecto se recalcula sobre la composición CON el producto: el mapa del cuarto vacío
    (el que entrega main_pipeline) describe suelo plano donde ahora hay un sofá.
    Con recalcular_profundidad=False se usa el mapa recibido.
    """
    if escena["profundidad"] is not None:
        return escena["profundidad"]

    ruta = escena["profundidad_path"]
    if escena["recalcular_profundidad"] or not ruta or not os.path.exists(ruta):
        ruta = f"{escena['prefix']}_profundidad.png"
        print("Calculando profundidad (Módulo 2) sobre la composición con el producto...")
        from modulos.m2_depth import estimar_profundidad
        estimar_profundidad(input_path=escena["compo_path"], output_path=ruta)
        liberar_memoria()

    prof = Image.open(ruta).convert("RGB")
    if prof.size != escena["compo"].size:
        prof = prof.resize(escena["compo"].size, Image.Resampling.BICUBIC)
    escena["profundidad"] = prof
    return prof


# ----------------------------------------------------------------------------
# Pipelines y generación
# ----------------------------------------------------------------------------
def _cargar_pipeline(con_controlnet: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if con_controlnet:
        print("Cargando ControlNet Depth + Inpainting...")
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            INPAINT_MODEL, controlnet=controlnet, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        )
    else:
        print("Cargando Inpainting (sin ControlNet)...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINT_MODEL, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def generar(
    escena,
    con_controlnet,
    prompt=PROMPT_DEFAULT,
    negative_prompt=NEGATIVE_DEFAULT,
    strength=0.30,
    guidance_scale=7.0,
    num_steps=25,
    controlnet_scale=0.55,
    seed=42,
    pipe=None,
) -> Image.Image:
    """
    Inpainting sobre la ROI con la máscara de integración. Solo cambia lo que está dentro
    de la máscara; al final el producto original se vuelve a pegar intacto.
    """
    roi = escena["roi"]
    lado = roi[2] - roi[0]
    box = (WORK_SIZE, WORK_SIZE)

    img_roi = escena["compo"].crop(roi).resize(box, Image.Resampling.LANCZOS)
    mask_roi = (
        escena["mask_local"].crop(roi).resize(box, Image.Resampling.BOX)
        .point(lambda v: 255 if v > 32 else 0)
    )

    kwargs = dict(
        prompt=prompt, negative_prompt=negative_prompt,
        image=img_roi, mask_image=mask_roi,
        num_inference_steps=num_steps, guidance_scale=guidance_scale, strength=strength,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    if con_controlnet:
        ctrl = _asegurar_profundidad(escena).crop(roi).resize(box, Image.Resampling.BICUBIC)
        kwargs.update(control_image=ctrl, controlnet_conditioning_scale=controlnet_scale)

    propio = pipe is None
    if propio:
        pipe = _cargar_pipeline(con_controlnet)
    gen_512 = pipe(**kwargs).images[0]
    if propio:
        del pipe
        liberar_memoria()

    gen = gen_512.resize((lado, lado), Image.Resampling.LANCZOS)
    base_roi = escena["compo"].crop(roi)
    feather = _blur(escena["mask_local"].crop(roi), max(1.5, lado * 0.004))
    roi_mezcla = Image.composite(gen, base_roi, feather)

    final = escena["compo"].copy()
    final.paste(roi_mezcla, roi[:2])
    # El producto original se re-pega intacto, salvo una banda de 'borde_editable_px' en su contorno:
    # ahí se conserva lo retocado por el modelo (costura orgánica con la sombra de contacto).
    prod = escena["prod"]
    alfa_pegado = prod.getchannel("A")
    borde = int(escena.get("borde_editable_px", 0))
    if borde > 0:
        nucleo = _erosionar(alfa_pegado.point(lambda v: 255 if v > 127 else 0), borde)
        alfa_pegado = ImageChops.darker(alfa_pegado, _blur(nucleo, max(0.8, borde / 3.0)))
    final.paste(prod, escena["pos"], alfa_pegado)
    return final


def _hoja_comparacion(items, roi, ruta):
    tiles = []
    for titulo, img in items:
        t = img.crop(roi).resize((WORK_SIZE, WORK_SIZE), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(t)
        d.rectangle((0, 0, WORK_SIZE, 22), fill=(0, 0, 0))
        d.text((6, 5), titulo, fill=(255, 255, 255))
        tiles.append(t)
    hoja = Image.new("RGB", (WORK_SIZE * len(tiles), WORK_SIZE))
    for i, t in enumerate(tiles):
        hoja.paste(t, (i * WORK_SIZE, 0))
    hoja.save(ruta)


# ----------------------------------------------------------------------------
# API principal (compatible con main_pipeline.py)
# ----------------------------------------------------------------------------
def insertar_producto(
    habitacion_path: str,
    mascara_path: str,
    producto_path: str,
    profundidad_path: str = None,
    output_path: str = None,
    prompt: str = PROMPT_DEFAULT,
    negative_prompt: str = NEGATIVE_DEFAULT,
    modo: str = MODO_DEFAULT,
    preferido: str = PREFERIDO,
    # Todo lo siguiente toma su valor por defecto de la CONFIGURACIÓN GLOBAL (arriba del archivo)
    ancho_rel: float = ANCHO_REL,
    alto_max_rel: float = ALTO_MAX_REL,
    pos_x_rel: float = POS_X_REL,
    base_rel: float = BASE_REL,
    ancho_objetivo: int = ANCHO_OBJETIVO,
    offset=OFFSET,
    sombra: bool = SOMBRA,
    intensidad_contacto: float = INTENSIDAD_CONTACTO,
    intensidad_proyectada: float = INTENSIDAD_PROYECTADA,
    direccion_luz: float = DIRECCION_LUZ,
    iluminacion: str = ILUMINACION,
    escala_intensidad: float = ESCALA_SOMBRA,
    factor_perspectiva: float = FACTOR_PERSPECTIVA,
    azimut_luz_deg: float = AZIMUT_LUZ_DEG,
    elevacion_luz_deg: float = ELEVACION_LUZ_DEG,
    prompt_con_luz: bool = PROMPT_CON_LUZ,
    anillo_px: int = ANILLO_PX,
    borde_editable_px: int = BORDE_EDITABLE_PX,
    incluir_sombra_en_mascara: bool = INCLUIR_SOMBRA_EN_MASCARA,
    armonizar: bool = ARMONIZAR,
    armonizar_fuerza: float = ARMONIZAR_FUERZA,
    armonizar_pesos: dict = None,    # None = ARMONIZAR_PESOS; p. ej. {"ruido": 0.5, "color": 0}
    strength: float = STRENGTH,
    guidance_scale: float = GUIDANCE_SCALE,
    num_steps: int = NUM_STEPS,
    controlnet_scale: float = CONTROLNET_SCALE,
    seed: int = SEED,
    recalcular_profundidad: bool = RECALCULAR_PROFUNDIDAD,
) -> str:
    """
    modo:
      "base"           -> compositing + sombra, sin difusión (valida colocación, sombra y máscara)
      "sin_controlnet" -> inpainting local sin ControlNet
      "controlnet"     -> inpainting local con ControlNet Depth
      "comparar"       -> ambos + hoja comparativa
    """
    print("\n=== MÓDULO 4: Inserción local (sombra + inpainting en máscara de integración) ===")
    if modo not in ("base", "sin_controlnet", "controlnet", "comparar"):
        raise ValueError(f"modo inválido: {modo}")
    if iluminacion not in ("auto", "manual"):
        raise ValueError(f"iluminacion inválida: {iluminacion}")

    # Compatibilidad posicional: (h, m, p, out)
    if output_path is None and profundidad_path is not None:
        output_path, profundidad_path = profundidad_path, None
    if output_path is None:
        output_path = "data/outputs/resultado_final.png"

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, os.path.splitext(os.path.basename(output_path))[0])

    escena = preparar_escena(
        habitacion_path, mascara_path, producto_path, prefix,
        ancho_rel=ancho_rel, alto_max_rel=alto_max_rel, pos_x_rel=pos_x_rel, base_rel=base_rel,
        ancho_objetivo=ancho_objetivo, offset=offset,
        sombra=sombra, intensidad_contacto=intensidad_contacto,
        intensidad_proyectada=intensidad_proyectada, direccion_luz=direccion_luz,
        iluminacion=iluminacion, escala_intensidad=escala_intensidad,
        factor_perspectiva=factor_perspectiva, azimut_luz_deg=azimut_luz_deg,
        elevacion_luz_deg=elevacion_luz_deg,
        anillo_px=anillo_px, borde_editable_px=borde_editable_px,
        incluir_sombra_en_mascara=incluir_sombra_en_mascara,
        armonizar=armonizar, armonizar_fuerza=armonizar_fuerza, armonizar_pesos=armonizar_pesos,
        profundidad_path=profundidad_path, recalcular_profundidad=recalcular_profundidad,
    )

    if modo == "base":
        escena["compo"].save(output_path)
        print(f"Resultado (compositing + sombra, sin difusión) en: {output_path}")
        return output_path

    if prompt_con_luz and escena.get("luz") is not None:
        sufijo = sufijo_prompt_luz(escena["luz"])
        if sufijo:
            prompt = f"{prompt}, {sufijo}"
            print(f"Prompt + luz: {sufijo}")

    params = dict(
        prompt=prompt, negative_prompt=negative_prompt, strength=strength,
        guidance_scale=guidance_scale, num_steps=num_steps,
        controlnet_scale=controlnet_scale, seed=seed,
    )
    resultados = {}

    if modo in ("sin_controlnet", "comparar"):
        print(f"Generando SIN ControlNet (steps={num_steps}, strength={strength})...")
        resultados["sin_controlnet"] = generar(escena, False, **params)
        resultados["sin_controlnet"].save(f"{prefix}_02_sin_controlnet.png")

    if modo in ("controlnet", "comparar"):
        print(f"Generando CON ControlNet (controlnet_scale={controlnet_scale})...")
        resultados["controlnet"] = generar(escena, True, **params)
        resultados["controlnet"].save(f"{prefix}_03_controlnet.png")

    if modo == "comparar":
        _hoja_comparacion(
            [
                ("1. Compositing", escena["sin_sombra"]),
                ("2. + sombra y armonizado", escena["compo"]),
                ("3. Sin ControlNet", resultados["sin_controlnet"]),
                ("4. Con ControlNet", resultados["controlnet"]),
            ],
            escena["roi"], f"{prefix}_04_comparacion.png",
        )
        print(f"Hoja comparativa: {prefix}_04_comparacion.png")
        final = resultados[preferido]
    else:
        final = resultados[modo]

    final.save(output_path)
    print(f"Resultado final guardado en: {output_path}")
    return output_path


# ----------------------------------------------------------------------------
# Ajuste fino (solo al final)
# ----------------------------------------------------------------------------
def barrido_parametros(
    habitacion_path,
    mascara_path,
    producto_path,
    output_dir,
    con_controlnet=False,
    strengths=(0.20, 0.30, 0.40, 0.50),
    guidance_scales=(7.0,),
    controlnet_scales=(0.40, 0.55, 0.70),
    num_steps=25,
    seed=42,
    **opciones_escena,  # ancho_rel, base_rel, offset, anillo_px, borde_editable_px, ...
):
    """Carga el modelo una sola vez y guarda una imagen por combinación. Retorna las rutas."""
    os.makedirs(output_dir, exist_ok=True)
    escena = preparar_escena(
        habitacion_path, mascara_path, producto_path,
        os.path.join(output_dir, "barrido"), **opciones_escena,
    )
    if con_controlnet:
        _asegurar_profundidad(escena)

    pipe = _cargar_pipeline(con_controlnet)
    rutas = []
    cns = controlnet_scales if con_controlnet else (None,)
    for st, gs, cn in itertools.product(strengths, guidance_scales, cns):
        etiqueta = f"st{st}_gs{gs}" + (f"_cn{cn}" if cn is not None else "")
        print(f"Barrido: {etiqueta}")
        img = generar(
            escena, con_controlnet, strength=st, guidance_scale=gs, num_steps=num_steps,
            controlnet_scale=cn if cn is not None else 0.55, seed=seed, pipe=pipe,
        )
        ruta = os.path.join(output_dir, f"{'cn' if con_controlnet else 'sincn'}_{etiqueta}.png")
        img.save(ruta)
        rutas.append(ruta)

    del pipe
    liberar_memoria()
    return rutas