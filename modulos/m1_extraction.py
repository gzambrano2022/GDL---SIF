import os
import gc
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
import importlib.util
import sys

def extraer_producto(input_path: str, output_path: str) -> str:
    print("\n=== MÓDULO 1: Extracción del Producto (RMBG-1.4) ===")

    print("Descargando archivos del modelo...")
    config_path = hf_hub_download(repo_id="briaai/RMBG-1.4", filename="MyConfig.py")
    model_path  = hf_hub_download(repo_id="briaai/RMBG-1.4", filename="briarmbg.py")
    weights_path = hf_hub_download(repo_id="briaai/RMBG-1.4", filename="pytorch_model.bin")

    # Cargar MyConfig
    spec_config = importlib.util.spec_from_file_location("MyConfig", config_path)
    myconfig = importlib.util.module_from_spec(spec_config)
    sys.modules["MyConfig"] = myconfig
    spec_config.loader.exec_module(myconfig)

    # Cargar y parchear briarmbg.py
    with open(model_path, "r", encoding="utf-8") as f:
        code = f.read()
    code = code.replace("from .MyConfig import RMBGConfig", "from MyConfig import RMBGConfig")

    bria_module = type(sys)("BriaRMBG_module")
    exec(code, bria_module.__dict__)

    net = bria_module.BriaRMBG()
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()
    print(f"Modelo cargado en: {device}")

    # Imagen
    orig_im = Image.open(input_path).convert("RGB")
    orig_size = orig_im.size

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])
    input_tensor = transform(orig_im).unsqueeze(0).to(device)

    with torch.no_grad():
        result = net(input_tensor)

    # ---------- Extracción robusta de la máscara ----------
    # El modelo puede devolver estructuras anidadas
    mask = result
    while isinstance(mask, (list, tuple)):
        mask = mask[0]

    mask = mask.cpu().detach()

    # Quitar dimensiones extra (batch, channel...)
    while mask.ndim > 2:
        mask = mask.squeeze(0)

    # Normalizar a [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    mask_pil = transforms.ToPILImage()(mask).resize(orig_size, Image.Resampling.BILINEAR)

    output_im = orig_im.copy()
    output_im.putalpha(mask_pil)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_im.save(output_path)
    print(f"Producto aislado guardado en: {output_path}")

    # Liberar memoria
    net.to("cpu")
    del net, input_tensor, result, mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("→ VRAM liberada")

    return output_path