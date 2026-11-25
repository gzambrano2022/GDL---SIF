from PIL import Image

def to512(input_path, output_path):

    with Image.open(input_path) as img:
        img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        img_resized.save(output_path)
