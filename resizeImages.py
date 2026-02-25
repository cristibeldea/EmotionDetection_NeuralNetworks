import os
from PIL import Image

# Setăm calea către folderul cu poze
folder_path = "nonface_patches"

# Ne asigurăm că folderul există
if not os.path.exists(folder_path):
    raise ValueError(f"Folderul '{folder_path}' nu există!")

# Parcurgem toate fișierele din folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    try:
        with Image.open(file_path) as img:
            if img.size != (256, 256):
                print(f"Redimensionez: {filename} (dimensiune curentă: {img.size})")
                img = img.resize((256, 256), Image.LANCZOS)
                img.save(file_path)  # Suprascriem imaginea originală
            else:
                print(f"Imagine deja corectă: {filename}")
    except Exception as e:
        print(f"Eroare la procesarea fișierului {filename}: {e}")
