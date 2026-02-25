import os
import cv2
import random

# Setări de bază
SOURCE_DIR = "face_patches_kaggle"  # Folderul cu imaginile originale
OUTPUT_DIR = "nonface_patches(test_copy)"  # Folder unde salvăm patch-urile
TARGET_SIZE = (64, 64)  # Dimensiunea finală
NUM_CROPS_PER_IMAGE = 2  # Câte crop-uri vrem pe fiecare imagine
MAX_SAMPLES = 4000  # Maximum de patch-uri generate

# Inițializare
os.makedirs(OUTPUT_DIR, exist_ok=True)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0

# Luăm doar imaginile de la index 4000 la 7000
filenames = os.listdir(SOURCE_DIR)[7000:8000]

for filename in filenames:
    if count >= MAX_SAMPLES:
        break

    path = os.path.join(SOURCE_DIR, filename)
    image = cv2.imread(path)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Dacă imaginea are deja față, o sărim
    if len(faces) > 0:
        continue

    roi_h, roi_w = image.shape[:2]

    # Sărim dacă imaginea este prea mică
    if min(roi_w, roi_h) < 64:
        continue

    for i in range(NUM_CROPS_PER_IMAGE):
        if count >= MAX_SAMPLES:
            break

        crop_size = random.randint(64, min(256, min(roi_w, roi_h)))

        max_x = roi_w - crop_size
        max_y = roi_h - crop_size
        if max_x <= 0 or max_y <= 0:
            continue

        cx = random.randint(0, max_x)
        cy = random.randint(0, max_y)

        crop = image[cy:cy + crop_size, cx:cx + crop_size]

        # Verificăm dacă crop-ul extras conține față
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_faces = face_cascade.detectMultiScale(crop_gray, scaleFactor=1.1, minNeighbors=5)

        if len(crop_faces) > 0:
            continue  # Dacă crop-ul are față, îl ignorăm

        resized = cv2.resize(crop, TARGET_SIZE)
        save_name = f"{os.path.splitext(filename)[0]}_partial_{i}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, resized)

        count += 1
        print(f"[{count}] Salvat: {save_name}")

print(f"\n✅ Gata! Au fost generate {count} patch-uri fără fețe.")
