import os
import cv2

# Paths
SOURCE_DIR = "face_patches_kaggle"      # Folder with input face images
OUTPUT_DIR = "face_patches"             # Folder to save cropped 64x64 color face patches
TARGET_SIZE = (64, 64)

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Haar cascade (still requires grayscale for detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Process each image
for filename in os.listdir(SOURCE_DIR):
    filepath = os.path.join(SOURCE_DIR, filename)
    image = cv2.imread(filepath)

    if image is None:
        print(f"[WARNING] Could not read: {filename}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Keep only the largest detected face
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face = image[y:y + h, x:x + w]  # Use original color image
        resized = cv2.resize(face, TARGET_SIZE)

        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_face.png")
        cv2.imwrite(save_path, resized)
        print(f"[INFO] Saved: {save_path}")
    else:
        print(f"[INFO] No face found in: {filename}")

print("\n✅ All color face patches generated.")
