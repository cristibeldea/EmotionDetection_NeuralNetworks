import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from face_detector_model import build_face_detector
import random
import cv2

TARGET_SIZE = (64, 64)  # Final model input size
MULTISCALE_SIZES = [(48, 48), (64, 64), (96, 96), (128, 128), (256, 256), (528, 528)]  # Simulated scales

def load_data(face_dir, non_face_dir):
    data = []
    labels = []

    print("[INFO] Loading images...")

    # Calculate total number of images for progress display
    face_files = os.listdir(face_dir)
    non_face_files = os.listdir(non_face_dir)
    total_files = len(face_files) + len(non_face_files)
    current = 0

    for folder, label in [(face_dir, 1), (non_face_dir, 0)]:
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)

            try:
                # Randomly pick a scale to simulate size variation
                scale = random.choice(MULTISCALE_SIZES)
                img = load_img(path, target_size=scale)
                img = img_to_array(img)

                # Resize back to model input size (64x64)
                img_resized = cv2.resize(img, TARGET_SIZE)

                data.append(img_resized)
                labels.append(label)

            except Exception as e:
                print(f"[WARNING] Could not load {path}: {e}")

            current += 1
            if current % 500 == 0 or current == total_files:
                print(f"[INFO] Loaded {current}/{total_files} images")

    data = np.array(data).astype('float32') / 255.0
    labels = np.array(labels)

    print(f"[INFO] Finished loading {len(data)} images.")
    return data, labels

def train_model():
    face_dir = 'face_patches'
    non_face_dir = 'nonface_patches'

    print("[INFO] Starting training process...")
    data, labels = load_data(face_dir, non_face_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    model = build_face_detector(input_shape=(64, 64, 3))  # Make sure this matches!

    print(f"[INFO] Training set size: {len(x_train)}")
    print(f"[INFO] Test set size: {len(x_test)}")
    print("[INFO] Compiling and training model...")

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=32,
        epochs=11,
        shuffle=True,
        verbose=1
    )

    model.save("face_detector_model.h5")
    print("[INFO] Model saved as face_detector_model.h5")

if __name__ == "__main__":
    train_model()
