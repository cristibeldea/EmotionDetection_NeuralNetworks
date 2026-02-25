import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from tensorflow.keras.models import load_model
from multiprocessing import Pool
import tensorflow as tf

# ==== Config ====
MODEL_PATH = "SAVED_MODELS/face_detector_model.h5"
TEST_FOLDER = "testPerformance"
DETECTII_PERSONAL_FOLDER = "detectii_personal"
DETECTII_IMPORT_FOLDER = "detectii_import"
WINDOW_SIZES = [48, 64, 96, 128, 256, 528]
THRESHOLD = 0.9
MAX_PATCHES = 2000
BATCH_SIZE = 128
SAVE_IMAGES = True
NUM_PROCESSES = 4

# ==== Init ====
os.makedirs(DETECTII_PERSONAL_FOLDER, exist_ok=True)
os.makedirs(DETECTII_IMPORT_FOLDER, exist_ok=True)
model = load_model(MODEL_PATH, compile=False)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==== Utilitare ====
def non_max_suppression_strict(boxes, scores, min_overlap_ratio=0.2):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        suppress = [0]
        for pos in range(1, len(order)):
            j = order[pos]
            xx1 = max(boxes[i][0], boxes[j][0])
            yy1 = max(boxes[i][1], boxes[j][1])
            xx2 = min(boxes[i][2], boxes[j][2])
            yy2 = min(boxes[i][3], boxes[j][3])
            w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)
            inter = w * h
            if inter == 0:
                continue
            ratio = inter / min(areas[i], areas[j])
            if ratio >= min_overlap_ratio:
                suppress.append(pos)
        order = np.delete(order, suppress)
    return keep

def predict_in_batches(patches):
    results = []
    for i in range(0, len(patches), BATCH_SIZE):
        batch = np.array(patches[i:i+BATCH_SIZE])
        preds = model.predict(batch, batch_size=BATCH_SIZE, verbose=0)
        results.extend(preds)
    return results

def detect_faces_custom_model(image):
    height, width = image.shape[:2]
    all_boxes, all_scores = [], []

    for ws in WINDOW_SIZES:
        if ws > height or ws > width:
            continue
        stride = ws // 4
        patches, positions = [], []

        for y in range(0, height - ws + 1, stride):
            for x in range(0, width - ws + 1, stride):
                patch = image[y:y+ws, x:x+ws]
                resized = cv2.resize(patch, (64, 64)).astype("float32") / 255.0
                patches.append(resized)
                positions.append((x, y, ws))
                if len(patches) >= MAX_PATCHES:
                    break
            if len(patches) >= MAX_PATCHES:
                break

        if not patches:
            continue

        preds = predict_in_batches(patches)
        for conf, (x, y, ws) in zip(preds, positions):
            if conf[0] >= THRESHOLD:
                all_boxes.append((x, y, x + ws, y + ws))
                all_scores.append(conf[0])

    keep = non_max_suppression_strict(all_boxes, all_scores)
    return [all_boxes[i] for i in keep]

def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]

def center_inside(box, ref_box):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return ref_box[0] <= cx <= ref_box[2] and ref_box[1] <= cy <= ref_box[3]

def draw_boxes(image, boxes, color=(0, 255, 0)):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

def process_image(filename):
    img_path = os.path.join(TEST_FOLDER, filename)
    image = cv2.imread(img_path)
    if image is None:
        return (0, 0, 0, 0)

    if image.shape[0] < 128 or image.shape[1] < 128:
        image = cv2.resize(image, (128, 128))

    custom_boxes = detect_faces_custom_model(image)
    haar_boxes = detect_faces_haar(image)

    TP = FP = FN = TN = 0

    if SAVE_IMAGES:
        img_custom = draw_boxes(image.copy(), custom_boxes, (255, 0, 0))
        img_haar = draw_boxes(image.copy(), haar_boxes, (0, 255, 0))
        cv2.imwrite(os.path.join(DETECTII_PERSONAL_FOLDER, filename), img_custom)
        cv2.imwrite(os.path.join(DETECTII_IMPORT_FOLDER, filename), img_haar)

    if len(custom_boxes) == 0 and len(haar_boxes) == 0:
        TN += 1
    else:
        matched = [False] * len(custom_boxes)
        for haar_box in haar_boxes:
            found = False
            for idx, custom_box in enumerate(custom_boxes):
                if center_inside(custom_box, haar_box) and not matched[idx]:
                    matched[idx] = True
                    found = True
                    break
            if not found:
                FN += 1
        FP += matched.count(False)
        TP += matched.count(True)

    return (TP, TN, FP, FN)

def evaluate_models():
    files = [f for f in os.listdir(TEST_FOLDER) if f.endswith((".jpg", ".png"))]
    start_time = time.time()

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(process_image, files), total=len(files), desc="Evaluare imagini"))

    TP = sum(r[0] for r in results)
    TN = sum(r[1] for r in results)
    FP = sum(r[2] for r in results)
    FN = sum(r[3] for r in results)
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0

    print("\n--- Rezultate Finale ---")
    print(f"True Positives: {TP}")
    print(f"True Negatives: {TN}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Durata totală: {time.time() - start_time:.2f} secunde")

if __name__ == "__main__":
    evaluate_models()
