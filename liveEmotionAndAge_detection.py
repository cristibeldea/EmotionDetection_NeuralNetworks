import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model
import time
from deepface import DeepFace

model = load_model("face_detector_model.h5", compile=False)

AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

WINDOW_SIZE = 64
STRIDE = 32
THRESHOLD = 0.9
FRAME_WIDTH = 640
ANALYZE_INTERVAL = 2.0

last_analysis_time = 0
face_labels = []

def non_max_suppression_strict(boxes, scores, min_overlap_ratio=0.2):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = areas.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        suppress = [0]
        for pos in range(1, len(order)):
            j = order[pos]
            xx1 = max(boxes[i][0], boxes[j][0])
            yy1 = max(boxes[i][1], boxes[j][1])
            xx2 = min(boxes[i][2], boxes[j][2])
            yy2 = min(boxes[i][3], boxes[j][3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            if inter == 0:
                continue
            smaller_area = min(areas[i], areas[j])
            overlap_ratio = inter / smaller_area
            if overlap_ratio >= min_overlap_ratio:
                suppress.append(pos)
        order = np.delete(order, suppress)
    return keep

def detect_faces(image):
    height, width = image.shape[:2]
    all_boxes = []
    all_scores = []

    for window_size in [64, 128, 200, 256, 350, 500]:
        patches = []
        positions = []

        for y in range(0, height - window_size + 1, STRIDE):
            for x in range(0, width - window_size + 1, STRIDE):
                patch = image[y:y + window_size, x:x + window_size]
                patch_resized = cv2.resize(patch, (64, 64))
                patch_resized = patch_resized.astype("float32") / 255.0
                patches.append(patch_resized)
                positions.append((x, y, window_size))

        if not patches:
            continue

        patches = np.array(patches)
        preds = model.predict(patches, batch_size=128, verbose=0)

        for (conf, (x, y, ws)) in zip(preds, positions):
            if conf[0] >= THRESHOLD:
                all_boxes.append((x, y, x + ws, y + ws))
                all_scores.append(conf[0])

    keep_indices = non_max_suppression_strict(all_boxes, all_scores)
    final_boxes = [all_boxes[i] for i in keep_indices]
    return final_boxes

def predict_age(face_crop):
    try:
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        i = preds[0].argmax()
        return AGE_BUCKETS[i]
    except Exception as e:
        print(f"[WARNING] Age prediction failed: {e}")
        return "unknown"

def analyze_face(face_crop):
    try:
        h, w = face_crop.shape[:2]
        if h < 50 or w < 50:
            print("[WARNING] Face crop too small, skipping.")
            return "unknown", "unknown"

        resized_face = cv2.resize(face_crop, (224, 224))

        # Emotion prediction
        result = DeepFace.analyze(resized_face, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        emotion = result['dominant_emotion']

        # Age prediction (fast lightweight)
        age = predict_age(face_crop)

        return emotion, age

    except Exception as e:
        print(f"[WARNING] Face analysis failed: {e}")
        return "unknown", "unknown"

def draw_boxes(image, boxes):
    global last_analysis_time, face_labels

    now = time.time()

    if now - last_analysis_time > ANALYZE_INTERVAL:
        new_labels = []
        for (x1, y1, x2, y2) in boxes:
            face_crop = image[y1:y2, x1:x2]
            emotion, age = analyze_face(face_crop)
            new_labels.append((emotion, age))
        face_labels = new_labels
        last_analysis_time = now

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if idx < len(face_labels):
            emotion, age = face_labels[idx]
            label = f"{emotion} {age}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return image

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        threading.Thread(target=self.update, args=(), daemon=True).start()

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


detected_boxes = []
last_frame = None
lock = threading.Lock()

def detection_thread():
    global detected_boxes, last_frame
    while True:
        if last_frame is not None:
            frame_copy = last_frame.copy()
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            boxes = detect_faces(rgb_frame)
            with lock:
                detected_boxes = boxes
        time.sleep(0.2)

def main():
    global last_frame
    vs = VideoStream()
    time.sleep(1.0)

    threading.Thread(target=detection_thread, args=(), daemon=True).start()

    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
        with lock:
            last_frame = frame
            boxes = detected_boxes.copy()

        output_frame = draw_boxes(frame.copy(), boxes)

        cv2.imshow("Face Detector", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
