import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = "SAVED_MODELS/face_detector_model.h5"
WINDOW_SIZE = 64
STRIDE = 16
THRESHOLD = 0.9

model = load_model(MODEL_PATH, compile=False)

def non_max_suppression_strict(boxes, scores, min_overlap_ratio=0.2):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = areas.argsort()[::-1]  # largest area first

    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        suppress = [0]

        for pos in range(1, len(order)):
            j = order[pos]

            # compute intersection
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

def detect_faces(image, model, threshold=THRESHOLD, stride=STRIDE, window_sizes=[128, 256, 512, 1024, 2048, 3000, 4000]):
    height, width = image.shape[:2]

    all_boxes = []
    all_scores = []

    for window_size in window_sizes:
        patches = []
        positions = []

        for y in range(0, height - window_size, stride):
            for x in range(0, width - window_size, stride):
                patch = image[y:y + window_size, x:x + window_size]
                patch = cv2.resize(patch, (64, 64))
                patch = patch.astype("float32") / 255.0
                patch = patch.reshape(64, 64, 3)  # this is now valid
                patches.append(patch)
                positions.append((x, y, window_size))  # Store window size too

        if not patches:
            continue

        patches = np.array(patches)
        preds = model.predict(patches, batch_size=128, verbose=0)

        for (conf, (x, y, ws)) in zip(preds, positions):
            if conf[0] >= threshold:
                all_boxes.append((x, y, x + ws, y + ws))
                all_scores.append(conf[0])

    # Final NMS across all scales
    keep_indices = non_max_suppression_strict(all_boxes, all_scores)
    final_boxes = [all_boxes[i] for i in keep_indices]

    return final_boxes

def draw_boxes(image, boxes):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

if __name__ == "__main__":
    image = cv2.imread("test.jpg")
    detections = detect_faces(image, model)
    output = draw_boxes(image, detections)
    display_image = cv2.resize(output, (800, 800))  # or fx=0.5, fy=0.5
    cv2.imshow("Detected Faces", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
