import cv2
from ultralytics import YOLO
import numpy as np

#model menggunakan YOLOv8
model = YOLO("yolov8n.pt")


# Mengubah sesuai yang digunakan. Webcam / Video
webcam = cv2.VideoCapture('TestVid1.mp4')

if not webcam.isOpened():
    print("Could not open video file / webcam")
    exit()


centre_prev = {}
drowning_counters = {}
isDrowning = {}
movement_history = {}
frame_rate = 30


movement_threshold = 15
aspect_ratio_threshold = 1.2
drowning_frames = frame_rate * 15 
rapid_movement_threshold = 25
history_length = 40 
resize_width = 640
resize_height = 480 
frame_skip = 2
delay = int(1000 / frame_rate)
current_frame = 0


class_list = []
with open("utils/coco.txt", "r") as file:
    class_list = file.read().strip().split("\n")

def draw_bbox(image, boxes, labels, confidences, drowning_status):
    for box, label, conf, drowning in zip(boxes, labels, confidences, drowning_status):
        color = (0, 0, 255) if drowning else (0, 255, 0)  # Merah jika tenggelam, Hijau jika tidak
        cv2.rectangle(image, (int(box[0]), int(box[1])), 
                      (int(box[2]), int(box[3])), color, 2)
        cv2.putText(image, f"{label} {conf:.2f}", 
                    (int(box[0]), int(box[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        print("Finished processing video")
        break

    current_frame += 1
    if current_frame % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (resize_width, resize_height))

    results = model(frame)
    detections = results[0]

    bbox = []
    labels = []
    conf = []

    for det in detections.boxes:
        cls = int(det.cls[0])
        confidence = float(det.conf[0])
        label = class_list[cls]

        if label != 'person':
            continue

        coordinates = det.xyxy[0].cpu().numpy()
        bbox.append(coordinates)
        labels.append(label)
        conf.append(confidence)

    current_drowning = {}

    for i, lbl in enumerate(labels):
        if lbl == 'person':
            box = bbox[i]
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            width = box[2] - box[0]
            height = box[3] - box[1]
            aspect_ratio = height / width if width != 0 else 0

            if i not in centre_prev:
                centre_prev[i] = centre
                drowning_counters[i] = 0
                isDrowning[i] = False
                movement_history[i] = []
                continue

            hmov = abs(centre[0] - centre_prev[i][0])
            vmov = abs(centre[1] - centre_prev[i][1])

            movement_history[i].append((hmov, vmov))
            if len(movement_history[i]) > history_length:
                movement_history[i].pop(0)

            avg_hmov = np.mean([m[0] for m in movement_history[i]])
            avg_vmov = np.mean([m[1] for m in movement_history[i]])
            total_avg_movement = avg_hmov + avg_vmov

            if total_avg_movement > rapid_movement_threshold:
                isDrowning[i] = True
            elif hmov > movement_threshold or vmov > movement_threshold:
                drowning_counters[i] = 0
                isDrowning[i] = False
            else:
                drowning_counters[i] += 1
                if drowning_counters[i] >= drowning_frames and aspect_ratio > aspect_ratio_threshold:
                    isDrowning[i] = True

            centre_prev[i] = centre

            current_drowning[i] = isDrowning[i]

    detected_indices = [i for i, lbl in enumerate(labels) if lbl == 'person']
    to_remove = [key for key in centre_prev.keys() if key not in detected_indices]
    for key in to_remove:
        del centre_prev[key]
        del drowning_counters[key]
        del isDrowning[key]
        del movement_history[key]

    out = draw_bbox(frame.copy(), bbox, labels, conf, 
                   [current_drowning.get(i, False) for i in range(len(labels)) if labels[i] == 'person'])

    cv2.imshow("Drowning Detection", out)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()