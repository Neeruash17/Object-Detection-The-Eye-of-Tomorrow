!pip install opencv-python-headless==4.8.0.76
!pip install opencv-python==4.8.0.76
!pip install numpy==1.25.2

import cv2
from google.colab.patches import cv2_imshow
import time

# Load the object detection model
model_path = 'frozen_inference_graph.pb'
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
model = cv2.dnn_DetectionModel(model_path, config_path)

# Load class labels
with open('labels.txt', 'rt') as f:
    class_labels = f.read().rstrip('\n').split('\n')

# Model configuration
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))

# Initialize variables for tracking
person_count = 0
person_ids = {}  # Dictionary to store person IDs and their bounding boxes
next_person_id = 1
distance_threshold = 50  # Threshold for distance between centers

# Video capture
cap = cv2.VideoCapture('ty.mp4')  # Replace with your video file path
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # Use default camera if video file not found
if not cap.isOpened():
    raise IOError("Cannot open video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    class_ids, confidences, boxes = model.detect(frame, confThreshold=0.55)

    # Process detected objects
    for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
        if class_id == 1:  # Person class ID
            # Calculate center of the bounding box
            center_x = box[0] + box[2] // 2
            center_y = box[1] + box[3] // 2

            # Check if the person is already tracked
            matched_id = None
            for person_id, prev_box in list(person_ids.items()):  # Iterate over a copy for safe deletion
                prev_center_x = prev_box[0] + prev_box[2] // 2
                prev_center_y = prev_box[1] + prev_box[3] // 2

                # Calculate distance between centers
                distance = ((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)**0.5

                # If distance is within the threshold, consider it the same person
                if distance < distance_threshold:
                    matched_id = person_id
                    person_ids[person_id] = box  # Update the tracked person's box
                    break

            # If not tracked, assign a new ID
            if matched_id is None:
                matched_id = next_person_id
                next_person_id += 1
                person_count += 1  # Increment person count for new IDs
                person_ids[matched_id] = box

            # Draw bounding box and label with person ID
            cv2.rectangle(frame, box, (255, 0, 0), 2)
            label = f"Person {matched_id}"
            cv2.putText(frame, label, (box[0] + 10, box[1] + 40),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Display person count
    cv2.putText(frame, f"Total Persons: {person_count}", (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    # Display frame
    cv2_imshow(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()