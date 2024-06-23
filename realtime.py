import cv2
import numpy as np

# Load pre-trained YOLOv4 model and configuration files
net = cv2.dnn.readNet("darknet/yolov4-5000.weights", "darknet/yolov4.cfg")

# Load COCO class labels
with open("darknet/yolov4.names", "r") as f:
    classes = f.read().strip().split("\n")

# Function to detect objects in a frame
def detect_objects(frame):
    # Get original input frame size
    original_h, original_w = frame.shape[:2]

    # Forward pass through the network
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # List to store valid bounding boxes
    valid_boxes = []

    # Loop over each of the layer outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0:
                # Scale the bounding box coordinates back to the original frame
                box = detection[0:4] * np.array([original_w, original_h, original_w, original_h])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculate top-left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                valid_boxes.append((x, y, int(width), int(height), confidence, classID))

    return valid_boxes

# Function to calculate intersection over union (IoU)
def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[0] + box1[2], box2[0] + box2[2])
    yB = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] + 1) * (box1[3] + 1)
    box2_area = (box2[2] + 1) * (box2[3] + 1)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

# Path to input video
input_video_path = "images/video3.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Loop to process each frame
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # List to store bounding boxes that have been drawn
    drawn_boxes = []

    # Detect objects in the frame
    detections = detect_objects(frame)

    # Draw bounding boxes and labels on the original frame
    for (x, y, width, height, confidence, classID) in detections:
        # Flag to indicate if the current box intersects with any drawn box
        intersects = False

        # Check if the current box intersects with any drawn box
        for (drawn_x, drawn_y, drawn_width, drawn_height, _, _) in drawn_boxes:
            if calculate_iou((x, y, width, height), (drawn_x, drawn_y, drawn_width, drawn_height)) > 0.5:
                intersects = True
                break

        # If the current box does not intersect with any drawn box, draw it
        if not intersects:
            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            # Draw label
            text = "{}: {:.4f}".format(classes[classID], confidence)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add the current box to the list of drawn boxes
            drawn_boxes.append((x, y, width, height, confidence, classID))

    # Resize frame to 1024x768
    frame_resized = cv2.resize(frame, (800, 600))

    # Display object counts
    text_offset_x = 10
    text_offset_y = frame_resized.shape[0] - 30

    object_counts = {}
    for _, _, _, _, confidence, classID in drawn_boxes:
        class_name = classes[classID]
        object_counts[class_name] = object_counts.get(class_name, 0) + 1

    for i, (class_name, count) in enumerate(object_counts.items()):
        cv2.putText(frame_resized, f"Number of {class_name}: {count}", (text_offset_x, text_offset_y - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the output frame
    cv2.imshow("Detected Objects", frame_resized)
    
    # Press 'q' to exit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close OpenCV window
cv2.destroyAllWindows()
