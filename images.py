import os
import cv2
import numpy as np

# Load pre-trained YOLOv4 model and configuration files
net = cv2.dnn.readNet("darknet/yolov4-5000.weights", "darknet/yolov4.cfg")

# Load COCO class labels
with open("darknet/yolov4.names", "r") as f:
    classes = f.read().strip().split("\n")

# Function to detect objects in an image
def detect_objects(image):
    # Get original input image size
    original_h, original_w = image.shape[:2]

    # Forward pass through the network
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
                # Scale the bounding box coordinates back to the original image
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

# Directory containing input images
input_dir = "images/"

# Create a list of input image paths
input_image_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)
                     if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Loop to process each image
for input_image_path in input_image_paths:
    # Load input image
    image = cv2.imread(input_image_path)

    # List to store bounding boxes that have been drawn
    drawn_boxes = []

    # Detect objects in the image
    original_h, original_w = image.shape[:2]
    detections = detect_objects(image)

    # Draw bounding boxes and labels on the original image
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
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            # Draw label
            text = "{}: {:.4f}".format(classes[classID], confidence)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add the current box to the list of drawn boxes
            drawn_boxes.append((x, y, width, height, confidence, classID))

    # Resize image to 800x600
    image_resized = cv2.resize(image, (1024, 768))

    # Display object counts
    text_offset_x = 10
    text_offset_y = image_resized.shape[0] - 30

    object_counts = {}
    for _, _, _, _, confidence, classID in drawn_boxes:
        class_name = classes[classID]
        object_counts[class_name] = object_counts.get(class_name, 0) + 1

    for i, (class_name, count) in enumerate(object_counts.items()):
        cv2.putText(image_resized, f"Number of {class_name}: {count}", (text_offset_x, text_offset_y - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the output image
    cv2.imshow("Detected Objects", image_resized)
    cv2.waitKey(0)

# Close OpenCV window
cv2.destroyAllWindows()
