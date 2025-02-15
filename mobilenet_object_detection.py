import tensorflow as tf
import numpy as np
import cv2
import tensorflow_hub as hub

# Load SSD MobileNet V2 model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Load COCO label names
labels = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

def detect_objects(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (320, 320))
    img_tensor = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize

    # Run object detection
    results = model(img_tensor)
    boxes = results["detection_boxes"].numpy()
    classes = results["detection_classes"].numpy()
    scores = results["detection_scores"].numpy()

    # Draw bounding boxes
    for i in range(len(scores[0])):
        # if scores[0][i] > 0.5:  # Confidence threshold
        y1, x1, y2, x2 = boxes[0][i]
        class_id = int(classes[0][i])
        label = labels.get(class_id, "Unknown")
        print(f"Detected {label} with confidence {scores[0][i]:.2f}")

        # Draw bounding box on the original image
        h, w, _ = img.shape
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)
        cv2.putText(img, f"{label}: {scores[0][i]:.2f}", (int(x1 * w), int(y1 * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with bounding boxes
    cv2.imwrite("output.jpg", img)
    print("Output image saved as output.jpg")

detect_objects(r"/workspaces/image_analytics/test.jpg")
print('completed')