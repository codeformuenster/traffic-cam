"""
Counting persons with YOLO.
Based on: https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from traffic_cam import paths

# read input image
# TODO: download new image, within which to count people
image_path = Path("data/train/south_street_cityhall/image_2020-09-10T07:43:44.868929+00:00.jpg")
image = cv2.imread(str(image_path))

# dimensions of original images (for drawing bounding boxes)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392  # TODO: what does this parameter tune?

# read class names from text file
classes = None
with open(str(paths.YOLO_CLASSES), "r") as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(str(paths.YOLO_WEIGHTS), str(paths.YOLO_CFG))

# load YOLO model
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)


def get_output_layers(net):
    """get the output layer names in the architecture
    Args:
        net ([type]): [description]
    Returns:
        [type]: [description]
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """draw bounding box on the detected object with class name
    Args:
        img ([type]): [description]
        class_id ([type]): [description]
        confidence ([type]): [description]
        x ([type]): [description]
        y ([type]): [description]
        x_plus_w ([type]): [description]
        y_plus_h ([type]): [description]
    """
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# run inference through the network
# and gather predictions from output layers
outs = net.forward(get_output_layers(net))

# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.3
nms_threshold = 0.4

# for each detetion from each output layer
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# go through the detections remaining
# after nms and draw bounding box
person_count = 0

for i in indices.flatten():
    # draw for persons only
    if class_ids[i] != 0:
        continue
    # increment person count
    person_count += 1
    # draw bounding box
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_bounding_box(
        image,
        class_ids[i],
        confidences[i],
        round(x),
        round(y),
        round(x + w),
        round(y + h),
    )

print(f"Number on persons: {person_count}")

# save image
plt.axis("off")
plt.imshow(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="bicubic", aspect="auto"
)
plt.savefig(paths.OUTPUT_DIR / image_path.name)
