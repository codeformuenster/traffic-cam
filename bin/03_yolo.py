"""Yolo object detection, labels from COCO dataset."""
# %%
from pprint import pprint

import cv2
from matplotlib import pyplot as plt
from pydarknet import Detector, Image

# %% load yolo model
net = Detector(
    bytes("cfg/yolov3.cfg", encoding="utf-8"),
    bytes("weights/yolov3.weights", encoding="utf-8"),
    0,
    bytes("cfg/coco.data", encoding="utf-8"),
)

# %% object detection
img = cv2.imread("data/shot0001.png")
plt.imshow(img)

# %% loading image into library
print(img.shape)

# %% report results (slow on large image)
img_darknet = Image(img)
results = net.detect(img_darknet)
pprint(results)

# %%
print("What types of objects are seen on the image:")
set([i[0] for i in results])


# %% showing detected boxes, highlighting backpacks

img2 = img.copy()

for category, score, bounds in results:
    x, y, w, h = bounds
    if category == b"backpack":
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    cv2.rectangle(
        img2,
        (int(x - w / 2), int(y - h / 2)),
        (int(x + w / 2), int(y + h / 2)),
        color,
        thickness=2,
    )

plt.rcParams["figure.figsize"] = [20, 10]
plt.imshow(img2)

# %%
