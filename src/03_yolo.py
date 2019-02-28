'''Yolo object detection, labels from COCO dataset.'''

from collections import Counter

import cv2
from pydarknet import Detector, Image
from pprint import pprint


# load yolo model
net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"),
               bytes("weights/yolov3.weights", encoding="utf-8"),
               0,
               bytes("cfg/coco.data",encoding="utf-8"))

# object detection
img = cv2.imread('data/2018-03-20T19:27:26+0000.png')
img_darknet = Image(img)

# report results
results = net.detect(img_darknet)
pprint(results)
