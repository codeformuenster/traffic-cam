'''Yolo object detection, labels from COCO dataset.'''

import lightnet
from collections import Counter

# load yolo model
model = lightnet.load('yolo')  # to download net: 'python -m lightnet download yolo'

# object detection
image_file = 'data/2018-03-20T19:27:26+0000.png'
image = lightnet.Image.from_bytes(
    open(image_file, 'rb').read())
boxes = model(image, thresh=0.1)

# count boxes
objects = [e[1] for e in boxes]
obj_count = dict(Counter(objects))
print('Objects found in image: {}'.format(obj_count))
