#!/usr/local/bin/python

import lightnet

model = lightnet.load('yolo')
image = lightnet.Image.from_bytes(open('angela.jpg', 'rb').read())
boxes = model(image)

items = { 'person': 2 }
for box in boxes:
    item = box[1]
    if not item in items.keys():
        items[item] = 1
    else:
        items[item] = items[item] + 1

print(items)
