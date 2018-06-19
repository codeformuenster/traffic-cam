# REGION SELECTION

import selectivesearch
import cv2
import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


img = cv2.imread('data/2018-03-20T19:27:17+0000.png')
pylab.imshow(img)
pylab.show()

# selective search algorithm
img2, regions = \
    selectivesearch.selective_search(img, scale=500, sigma=0.8, min_size=200)

# filter regions
candidates = set()
for r in regions:
    # excluding same rectangle (with different segments)
    if r['rect'] in candidates:
        continue
    # excluding regions smaller than 2000 pixels
    if r['size'] < 100:
        continue
    # distorted rects
    x, y, w, h = r['rect']
    if h is 0 or w is 0:
        continue
    if w / h > 3 or h / w > 3:
        continue
    candidates.add(r['rect'])

# draw rectangles on the original image
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img)
for x, y, w, h in candidates:
    print(x, y, w, h)
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

plt.show()

