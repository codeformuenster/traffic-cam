"""Yolo object detection, labels from COCO dataset."""

import shutil
import subprocess
from pathlib import Path

IMG_PATH = Path("data/shot0004.jpg")
YOLO_CMD = [
    "./darknet/darknet",
    "detect",
    "darknet/cfg/yolov3.cfg",
    "data/yolo/yolov3.weights",
]

# run detection
command = YOLO_CMD + [str(IMG_PATH)]
result = subprocess.run(command, capture_output=True)
assert result.returncode == 0, "Object detection command failed."

# count persons in detection
stdout = result.stdout
lines = str(stdout).split("\\n")
person_count = sum([True if ("person" in line) else False for line in lines])
print(f"number of persons counted: {person_count}")

# move image to output folder
target = Path("output") / IMG_PATH.name
shutil.move("predictions.jpg", str(target))
