#!/usr/bin/env bash
# based on CLI instructions at https://pjreddie.com/darknet/yolo/

# setup darknet
git clone https://github.com/pjreddie/darknet
cd darknet
make
cd ..

# download model
wget -c -O data/yolo/yolov3.weights https://pjreddie.com/media/files/yolov3.weights

# copy artifacts to run detection with yolo
mkdir -p cfg
cp darknet/cfg/coco.data cfg/.
cp darknet/data/coco.names data/.
cp -r darknet/data/labels data/.
