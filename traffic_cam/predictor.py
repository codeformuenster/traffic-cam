from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from traffic_cam import paths


class Predictor:

    def __init__(self):
        # read class names from text file
        with open(str(paths.YOLO_CLASSES), "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Check if the YOLO weights and cfg was downloaded
        if not paths.YOLO_WEIGHTS.is_file():
            print("Download YOLO Weights")
            raise FileNotFoundError()

        if not paths.YOLO_CFG.is_file():
            print("Download YOLO CFG")
            raise FileNotFoundError()

        # read pre-trained model and config file
        self.net = cv2.dnn.readNet(str(paths.YOLO_WEIGHTS), str(paths.YOLO_CFG))

        # scaling to image values within [0, 1]
        self.scale = 1 / 255

    def predict_image(self, image_path: Path, plot: bool = False) -> int:
        if not image_path.is_file():
            raise FileNotFoundError()

        # read input image
        image = cv2.imread(str(image_path))

        # load YOLO model
        blob = cv2.dnn.blobFromImage(
            image=image,
            scalefactor=self.scale,
            size=(800, 800),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = self.net.forward(self.get_output_layers(self.net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.3
        nms_threshold = 0.4

        # dimensions of original images (for drawing bounding boxes)
        Width = image.shape[1]
        Height = image.shape[0]

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
            self.draw_bounding_box(
                image,
                class_ids[i],
                confidences[i],
                round(x),
                round(y),
                round(x + w),
                round(y + h),
            )

        print(f"Number on persons: {person_count}")

        if plot is True:
            self.plot_image(image, image_path)

        return person_count

    def get_output_layers(self, net):
        """get the output layer names in the architecture
        Args:
            net ([type]): [description]
        Returns:
            [type]: [description]
        """
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
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
        label = str(self.classes[class_id])
        color = self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def plot_image(self, image, image_path):
        # save image
        plt.axis("off")
        plt.imshow(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="bicubic", aspect="auto"
        )
        plt.savefig(paths.OUTPUT_DIR / image_path.name)
