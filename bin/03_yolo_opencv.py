"""
Counting persons with YOLO.
Based on: https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
"""

import numpy as np
from pathlib import Path
from traffic_cam import predictor


if __name__ == "__main__":
    try:
        predictor = predictor.Predictor()

        images = {
            "image_2020-09-09T20:13:29.985893+00:00s": 1,
            "image_2020-09-10T07:58:06.167747+00:00": 4,
            "image_2020-09-25T12:51:31.666045+00:00": 15,
            "image_2020-09-25T12:51:52.302745+00:00": 23,
            "image_2020-09-25T12:52:08.598229+00:00": 24,
            "image_2020-09-25T12:55:24.172202+00:00": 23,
            "image_2020-09-25T13:59:54.823420+00:00": 34,
            "image_2020-09-25T14:00:11.859519+00:00": 22,
            "image_2020-09-25T14:00:28.696451+00:00": 25,
            "image_2020-09-25T14:03:33.700749+00:00": 29,
        }

        squares = []
        for image, reference in images.items():
            print(f"Predicting {image} with {reference} people in it...")
            image_path = Path(
                f"data/train/north_street_petzhold/{image}.jpg"
            )
            try:
                prediction = predictor.predict_image(image_path=image_path, plot=True)
            except FileNotFoundError:
                print("File not found, skipping")
                continue
            error = reference - prediction
            print(f"Prediction error: {error}")
            squares += [error ** 2]

        nrmsd = np.mean(squares) ** 0.5 / sum(images.values())
        print(f"nrmsd: {nrmsd*100} %")
    except KeyboardInterrupt:
        print("Interrupted by user.")
