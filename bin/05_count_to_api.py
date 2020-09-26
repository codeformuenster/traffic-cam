"""Count persons on Prinzipalmarkt and publish to API. """

import requests
from traffic_cam import classifier, io, predictor, paths
from time import sleep


def loop(classifier_model, predictor_model):
    # download image
    path = io.download_image()

    # classify image location
    location = classifier.move_image_by_class(source_path=path, model=classifier_model)

    # if image location not useful, wait and download new image
    if "street" not in location:
        print("Not a street, skipping.")
        return False

    # count persons on image
    persons = predictor_model.predict_image(image_path=paths.TRAIN_DIR / location / path.name, plot=True)

    # collect additional information for API
    rain = True  # 90 % accuracy for Münster, good enough

    # build response
    response = {
        "count": persons,
        "timestamp": io.get_timestamp_isoformat(),
        "device_id": location,
        "data": {
            "rain": rain,
        }
    }
    print(response)

    # send count to API
    response = requests.post(
        "https://counting-backend.codeformuenster.org/counts/",
        json=response,
    )


if __name__ == "__main__":
    cl = classifier.get_classifier_model()
    pr = predictor.Predictor()
    while True:
        try:
            loop(classifier_model=cl, predictor_model=pr)
            sleep(10)
        except KeyboardInterrupt:
            print("Interrupted by user.")
