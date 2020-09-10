# Observing traffic in the City of Muenster

Image classification and object detection, applied to a public webcam in the city of Münster.

## Dependencies

* Debian-based operating system
* ffmpeg (e.g. install with apt-get)

## Getting started

1. Create and activate virtual environment. For example with `conda`:

    ```bash
    conda env create -n cam python=3.7
    conda activate cam
    ```

2. Install Python dependencies and `traffic-cam` package (with virtual environment activated):

    ```bash
    pip install -r requirements.txt
    pip install .
    ```

3. Run scripts in `bin/` in their numerical order.
    This includes downloading images, training a classifier, and counting persons.
