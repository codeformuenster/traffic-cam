# Observing traffic in the City of Muenster

Image classification and object detection, applied to a public webcam in the city of Münster.

See webcam data source [here](https://www.blick.ms/webcam-auf-dem-prinzipalmarkt-muenster.php).


## Dependencies

* Tested on `Ubuntu 20.04`
* `ffmpeg`, e.g. on Ubuntu install with:

```bash
sudo apt install ffmpeg
```

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
