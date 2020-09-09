# YOLO demo of Prinzipalmarkt

Object detection, applied to public images from the city of Münster.

https://cdn-images-1.medium.com/max/1600/1*Hz6t-tokG1niaUfmcysusw.jpeg

Webcam Example: https://codesandbox.io/s/z364noozrm

## Getting started

### Download some images

1. Open stream with mplayer (alternatively, use VLC player)

    ```bash
    cd data
    mplayer -fs -vf screenshot -playlist https://56f2a99952126.streamlock.net/833/default.stream/playlist.m3u8
    ```

2. Press 's' key to take screenshots.

### Object detection

1. Build and activate conda environment:

    ```bash
    conda env create -f environment.yml
    source activate traffic-cam
    ```

2. Download Yolo weights: 

    ```bash
    bash src/02_download_yolo.sh
    ```

3. Run Yolo demo:

    ```bash
    python -m src.03_yolo
    ```
