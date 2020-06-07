timestamp=$(date -u "+%Y-%m-%d-%H:%M:%S.%N:%S3+00:00")

## downloading sample images.
ffmpeg \
    -i https://56f2a99952126.streamlock.net/833/default.stream/playlist.m3u8 \
    -vframes 1 \
    data/image_$timestamp.jpg
