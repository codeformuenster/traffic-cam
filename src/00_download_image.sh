#!/bin/sh
## downloading sample images.

ffmpeg \
    -i https://56f2a99952126.streamlock.net/833/default.stream/playlist.m3u8 \
    -r 0.2 \
    -vframes 4 \
    data/image%03d.jpg
