# DOWNLOAD IMAGE FROM WEBSITE

n <- 10
sleep <- 5

# download n images
for (i in 1:n) {
  print(paste("Downloading image", i))
  # download image
  timestamp <- strftime(as.POSIXlt(Sys.time(), "UTC"), "%Y-%m-%dT%H:%M:%S%z")
  url <- 'https://56f2a99952126.streamlock.net/833/default.stream/playlist.m3u8'
  command <- paste0("ffmpeg -i ", 
                    url,
                    " -ss 5 -frames:v 1 data/",
                    timestamp,
                    ".png")
  system(command)
  # sleep
  Sys.sleep(sleep)
}

