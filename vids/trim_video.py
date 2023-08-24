#!/usr/bin/env ipython
import cv2
from tqdm import trange
import numpy as np
import skvideo.io
import sys

# Specify the path to the input video
input_path = sys.argv[1]

# Specify the path to the output video
output_path = sys.argv[2]

print(output_path)

# Open the input video
video = cv2.VideoCapture(input_path)

# Get the properties of the input video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a VideoWriter object to write the output video
writer = skvideo.io.FFmpegWriter(output_path, inputdict={
    '-framerate': str(fps),
}, outputdict={
    '-vcodec': 'h264',
    '-pix_fmt': 'yuv420p', # to support more players
    '-vf': 'pad=ceil(iw/2)*2:ceil(ih/2)*2' # to handle width/height not divisible by 2
})


# trim parameters
top, bot, left, right = [int(x) for x in sys.argv[3:7]]

if length < 10:
    iterator = range(10000)
else:
    iterator = trange(length, ncols=70)

for _ in iterator:
    # Read a frame from the input video
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = frame[top:-bot-1, left:-right-1]

    # Write the frame with text to the output video
    writer.writeFrame(frame)

# Release video resources
video.release()
writer.close()
