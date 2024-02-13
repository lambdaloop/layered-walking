#!/usr/bin/env ipython
import cv2
from tqdm import trange
from PIL import Image, ImageDraw, ImageFont
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


# Path to the font file
font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'

# Font size and color
font_size = 40
font_color = (0, 0, 0)

# texts = ['8 mm/s forward\ntest', '10 mm/s forward',
#          '12 mm/s forward', '14 mm/s forward']

text_header = sys.argv[3].replace('|', '\n')
texts = [x.replace('|', '\n') for x in sys.argv[4:]]

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

    # Convert the frame to PIL Image
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)

    # Write the styled text on the frame
    font = ImageFont.truetype(font_path, font_size)
    width_1 = frame.shape[1] // 4
    if frame.shape[0] > 1000:
        height_1 = 600 # header of 2nd row
    else:
        height_1 = 550 # bottom of 1st row

    for i, text in enumerate(texts):
        text_width, text_height = draw.textsize(text, font)
        text_x = int((width_1 - text_width) / 2) + (i % 4) * width_1
        text_y = 160 + (i//4) * height_1
        draw.text((text_x, text_y), text, font=font, fill=font_color, align='center')

    text_width, text_height = draw.textsize(text_header, font)
    text_x = int((frame.shape[1] - text_width) / 2)
    text_y = 20
    draw.text((text_x, text_y), text_header, font=font, fill=font_color, align='center')

    # Convert the modified image back to a NumPy array
    frame = np.array(frame_pil)

    # Write the frame with text to the output video
    writer.writeFrame(frame)

# Release video resources
video.release()
writer.close()
