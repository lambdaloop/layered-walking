#!/usr/bin/env python3

import numpy as np
import cv2
from tqdm import trange
import sys
import os.path
import skvideo.io

fnames = sys.argv[1:-1]
outname = sys.argv[-1]

def get_props(cap):
    out = dict()
    out['length'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out['fps'] = float(cap.get(cv2.CAP_PROP_FPS))
    return out

props = []
caps = []
for fname in fnames:
    cap = cv2.VideoCapture(fname)
    caps.append(cap)
    prop = get_props(cap)
    props.append(prop)


width = max([p['width'] for p in props])
height = sum([p['height'] for p in props])
length = min([p['length'] for p in props])
fps = props[0]['fps']
fps = round(fps, 2)
print(fps)

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# writer = cv2.VideoWriter(outname,fourcc, props[0]['fps'], (width,height))

writer = skvideo.io.FFmpegWriter(outname, inputdict={
    '-framerate': str(fps),
}, outputdict={
    '-vcodec': 'h264',
    '-pix_fmt': 'yuv420p', # to support more players
    '-vf': 'pad=ceil(iw/2)*2:ceil(ih/2)*2' # to handle width/height not divisible by 2
})

keep_going = True

if length < 10:
    iterator = range(10000)
else:
    iterator = trange(length, ncols=70)
for i in trange(length):
    frame_out = np.zeros((height, width, 3), dtype='uint8')
    frame_out[:] = 255

    x = 0
    for j, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            keep_going = False
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        w0 = int((width - w) / 2)
        frame_out[x:(x+h),w0:(w0+w)] = frame
        x += h
        
    if keep_going:
        writer.writeFrame(frame_out)
    else:
        break

# Release everything if job is finished
for cap in caps:
    cap.release()
writer.close()
