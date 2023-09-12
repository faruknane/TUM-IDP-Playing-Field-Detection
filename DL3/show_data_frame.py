import cv2
import json
import os
import numpy as np

import video_process

json_file = r"C:\Users\FarukNane\Desktop\DL2\data\train\labels\3.json"
video_path = r"C:\Users\FarukNane\Desktop\DL2\data\train\videos\3.MP4"
frame_index = 10

frame = video_process.TakeFrameFromVideo(video_path, frame_index)

# read json_file
with open(json_file) as f:
    data = json.load(f)

# get the frame data
frame_data = data[f"{frame_index}"]

for point in frame_data:
    # draw the point
    cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

# resize frame to width = 1800
frame = cv2.resize(frame, (1600, 1600 * frame.shape[0] // frame.shape[1]))

cv2.imshow("Frame", frame)
cv2.waitKey(0)












