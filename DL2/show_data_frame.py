import cv2
import json
import os
import numpy as np

import video_process

json_file = "debug/video/3.json"
video_path = "videos/done/3.MP4"
frame_index = 510

frame = video_process.TakeFrameFromVideo(video_path, frame_index)

# read json_file
with open(json_file) as f:
    data = json.load(f)

# get the frame data
frame_data = data[f"{frame_index}"]

point1 = frame_data[0] # [721.1156008647902, 57.15926092326239]

# draw the point
cv2.circle(frame, (int(point1[0]), int(point1[1])), 5, (0, 0, 255), -1)

cv2.imshow("Frame", frame)
cv2.waitKey(0)












