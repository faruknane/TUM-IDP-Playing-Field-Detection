import cv2
import numpy as np
import os
import dataset
import torch
import training
import field_inference
import model as models

config = {
    "model": "resnet18_customized",
    "model_path":"saved_models/model_epoch18.pth",
    "image_size": 720,
    "image_path": r"input.png",
    "frame_rate": 24,
}

field_types = ["tennis", "football", "frisbee"]
field_types.sort()
field_inference.SetFieldTypes(field_types)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.get_model(config["model"], class_count=len(field_types))
model.to(device)
model.eval()

# load model weights
model.load_state_dict(torch.load(config["model_path"]))

# load image
frame = cv2.imread(config["image_path"], cv2.IMREAD_COLOR)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

input = cv2.resize(frame, (config["image_size"], config["image_size"]))
input = dataset.ToTensor.PreprocessImage(input, False) # input is numpy array [3, H, W]

# make input torch array
input = torch.from_numpy(input).to(device)

# forward pass
prediction, class_prediction = model(input.unsqueeze(0))

# to numpy
class_prediction = torch.argmax(class_prediction, dim=1) # [N]
class_prediction = class_prediction.cpu().detach().numpy()

predicted_points, Ms = training.get_predicted_points(prediction, class_prediction)

dlines = field_inference.get_playing_field_lines(class_prediction[0], Ms[0])

imageH = frame.shape[0]
imageW = frame.shape[1]


# frame = np.zeros_like(frame)

for dline in dlines:
    cv2.line(frame, (int(dline[0][0]*imageW), int(dline[0][1]*imageH)), (int(dline[1][0]*imageW), int(dline[1][1]*imageH)), (0, 255, 0), 2)

# write as result.png
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imwrite("result.png", frame)



