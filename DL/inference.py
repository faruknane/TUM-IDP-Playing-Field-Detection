import torch
import torchvision
from torchvision import models 
import torch.nn as nn
import glob
import os
import pathlib
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import dataloader
from dataloader import HomographyDataset, HorizontalFlip, RandomCrop, ResizeImage, CalculateTargetKeyPoints, ToTensor
import random
from model import get_model
import data_processing

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":

    # read template.png rgb
    template = cv2.imread("template.png")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model()

    # load model weights from pth
    model.load_state_dict(torch.load("saved_models_2/model_epoch33.pth"))

    # send model to device
    model = model.to(device)

    # set model to eval mode
    model.eval()


    images = glob.glob("soccer_data/mytest/*.jpg") + glob.glob("soccer_data/mytest/*.png")
    # images = glob.glob("soccer_data/test/*.jpg")

    for image_path in images:

        # read image rgb
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image 115, 75
        image_rgb = cv2.resize(image, (300,300))

        image_rgb = ToTensor.PreprocessImage(image_rgb, True)

        # convert to tensor
        image_tensor = torch.from_numpy(image_rgb).float().to(device)

        # add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # get predictions
        preds = model(image_tensor)

        # convert to numpy
        preds = preds.cpu().detach().numpy()

        preds = preds.reshape(-1, 2)

        preds[:, 0] *= 115*15
        preds[:, 1] *= 75*15

        # USE PREDICTED HOMOGRAPHY MATRIX
        source_key_points = np.array(data_processing.GenerateSourceKeyPoints(image))
        homography, mask = cv2.findHomography(source_key_points, preds)

        # # USE PERFECT HOMOGRAPHY MATRIX
        # homography_path = image_path.replace(".jpg", ".homographyMatrix")
        # homography = np.loadtxt(homography_path)
        # homography[2] /= 15

        # # generate target key points
        # target_key_points = data_processing.GenerateTargetKeyPoints(homography, image)

        # create black image by 115, 75
        warped_image = cv2.warpPerspective(image, homography, (115*15,75*15))
        alpha_image = cv2.warpPerspective(np.ones_like(image)*255, homography, (115*15,75*15))
        # black_image = np.zeros((75, 115, 3), dtype=np.uint8)

        template_resized = cv2.resize(template, (115*15, 75*15))

        # add warped image onto template_resized image for only area where "alpha_image" is 255
        warped_image = (warped_image * (alpha_image/255) + template_resized * (1 - alpha_image/255)).astype(np.uint8)
        
        # draw points on black image
        for i in range(preds.shape[0]):
            cv2.circle(warped_image, (int(preds[i,0]), int(preds[i,1])), 2, (0, 255, 0), -1)

        # # draw target points on black image in red
        # for point in target_key_points:
        #     cv2.circle(black_image, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

        # save image convert to bgr
        cv2.imwrite(f"output/{Path(image_path).stem}_pred.jpg", cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))








