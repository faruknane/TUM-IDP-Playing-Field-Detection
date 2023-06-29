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
from dataloader import HomographyDataset, ColorShift, Noise, HorizontalFlip, RandomCrop, ResizeImage, CalculateTargetKeyPoints, ToTensor
import random
from model import get_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def Process(is_train, epoch, model, dataloader, optimizer, device):
    if is_train: model.train()
    else: model.eval()

    running_loss = 0
    running_count = 0
    total_count = len(dataloader.dataset)

    for i, sample in enumerate(dataloader):
        batchsize = sample["image"].shape[0]

        image = sample["image"].to(device)
        # homography = sample["homography"].to(device)
        target_key_points = sample["target_key_points"].to(device)

        
        if is_train: 
            optimizer.zero_grad()

        preds = model(image)

        target_key_points = target_key_points.view(-1, 8)

        loss = criterion(preds, target_key_points)

        if is_train:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.data.item() * batchsize
        running_count += batchsize

        
        if is_train:
            last_write = f"Train Epoch: {epoch}"
        else:
            last_write = f"Val Epoch: {epoch}"
            
        last_write += f" [{running_count}/{total_count}] Loss: {running_loss/running_count:.10f}"
        print("\r" + last_write, end="")

        del image, target_key_points, preds, loss

    print()


if __name__ == "__main__":

    save_model_path = "saved_models"

    # find all jpg files using glob
    train_images = glob.glob("soccer_data/train_val/*.jpg")
    val_images = glob.glob("soccer_data/test/*.jpg")

    for i in range(10):
        train_images += train_images

    train_dataset = HomographyDataset(train_images, transform=transforms.Compose([
        HorizontalFlip(0.5, 115),
        RandomCrop(0.8, 1280, 1000),
        ResizeImage((300,300)),
        ColorShift(0.5, 25, 25, 25),
        Noise(0.3, 5, 5),
        Noise(0.3, -5, 5),

        CalculateTargetKeyPoints(),
        ToTensor(True, 75, 115)
    ]))

    
    val_dataset = HomographyDataset(val_images, transform=transforms.Compose([
        # HorizontalFlip(0.5, 115),
        # RandomCrop(0.8, 1280, 1000),
        ResizeImage((300,300)),
        CalculateTargetKeyPoints(),
        ToTensor(True, 75, 115)
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=14)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model.to(device)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    for epoch in range(1000):

        # if epoch is greater than 15, decrease learning rate
        # if epoch % 5 == 4:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] / 2

        # train
        Process(True, epoch, model, train_dataloader, optimizer, device)

        # val
        Process(False, epoch, model, val_dataloader, optimizer, device)

        # if save folder does not exist create
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        # save model
        torch.save(model.state_dict(), save_model_path + f"/model_epoch{epoch}.pth")





