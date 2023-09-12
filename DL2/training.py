import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os, pathlib
from torchvision.io import read_image
import glob, json
import video_process as vp
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
from model import *

def epoch_step(is_train, epoch, model, dataloader, optimizer, device):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_count = 0
    total_count = len(dataloader.dataset)

    for i, batch in enumerate(dataloader):
        batchsize = batch["image"].shape[0]

        image = batch["image"]
        label = batch["label"]

        image = image.to(device)
        label = label.to(device)

        if is_train: 
            optimizer.zero_grad()

        prediction = model(image) # shape: (batch_size, 8)

        # reshape label to (batch_size, 8)
        label = label.view(-1, 8)

        loss = loss_function(prediction, label)

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

        del image, label, prediction, loss

    print()


if __name__ == "__main__":
    
    save_model_path = "saved_models"

    training_dataset = CustomImageDataset("data/train/videos", "data/train/labels", transform=transforms.Compose(
    [
        ResizeImage((900, 900)), 
        RandomCrop(700, 900, 0.9),
        ResizeImage((400, 400)), 
        ColorShift(0.5, 25, 25, 25),
        Noise(0.3, 5, 5),
        Noise(0.3, -5, 5),
        ToTensor(False),
    ]))

    validation_dataset = CustomImageDataset("data/val/videos", "data/val/labels", transform=transforms.Compose(
    [
        # ResizeImage((900, 900)), 
        # RandomCrop(700, 900, 0.9),
        ResizeImage((400, 400)), 
        ToTensor(False),
    ]))

    trainining_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True, num_workers=14)
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model.to(device)

    # define the loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):

        # train
        epoch_step(True, epoch, model, trainining_dataloader, optimizer, device)

        # val
        epoch_step(False, epoch, model, validation_dataloader, optimizer, device)

        # if save folder does not exist create
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        # save model
        torch.save(model.state_dict(), save_model_path + f"/model_epoch{epoch}.pth")

