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

def visualize(image, label, index, denormalize=False):
    if denormalize:
        for points in label:
            points[0] = points[0] * image.shape[2]
            points[1] = points[1] * image.shape[1]
        # detranspose
        image = image.transpose((1, 2, 0)) # (channel, height, width) -> (height, width, channel)

        # denormalize according to imagenet
        image = image * np.array([0.229, 0.224, 0.225])[None, None, :]
        image = image + np.array([0.485, 0.456, 0.406])[None, None, :]
        image = image * 255
        image = image.astype(np.uint8)
        
        image2 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        image2[:,:,:] = image[:,:,:]
        image = image2

    # draw image
    for point in label:
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

    # rgb to bgr
    image = image[...,::-1]

    #save image
    cv2.imwrite(f"val_results/image_{index}.png", image)

def val_pred(model, dataloader, device):
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

        prediction = model(image) # shape: (batch_size, 8)

        # save prediction with image
        for j in range(batchsize):
            visualize(image[j].cpu().detach().numpy(), prediction[j].cpu().detach().numpy().reshape((4,2)), running_count+j, denormalize=True)

        # reshape label to (batch_size, 8)
        label = label.view(-1, 8)

        loss = loss_function(prediction, label)

        running_loss += loss.data.item() * batchsize
        running_count += batchsize
        
        last_write = f"Val: [{running_count}/{total_count}] Loss: {running_loss/running_count:.10f}"
        print("\r" + last_write, end="")

        del image, label, prediction, loss

    print()


if __name__ == "__main__":
    
    model_path = "saved_models/model_epoch9.pth"

    validation_dataset = CustomImageDataset("data/val/videos", "data/val/labels", transform=transforms.Compose(
    [
        # ResizeImage((900, 900)), 
        # RandomCrop(700, 900, 0.9),
        ResizeImage((400, 400)), 
        ToTensor(False),
    ]))

    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model.to(device)

    # load model from file
    model.load_state_dict(torch.load(model_path))

    # define the loss function and optimizer
    loss_function = torch.nn.MSELoss()

    val_pred(model, validation_dataloader, device)


