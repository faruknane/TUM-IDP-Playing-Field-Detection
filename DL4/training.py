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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

loss_bce = torch.nn.BCEWithLogitsLoss()
loss_mse = torch.nn.MSELoss()

def calc_metrics(prediction, label, save_image = False, images=None, epoch = 0, running_count = 0):
    # label: [N, 5, 2]
    # prediction: [N, 7, 13, 13]

    H, W = prediction.shape[2], prediction.shape[3]

    true_preds = 0
    inside_count = 0
    total_offset_error = 0
    
    for batch_i in range(label.shape[0]):
        
        if save_image:
            image = images[batch_i].cpu().numpy()
            image = image.transpose((1, 2, 0)) # (channel, height, width) -> (height, width, channel)
            # denormalize according to imagenet
            image = image * np.array([0.229, 0.224, 0.225])[None, None, :]
            image = image + np.array([0.485, 0.456, 0.406])[None, None, :]
            image = image * 255
            image = image.astype(np.uint8)
            
            image2 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            image2[:,:,:] = image[:,:,:]
            image = image2

        for i in range(label.shape[1]):
            is_inside = (label[batch_i, i, 0] >= 0) & (label[batch_i, i, 0] < 1) & (label[batch_i, i, 1] >= 0) & (label[batch_i, i, 1] < 1)
            if is_inside:
                inside_count += 1

                grid_num_x = int(label[batch_i, i, 0]*W)
                grid_num_y = int(label[batch_i, i, 1]*H)
                # ground_truth[0, grid_num_y, grid_num_x] = 1
                flat_index = grid_num_y*W + grid_num_x

                my_pred = prediction[batch_i, i:i+1, :, :] # shape: 1, H, W
                max_pred_index = torch.argmax(my_pred).data.item()

                if max_pred_index == flat_index and my_pred[0,grid_num_y, grid_num_x] > 0:
                    true_preds += 1
                
                # calculate the offset prediction error use mae
                grid_center_x = grid_num_x+0.5
                grid_center_y = grid_num_y+0.5

                offset_x = label[batch_i, i, 0]*W - grid_center_x
                offset_y = label[batch_i, i, 1]*H - grid_center_y

                pred_offset_x = prediction[batch_i,5,grid_num_y,grid_num_x]
                pred_offset_y = prediction[batch_i,6,grid_num_y,grid_num_x]

                offset_error = abs(pred_offset_x.data.item() - offset_x.data.item()) + abs(pred_offset_y.data.item() - offset_y.data.item())
                total_offset_error += offset_error / 2

                if save_image:

                    pred_gridnum_x = max_pred_index % W 
                    pred_gridnum_y = max_pred_index // W

                    pred_x = pred_gridnum_x + prediction[batch_i,5,pred_gridnum_y,pred_gridnum_x] + 0.5
                    pred_y = pred_gridnum_y + prediction[batch_i,6,pred_gridnum_y,pred_gridnum_x] + 0.5

                    pred_x /= W
                    pred_y /= H

                    pred_x *= image.shape[1]
                    pred_y *= image.shape[0]

                    pred_x = int(pred_x.data.item())
                    pred_y = int(pred_y.data.item())

                    # image is rgb
                    # if i = 0, draw a black circle
                    # if i = 1, draw a blue circle
                    # if i = 2, draw an orange circle
                    # if i = 3, draw a brown circle

                    if i == 0:
                        cv2.circle(image, (int(label[batch_i, i, 0]*image.shape[1]), int(label[batch_i, i, 1]*image.shape[0])), 2, (55, 55, 55), -1)
                    elif i == 1:
                        cv2.circle(image, (int(label[batch_i, i, 0]*image.shape[1]), int(label[batch_i, i, 1]*image.shape[0])), 2, (0, 0, 255), -1)
                    elif i == 2:
                        cv2.circle(image, (int(label[batch_i, i, 0]*image.shape[1]), int(label[batch_i, i, 1]*image.shape[0])), 2, (0, 165, 165), -1)
                    elif i == 3:
                        cv2.circle(image, (int(label[batch_i, i, 0]*image.shape[1]), int(label[batch_i, i, 1]*image.shape[0])), 2, (255, 0, 0), -1)
                    else:
                        cv2.circle(image, (int(label[batch_i, i, 0]*image.shape[1]), int(label[batch_i, i, 1]*image.shape[0])), 2, (255, 165, 0), -1)

                    cv2.circle(image, (pred_x, pred_y), 2, (0, 0, 0), -1)

            else:
                my_pred = prediction[batch_i, i:i+1, :, :]
                max_pred = my_pred.max()
                if max_pred < 0:
                    true_preds += 1

        if save_image:
            # create epoch folder in val_results if not exist
            if not os.path.exists(f"val_results/epoch_{epoch}"):
                os.makedirs(f"val_results/epoch_{epoch}")

            # rgb to bgr
            image = image[...,::-1]

            # save image
            cv2.imwrite(f"val_results/epoch_{epoch}/image_{running_count+batch_i}.png", image)


    grid_acc = true_preds / (label.shape[0]*label.shape[1])
    offset_final_error = total_offset_error / inside_count

    return grid_acc, offset_final_error





def loss_function(prediction, label):
    # label: [N, 5, 2]
    # prediction: [N, 7, 13, 13]

    H, W = prediction.shape[2], prediction.shape[3]

    total_loss = []
    
    for batch_i in range(label.shape[0]):
        for i in range(label.shape[1]):
            is_inside = (label[batch_i, i, 0] >= 0) & (label[batch_i, i, 0] < 1) & (label[batch_i, i, 1] >= 0) & (label[batch_i, i, 1] < 1)
            try:
                if is_inside:
                    grid_num_x = int(label[batch_i, i, 0]*W)
                    grid_num_y = int(label[batch_i, i, 1]*H)
                    flat_index = grid_num_y*W + grid_num_x

                    ground_truth = torch.zeros((1, H, W), device=prediction.device)
                    ground_truth[0, grid_num_y, grid_num_x] = 1 # shape 1, H, W
                    my_pred = prediction[batch_i, i:i+1, :, :] # shape: 1, H, W
                    grid_loss = loss_bce(my_pred, ground_truth) + 0.2*loss_bce(my_pred[0,grid_num_y, grid_num_x], ground_truth[0,grid_num_y, grid_num_x])
                    
                    grid_center_x = grid_num_x+0.5
                    grid_center_y = grid_num_y+0.5

                    offset_x = label[batch_i, i, 0]*W - grid_center_x
                    offset_y = label[batch_i, i, 1]*H - grid_center_y

                    pred_offset_x = prediction[batch_i,5,grid_num_y,grid_num_x]
                    pred_offset_y = prediction[batch_i,6,grid_num_y,grid_num_x]

                    # if torch.argmax(my_pred).data.item() == flat_index and my_pred[0, grid_num_y, grid_num_x] > 0:
                    #     # use mse loss between offsets and pred_offsets
                    #     offset_loss = loss_mse(pred_offset_x, offset_x) + loss_mse(pred_offset_y, offset_y)
                    # else:
                    #     offset_loss = 0

                    offset_loss = loss_mse(pred_offset_x, offset_x) + loss_mse(pred_offset_y, offset_y)
                    
                    loss = grid_loss + offset_loss
                    total_loss.append(loss)
                else:
                    ground_truth = torch.zeros((1, H, W), device=prediction.device)
                    my_pred = prediction[batch_i, i:i+1, :, :] 
                    grid_loss = loss_bce(my_pred, ground_truth)
                    total_loss.append(grid_loss)
            except:
                pass
    
    # now we have a list of losses, average all of them and return
    total_loss = torch.stack(total_loss)
    return torch.mean(total_loss)

def epoch_step(is_train, epoch, model, dataloader, optimizer, device):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_count = 0
    total_count = len(dataloader.dataset)

    total_grid_acc = 0
    total_offset_error = 0

    for i, batch in enumerate(dataloader):
        batchsize = batch["image"].shape[0]

        image = batch["image"]
        label = batch["label"]
        label_mid = (label[:, 0:1, :]+label[:, 1:2, :]+label[:, 2:3, :]+label[:, 3:4, :])/4
        label = torch.cat((label, label_mid), dim=1)
        
        image = image.to(device)
        label = label.to(device)

        if is_train: 
            optimizer.zero_grad()

        prediction = model(image) # 16, 7, 13, 13

        loss = loss_function(prediction, label)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.data.item() * batchsize
        running_count += batchsize

        if is_train:
            grid_acc, offset_error = calc_metrics(prediction, label)
        else:
            grid_acc, offset_error = calc_metrics(prediction, label, save_image=True, images=image, epoch = epoch, running_count = running_count)

        total_grid_acc += grid_acc * batchsize
        total_offset_error += offset_error * batchsize

        if is_train:
            last_write = f"Train Epoch: {epoch}"
        else:
            last_write = f"Val Epoch: {epoch}"
            
        last_write += f" [{running_count}/{total_count}] Loss: {running_loss/running_count:.10f}"
        last_write += f" Grid Acc: {total_grid_acc/running_count:.10f}"
        last_write += f" Offset Error: {total_offset_error/running_count:.10f}"

        print("\r" + last_write, end="")

        del image, label, prediction, loss

    print()


if __name__ == "__main__":
    
    save_model_path = "saved_models"

    training_dataset = CustomImageDataset("../DL2/data/train/videos", "../DL2/data/train/labels", transform=transforms.Compose(
    [
        ResizeImage((1200, 1200)), 
        CustomRotateTransform(rotation=(-15, 15), prob=0.3),
        RandomCrop(900, 1200, 0.75),
        ResizeImage((720, 720)), 
        CustomBlurTransform(sigma=(0.1, 1.5), prob=0.3),
        CustomShadowTransform(prob=0.3),
        CustomFlipTransform("horizontal", prob=0.5),
        CustomFlipTransform("vertical", prob=0.5),
        ColorShift(0.5, 25, 25, 25),
        Noise(0.3, 5, 5),
        Noise(0.3, -5, 5),
        ToTensor(False),
    ]))

    validation_dataset = CustomImageDataset("../DL2/data/val/videos", "../DL2/data/val/labels", transform=transforms.Compose(
    [
        ResizeImage((720, 720)), 
        ToTensor(False),
    ]))

    trainining_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True, num_workers=12)
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

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

