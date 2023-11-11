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
import wandb
import random

import field_inference
import metrics

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

loss_bce = torch.nn.BCEWithLogitsLoss()
loss_mse = torch.nn.MSELoss()
loss_categorical = torch.nn.CrossEntropyLoss()

def get_predicted_points(prediction, class_prediction):
    # label: [N, 5, 2]
    # prediction: [N, 7, 13, 13]
    # class_prediction: [N] numpy array
    pred_points = []

    H, W = prediction.shape[2], prediction.shape[3]
    Ms = []

    for batch_i in range(prediction.shape[0]):

        points = []

        for i in range(prediction.shape[1]-2):
            my_pred = prediction[batch_i, i:i+1, :, :]
            max_pred = my_pred.max()
            
            if max_pred < 0: 
                points.append(None)  
                continue
            
            max_pred_index = torch.argmax(my_pred).data.item()

            pred_gridnum_x = max_pred_index % W 
            pred_gridnum_y = max_pred_index // W

            pred_x = pred_gridnum_x + prediction[batch_i,prediction.shape[1]-2,pred_gridnum_y,pred_gridnum_x] + 0.5
            pred_y = pred_gridnum_y + prediction[batch_i,prediction.shape[1]-2+1,pred_gridnum_y,pred_gridnum_x] + 0.5

            pred_x /= W
            pred_y /= H

            my_point = np.array([pred_x.data.item(), pred_y.data.item()], dtype=np.float32)
            points.append(my_point)

        # use field_inference
        points, M = field_inference.infer_field(points, class_prediction[batch_i])
        pred_points.append(points)
        Ms.append(M)

    return pred_points, Ms


def project_template(image, pred_real_coords):
    global template_img
    
    [topleft, topright, bottomright, bottomleft] = pred_real_coords[0:4]

    # first
    pts1 = np.float32([[0, 0], [template_img.shape[1], 0], [template_img.shape[1], template_img.shape[0]], [0, template_img.shape[0]]])
    pts2 = np.float32([topleft, topright, bottomright, bottomleft])

    # second
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(template_img, M, (image.shape[1], image.shape[0]))

    # create a white image like field template image
    accepted_area = dst[:,:,3] / 255

    # third
    image[:,:] = (accepted_area[:,:,np.newaxis] * dst[:,:, 0:3]) +  ((1-accepted_area)[:,:,np.newaxis] * image[:,:]) 
        
    return image

def save_predicted_points(pred_points, class_prediction, Ms, label, images, epoch, running_count):
    # label: [N, 5, 2]
    # class_prediction: [N] numpy array

    imageH, imageW = images.shape[2], images.shape[3]

    for batch_i in range(label.shape[0]):

        image = images[batch_i].cpu().numpy()
        image = image.transpose((1, 2, 0)) # (channel, height, width) -> (height, width, channel)
        # denormalize according to imagenet
        image = image * np.array([0.229, 0.224, 0.225])[None, None, :]
        image = image + np.array([0.485, 0.456, 0.406])[None, None, :]
        image = image * 255
        image = image.astype(np.uint8)
        
        image2 = np.zeros((image.shape[0]*3, image.shape[1]*3, 3), dtype=np.uint8)
        # put center
        image2[imageH:imageH*2, imageW:imageW*2, :] = image[:,:,:]
        image = image2

        pred_point = pred_points[batch_i] # [5, 2]

        for i in range(label.shape[1]):

            cv2.circle(image, (int(imageW + label[batch_i, i, 0]*imageW), int(imageH + label[batch_i, i, 1]*imageH)), 2, (255, 0, 0), -1)

            if pred_point is not None:
                cv2.circle(image, (int(imageW + pred_point[i, 0]*imageW), int(imageH + pred_point[i, 1]*imageH)), 2, (0, 0, 255), -1) 
                # write text "i" on the image
                cv2.putText(image, str(i), (int(imageW + pred_point[i, 0]*imageW), int(imageH + pred_point[i, 1]*imageH)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if pred_point is not None:
            dlines = field_inference.get_playing_field_lines(class_prediction[batch_i], Ms[batch_i])
            for dline in dlines:
                cv2.line(image, (int(imageW + dline[0][0]*imageW), int(imageH + dline[0][1]*imageH)), (int(imageW + dline[1][0]*imageW), int(imageH + dline[1][1]*imageH)), (0, 255, 0), 2)

        if not os.path.exists(f"debug/epoch_{epoch}"):
                os.makedirs(f"debug/epoch_{epoch}")
        
        # if running_count <= 32:
        #     wandb.log({
        #         f"debug/epoch_{epoch}/image_{running_count+batch_i-label.shape[0]}.png": wandb.Image(image)
        #     }, commit=False)

        # rgb to bgr
        image = image[...,::-1]

        # save image
        cv2.imwrite(f"debug/epoch_{epoch}/image_{running_count+batch_i-label.shape[0]}.png", image)

        


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

                pred_offset_x = prediction[batch_i,label.shape[1],grid_num_y,grid_num_x]
                pred_offset_y = prediction[batch_i,label.shape[1]+1,grid_num_y,grid_num_x]

                offset_error = abs(pred_offset_x.data.item() - offset_x.data.item()) + abs(pred_offset_y.data.item() - offset_y.data.item())
                total_offset_error += offset_error / 2

                if save_image:

                    pred_gridnum_x = max_pred_index % W 
                    pred_gridnum_y = max_pred_index // W

                    pred_x = pred_gridnum_x + prediction[batch_i,label.shape[1],pred_gridnum_y,pred_gridnum_x] + 0.5
                    pred_y = pred_gridnum_y + prediction[batch_i,label.shape[1]+1,pred_gridnum_y,pred_gridnum_x] + 0.5

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
                    grid_loss = loss_bce(my_pred, ground_truth) + 0.1*loss_bce(my_pred[0,grid_num_y, grid_num_x], ground_truth[0,grid_num_y, grid_num_x])
                    
                    grid_center_x = grid_num_x+0.5
                    grid_center_y = grid_num_y+0.5

                    offset_x = label[batch_i, i, 0]*W - grid_center_x
                    offset_y = label[batch_i, i, 1]*H - grid_center_y

                    pred_offset_x = prediction[batch_i,label.shape[1],grid_num_y,grid_num_x]
                    pred_offset_y = prediction[batch_i,label.shape[1]+1,grid_num_y,grid_num_x]

                    # if torch.argmax(my_pred).data.item() == flat_index and my_pred[0, grid_num_y, grid_num_x] > 0:
                    #     # use mse loss between offsets and pred_offsets
                    #     offset_loss = loss_mse(pred_offset_x, offset_x) + loss_mse(pred_offset_y, offset_y)
                    # else:
                    #     offset_loss = 0

                    offset_loss = loss_mse(pred_offset_x, offset_x) + loss_mse(pred_offset_y, offset_y)
                    
                    loss = grid_loss + offset_loss * 0.5
                    total_loss.append(loss)
                else:
                    ground_truth = torch.zeros((1, H, W), device=prediction.device)
                    my_pred = prediction[batch_i, i:i+1, :, :] 
                    grid_loss = loss_bce(my_pred, ground_truth)
                    loss = grid_loss
                    total_loss.append(loss)
            except:
                pass
    
    # now we have a list of losses, average all of them and return
    total_loss = torch.stack(total_loss)
    return torch.mean(total_loss)

def epoch_step(is_train, epoch, model, dataloader, optimizer, device):
    if is_train:
        model.train()
        step_type = "Train"
    else:
        model.eval()
        step_type = "Val"

    running_loss = 0.0
    running_count = 0
    total_count = len(dataloader.dataset)

    total_grid_acc = 0
    total_offset_error = 0
    total_class_acc = 0
    iou, valid_count, none_count = 0, 0, 0

    total_oks, total_oks_valid_count = 0, 0

    for i, batch in enumerate(dataloader):
        batchsize = batch["image"].shape[0]

        field_type = batch["field_type"]
        image = batch["image"]
        label = batch["label"] # [N, 4, 2]
        label_mid = (label[:, 0:1, :]+label[:, 1:2, :]+label[:, 2:3, :]+label[:, 3:4, :])/4
        label_top_mid = (label[:, 0:1, :]+label[:, 1:2, :])/2
        label_bot_mid = (label[:, 2:3, :]+label[:, 3:4, :])/2
        label = torch.cat((label, label_mid, label_top_mid, label_bot_mid), dim=1)
        
        image = image.to(device)
        label = label.to(device)
        field_type = field_type.to(device) # [N, 1]

        if is_train: 
            optimizer.zero_grad()

        prediction, class_prediction = model(image) # 16, 7, 13, 13
        # class_prediction is [N, class_count]

        loss1 = loss_function(prediction, label)
        loss2 = loss_categorical(class_prediction, field_type.squeeze(1).long())
        loss = loss1 + loss2 * 0.05

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.data.item() * batchsize
        running_count += batchsize

        # calculate classification accuracy
        class_prediction = torch.argmax(class_prediction, dim=1) # [N]
        class_prediction = class_prediction.cpu().detach().numpy() # [N] numpy array
        field_type = field_type.cpu().detach().numpy().squeeze(1)
        class_eq = (class_prediction == field_type)
        class_acc = class_eq.sum() / len(class_eq)
        total_class_acc += class_acc * batchsize
        # end of calculate classification accuracy
        
        if is_train:
            grid_acc, offset_error = calc_metrics(prediction, label)
        else:
            grid_acc, offset_error = calc_metrics(prediction, label, save_image=True, images=image, epoch = epoch, running_count = running_count)
            predicted_points, Ms = get_predicted_points(prediction, class_prediction)
            save_predicted_points(predicted_points, class_prediction, Ms, label, image, epoch, running_count)

            label_np = label.cpu().detach().numpy()

            current_iou, current_valid_count, current_none_count = metrics.calculate_miou(predicted_points, label_np, debug=False)
            iou += current_iou
            valid_count += current_valid_count
            none_count += current_none_count

            oks, oks_valid_count = metrics.calculate_oks(predicted_points, label_np, debug=False)
            total_oks += oks
            total_oks_valid_count += oks_valid_count


        total_grid_acc += grid_acc * batchsize
        total_offset_error += offset_error * batchsize

        last_write = f"{step_type} Epoch: {epoch}"
        last_write += f" [{running_count}/{total_count}] Loss: {running_loss/running_count:.10f}"
        last_write += f" Class Acc: {total_class_acc/running_count:.10f}"
        last_write += f" Grid Acc: {total_grid_acc/running_count:.10f}"
        last_write += f" Offset Error: {total_offset_error/running_count:.10f}"

        if not is_train:
            last_write += f" IoU: {iou/valid_count:.6f}"
            last_write += f" Miss Rate: {none_count/valid_count:.6f}"
            last_write += f" OKS: {total_oks/total_oks_valid_count:.6f}"

        print("\r" + last_write, end="")

        del image, label, prediction, loss

    if is_train:
        wandb.log({f"{step_type}/loss": running_loss/running_count, 
                    f"{step_type}/class_acc": total_class_acc/running_count,
                    f"{step_type}/grid_acc": total_grid_acc/running_count, 
                    f"{step_type}/offset_error": total_offset_error/running_count}, commit=False)
    else:
        wandb.log({f"{step_type}/loss": running_loss/running_count, 
                    f"{step_type}/class_acc": total_class_acc/running_count,
                    f"{step_type}/grid_acc": total_grid_acc/running_count, 
                    f"{step_type}/offset_error": total_offset_error/running_count,
                    f"{step_type}/iou": iou/valid_count, f"{step_type}/miss_rate": none_count/valid_count, f"{step_type}/oks": total_oks/total_oks_valid_count}, commit=False)
    
    print()


if __name__ == "__main__":
    
    # template_img = cv2.imread("field_template4.png", cv2.IMREAD_UNCHANGED)

    config = {
        "project": "tum-idp-dl5",
        "run": "exp10_7points_newdata_classification_finalrun_novalaug",
        "lr": 0.001,
        "batch_size": 16,
        "model": "resnet18_customized",
        "image_size": 720
    }

    save_model_path = "saved_models"

    training_dataset = CustomImageDataset("../DL2/data/train/videos", "../DL2/data/train/labels", 
    multiply_data=30,
    transform=transforms.Compose(
    [
        ResizeImage((1200, 1200)), 
        CustomRotateTransform(rotation=(-15, 15), prob=0.3),
        RandomCrop(900, 1200, 0.75),
        ResizeImage((config["image_size"], config["image_size"])), 
        CustomBlurTransform(sigma=(0.1, 1.5), prob=0.3),
        CustomShadowTransform(prob=0.3),
        CustomFlipTransform("horizontal", prob=0.5),
        CustomFlipTransform("vertical", prob=0.5),
        ColorShift(0.5, 25, 25, 25),
        Noise(0.3, 5, 5),
        Noise(0.3, -5, 5),
        ToTensor(False),
    ]))

    field_inference.SetFieldTypes(training_dataset.all_field_types)

    # validation_dataset = CustomImageDataset("../DL2/data/val/videos", "../DL2/data/val/labels", transform=transforms.Compose(
    # [
    #     ResizeImage((720, 720)), 
    #     ToTensor(False),
    # ]))

    validation_dataset = CustomImageDataset("../DL2/data/val/videos", "../DL2/data/val/labels", transform=transforms.Compose(
    [
        # ResizeImage((1200, 1200)), 
        # RandomCrop(900, 1200, 0.75),
        ResizeImage((config["image_size"], config["image_size"])), 
        ToTensor(False),
    ]))

    trainining_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=20)
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(config["model"], class_count=len(training_dataset.all_field_types))
    model.to(device)

    # load model weights
    # print("close this")
    # model.load_state_dict(torch.load("saved_models/model_epoch20.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    config["train_set"] = len(trainining_dataloader.dataset)
    config["val_set"] = len(validation_dataloader.dataset)

    wandb.init(
        entity="faruknane54",
        project=config["project"],
        name=config["run"],
        config=config,
    )

    for epoch in range(1000):

        # train
        epoch_step(True, epoch, model, trainining_dataloader, optimizer, device)

        # val
        epoch_step(False, epoch, model, validation_dataloader, optimizer, device)

        wandb.log({ }, commit=True)

        # if save folder does not exist create
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        # save model
        torch.save(model.state_dict(), save_model_path + f"/model_epoch{epoch}.pth")

