from typing import Any
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
import random

class CustomImageDataset(Dataset):
    def __init__(self, videos_folder, labels_folder, transform=None):
    
        self.videos_folder = videos_folder
        self.labels_folder = labels_folder
        self.transform = transform
        self.frames = []

        # read all videos at videos_folder
        all_videos = glob.glob(os.path.join(videos_folder, "*.mov")) + glob.glob(os.path.join(videos_folder, "*.mp4")) 

        for video in all_videos:
            #get file name without extension, use pathlib
            file_name = pathlib.Path(video).stem
            label_path = os.path.join(labels_folder, file_name + ".json")
            # read json_file
            with open(label_path) as f:
                data = json.load(f)

            keys = list(data.keys())
            
            # read frame 0
            image = vp.TakeFrameFromVideo(video, 0)

            for key in keys:
                # get width and height
                width = image.shape[1]
                height = image.shape[0]

                # data[key] is [[x1,y1], [x2,y2], ...]
                # if any point is out of image, remove this key from data dictionary
                for point in data[key]:
                    if point[0] < 0 or point[0] > width or point[1] < 0 or point[1] > height:
                        del data[key]
                        break

            keys = list(data.keys())
            random.seed(0)
            if len(keys) > 0:
                chosen_keys = random.sample(keys, k=max(50,len(keys)//2))
            else:
                chosen_keys = []
            # print(video, chosen_keys)
            for frame_index in chosen_keys:
                #print(frame_index,data[frame_index])
                self.frames.append((video, frame_index, data[frame_index]))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        
        image = vp.TakeFrameFromVideo(self.frames[idx][0], int(self.frames[idx][1]))
        
        label = self.frames[idx][2]
        label = np.array(label,dtype=np.float32)

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)
        
        label = sample["label"]
        
        # check if min element in label is less than 0 or max element in label is greater than 1
        # if np.min(label) < 0 or np.max(label) > 1:
        #     print(self.frames[idx][0], int(self.frames[idx][1]), "ERROR: label out of range", np.min(label), np.max(label), idx)          

        return sample
    
def visualize(dataset, index, denormalize=False):
    sample = dataset[index]
    image = sample["image"]
    label = sample["label"]

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

    # draw image
    for point in label:
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    # print shapes
    print("image shape:", image.shape)
    print("label shape:", label.shape)
    # print labels
    print("label:", label)
    #save image
    cv2.imwrite(f"debug/image-{index}.png", image)

  
class ColorShift:

    def __init__(self, prob, r_value, g_value, b_value) -> None:
        self.prob = prob
        self.r_value = r_value
        self.g_value = g_value
        self.b_value = b_value
    
    def __call__(self, sample):
        
        if np.random.uniform(0, 1) > self.prob:
            return sample
        
        image = sample["image"]

        # generate random float values for r, g, b between -r_value and +r_value
        r = np.random.uniform(-self.r_value, self.r_value)
        g = np.random.uniform(-self.g_value, self.g_value)
        b = np.random.uniform(-self.b_value, self.b_value)

        # add r, g, b values to image and center_crop_image
        image = image.astype(np.float32)
        image[:,:,0] += r
        image[:,:,1] += g
        image[:,:,2] += b

        # clip values between 0 and 255
        image = np.clip(image, 0, 255)
        image = image.astype(np.float32)

        sample["image"] = image
        return sample

class Noise:

    def __init__(self, prob, mean, std) -> None:
        self.prob = prob
        self.mean = mean
        self.std = std

    def AddNoise(image, mean, std):
        # create gaussian noise
        noise = np.random.normal(mean, std, image.shape)

        # img fp32
        image = image.astype(np.float32)

        # add noise to image
        image = image + noise

        # clip image to range [0, 255]
        image = np.clip(image, 0, 255)

        return image

    def __call__(self, sample):
        image = sample["image"]

        if np.random.uniform(0, 1) > self.prob:
            return sample
        
        sample["image"] = Noise.AddNoise(image, self.mean, self.std)

        return sample
    

class RandomCrop:

    def __init__(self, min_crop_size, max_crop_size, prob) -> None:
        self.min_crop_size = min_crop_size # just a number
        self.max_crop_size = max_crop_size # just a number
        self.prob = prob
    
    def __call__(self, sample):
        
        random_number = random.random()

        if random_number > self.prob:
            return sample
        
        image = sample["image"]
        label = sample["label"]

        w_h_ratio = image.shape[1] / image.shape[0]

        # random crop size
        crop_size_height = random.randint(self.min_crop_size, self.max_crop_size+1)
        crop_size_width = int(crop_size_height * w_h_ratio)
       
        # random crop position
        crop_position_x = random.randint(0, image.shape[1] - crop_size_width + 1)
        crop_position_y = random.randint(0, image.shape[0] - crop_size_height + 1)

        # crop image
        image = image[crop_position_y:crop_position_y+crop_size_height, 
                      crop_position_x:crop_position_x+crop_size_width, :]
        
        # crop label
        label[:, 0] = label[:, 0] - crop_position_x
        label[:, 1] = label[:, 1] - crop_position_y

        sample["image"] = image
        sample["label"] = label

        return sample

class ResizeImage:

    # size: (height, width)
    def __init__(self, size):
        self.size = size # (height, width)

    def __call__(self, sample):
        image = sample["image"]
        sample["image"] = cv2.resize(image, (self.size[1], self.size[0]))

        label = sample["label"]
        for point in label:
            point[0] = point[0] * self.size[1] / image.shape[1]
            point[1] = point[1] * self.size[0] / image.shape[0]

        return sample

class ToTensor:
    def __init__(self, divide255):
        self.divide255 = divide255
        print("ToTensor divide255:", divide255)

    def PreprocessImage(image, divide255):
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

        if divide255:
            image = image / 255
        else:
            image = image - np.min(image)
            image = image / np.max(image)

        if image.shape[2] == 1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        tmpImg = tmpImg.transpose((2, 0, 1))

        tmpImg = tmpImg.astype(np.float32)
        return tmpImg
    
    def NormalizePoints(points, w ,h):

        new_points = []

        for point in points:

            point[0] = point[0] / w
            point[1] = point[1] / h

            new_points.append(point)

        new_points = np.array(new_points, dtype=np.float32)

        return new_points
    
    def __call__(self, sample):
        sample['image'] = ToTensor.PreprocessImage(sample['image'], self.divide255) # (channel, height, width)
        sample["label"] = ToTensor.NormalizePoints(sample["label"],sample['image'].shape[2],sample['image'].shape[1])

        return sample

if __name__ == "__main__":

    training_dataset = CustomImageDataset("data/train/videos", "data/train/labels", transform=transforms.Compose(
    [
        ResizeImage((720, 720)), 
        ToTensor(False),
    ]))
    
    dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True)

    for i,batch in enumerate(dataloader):
        print(f'batch {i}')
