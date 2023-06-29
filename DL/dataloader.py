import cv2
import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import pathlib
import math
from torchvision import transforms
import glob

import data_processing

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

class HorizontalFlip:

    def __init__(self, prob=0.5, tw=115):
        self.prob = prob
        self.tw = tw

    def __call__(self, sample):
        
        if np.random.rand() > self.prob:
            return sample

        # flip image
        sample["image"] = cv2.flip(sample["image"], 1)

        # flip homography
        sample["homography"] = data_processing.ApplyHorizontalFlip(sample["homography"], sample["image"].shape[1], self.tw)

        return sample

class RandomCrop:

    def __init__(self, prob, w_max, w_min) -> None:
        self.prob = prob
        self.w_max = w_max
        self.w_min = w_min

    def __call__(self, sample):

        if np.random.rand() > self.prob:
            return sample
        
        image = sample["image"]

        # generate random width between w_min and w_max
        w_new = np.random.randint(self.w_min, self.w_max)
        h_new = int(image.shape[0] / image.shape[1] * w_new)

        # generate random crop position x y
        x = np.random.randint(0, image.shape[1] - w_new)
        y = np.random.randint(0, image.shape[0] - h_new)

        # crop image
        cropped_image = image[y:y+h_new, x:x+w_new]

        # source_key_points
        for i in range(len(sample["source_key_points"])):
            key_point = sample["source_key_points"][i]

            sx = key_point[0]
            sy = key_point[1]

            sx = sx / image.shape[1] * cropped_image.shape[1]
            sy = sy / image.shape[0] * cropped_image.shape[0]

            sx += x
            sy += y

            sample["source_key_points"][i] = np.array([sx, sy], dtype=np.float32)

        # image[0:y, :] = 0
        # image[y+h_new:, :] = 0
        # image[:, 0:x] = 0
        # image[:, x+w_new:] = 0

        sample["image"] = cropped_image

        return sample

class ResizeImage:

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image = sample["image"]

        image = cv2.resize(image, (self.size[1], self.size[0]))

        sample["image"] = image
        
        return sample
    
class ToTensor:

    def __init__(self, divide255, normalize_h, normalize_w) -> None:
        self.divide255 = divide255
        print("ToTensor divide255:", divide255)
        self.normalize_w = normalize_w
        self.normalize_h = normalize_h

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
    
    def NormalizePoints(self, points):

        new_points = []

        for point in points:

            point[0] = point[0] / self.normalize_w
            point[1] = point[1] / self.normalize_h

            new_points.append(point)

        new_points = np.array(new_points, dtype=np.float32)

        return new_points        


    def __call__(self, sample):
        sample['image'] = ToTensor.PreprocessImage(sample['image'], self.divide255)
        sample["target_key_points"] = self.NormalizePoints(sample["target_key_points"])
        return sample

class CalculateTargetKeyPoints:

    def __init__(self):
        pass

    def __call__(self, sample):

        
        # sample["homography"][2] /= 5 

        # project source key points to target key points
        target_key_points = []
        for key_point in sample["source_key_points"]:
            target_key_point = data_processing.ProjectPoint(sample["homography"], key_point[0], key_point[1])
            target_key_points.append(target_key_point)

        sample["target_key_points"] = target_key_points

        return sample


class HomographyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get homography matrix
        homography_path = image_path.replace(".jpg", ".homographyMatrix")
        homography = np.loadtxt(homography_path).astype(np.float32)

        source_key_points = data_processing.GenerateSourceKeyPoints(image)

        sample = {"image": image, "homography": homography, "source_key_points": source_key_points}

        if self.transform:
            sample = self.transform(sample)


        return sample


if __name__ == "__main__":

    # find all jpg files using glob
    images = glob.glob("soccer_data/train_val/*.jpg")

    dataset = HomographyDataset(images, transform=transforms.Compose([
        HorizontalFlip(0.5, 115),
        RandomCrop(0.8, 1280, 1000),
        ResizeImage((75, 115)),
        CalculateTargetKeyPoints(),
        ToTensor(True, 75, 115)
    ]))

    
    for i in range(10):
        sample = dataset[0]

        # img = sample["image"]
        # homography = sample["homography"]
        target_key_points = sample["target_key_points"]
        print(target_key_points)   

        # # save image convert rgb to bgr
        # cv2.imwrite(f"debug/original_{i}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # # warped image
        # img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

        # # draw target key points on warped image
        # for key_point in target_key_points:
        #     cv2.circle(img, (int(key_point[0]), int(key_point[1])), 5, (0, 255, 0), -1)
            
        # img = img[0:75*5, 0:115*5]
        # # save warped image
        # cv2.imwrite(f"debug/warped_{i}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))




