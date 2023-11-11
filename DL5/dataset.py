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
from PIL import ImageFilter
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, videos_folder, labels_folder, transform=None, multiply_data=None):
    
        self.videos_folder = videos_folder
        self.labels_folder = labels_folder
        self.transform = transform
        self.frames = []
        self.all_field_types = []

        # read all videos at videos_folder
        all_videos = glob.glob(os.path.join(videos_folder, "*.mov")) + glob.glob(os.path.join(videos_folder, "*.mp4")) 

        for video in all_videos:
            #get file name without extension, use pathlib
            file_name = pathlib.Path(video).stem
            label_path = os.path.join(labels_folder, file_name + ".json")
            # read json_file
            with open(label_path) as f:
                data = json.load(f)

            field_type = data["field_type"]
            if field_type not in self.all_field_types:
                self.all_field_types.append(field_type)

            keys = list(data["frames"].keys())
            
            # read frame 0
            image = vp.TakeFrameFromVideo(video, 0)

            for key in keys:
                # get width and height
                width = image.shape[1]
                height = image.shape[0]

                # data[key] is [[x1,y1], [x2,y2], ...]
                # if any point is out of image, remove this key from data dictionary
                for point in data["frames"][key]:
                    if point[0] < 0 or point[0] > width or point[1] < 0 or point[1] > height:
                        del data["frames"][key]
                        break

            keys = list(data["frames"].keys())
            chosen_keys = []
            if len(keys) > 0:
                select_count = 50
                
                tam = select_count // len(keys) 
                chosen_keys += keys * tam
                
                random.seed(0)
                chosen_keys += random.sample(keys,k=select_count % len(keys) )
                

            # print(video, chosen_keys)
            for frame_index in chosen_keys:
                #print(frame_index,data[frame_index])
                self.frames.append((video, frame_index, data["frames"][frame_index], field_type))

        # multiply data
        if multiply_data is not None:
            self.frames = self.frames * multiply_data

        self.all_field_types.sort()

    def get_field_type_index(self, field_type):
        return self.all_field_types.index(field_type)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        
        image = vp.TakeFrameFromVideo(self.frames[idx][0], int(self.frames[idx][1]))
        
        label = self.frames[idx][2]
        label = np.array(label,dtype=np.float32)

        field_type = self.frames[idx][3]
        field_type_index = self.get_field_type_index(field_type)
        # make it [1] numpy array integer
        field_type_index = np.array([field_type_index], dtype=np.int64)

        sample = {
            "image": image, 
            "label": label,
            "field_type": field_type_index,   
        }

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

class CustomRotateTransform:

    def __init__(self, rotation = (-15, 15), prob = 0.3) -> None:
        self.rotation = rotation
        self.prob = prob

    def __call__(self, sample):

        if np.random.uniform(0, 1) > self.prob:
            return sample
        
        # rotate image and label by random angle at the center of image
        image = sample["image"]
        label = sample["label"] # [4, 2] float32 numpy array for 4 points

        # get random angle in degree
        angle = np.random.uniform(self.rotation[0], self.rotation[1])

        # get rotation matrix
        center_point = (image.shape[1]//2, image.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, 1)

        # rotate image
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # rotate label
        for i in range(label.shape[0]):
            point = label[i]
            x = point[0] # refers to width
            y = point[1] # refers to height

            x -= center_point[0]
            y -= center_point[1]

            # rotate clockwise
            new_x = x * np.cos(np.deg2rad(angle)) + y * np.sin(np.deg2rad(angle))
            new_y = -x * np.sin(np.deg2rad(angle)) + y * np.cos(np.deg2rad(angle))

            new_x += center_point[0]
            new_y += center_point[1]

            label[i][0] = new_x
            label[i][1] = new_y

        sample["image"] = image
        sample["label"] = label

        return sample







class CustomShadowTransform:
    def __init__(self, prob=1, shape_num=(5, 10), shape_size=(30, 100), shadow_strength=(0.1, 0.5), blur_strength=(0.1, 10)):
        self.prob = prob
        self.shape_num = shape_num
        self.shape_size = shape_size
        self.shadow_strength = shadow_strength
        self.blur_strength = blur_strength

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.prob:
            return sample

        image = sample['image']
        label = sample['label']

        # get random int from shape_num
        number_of_shapes = random.randint(self.shape_num[0], self.shape_num[1])
        shadow_mask = np.zeros_like(image).astype(np.float32)

        for i in range(number_of_shapes):
            shape_shd_str = np.random.uniform(
                self.shadow_strength[0], self.shadow_strength[1])
            if np.random.uniform(0, 1) > 0.5:
                # draw circle
                cv2.circle(shadow_mask, (random.randint(0, image.shape[1]), random.randint(0, image.shape[0])),
                           random.randint(self.shape_size[0], self.shape_size[1]), (shape_shd_str, shape_shd_str, shape_shd_str), -1)
            else:
                # draw rectangle
                start_point = (random.randint(
                    0, image.shape[1]), random.randint(0, image.shape[0]))
                end_point = (start_point[0]+random.randint(self.shape_size[0], self.shape_size[1]),
                             start_point[1]+random.randint(self.shape_size[0], self.shape_size[1]))
                cv2.rectangle(shadow_mask, start_point, end_point,
                              (shape_shd_str, shape_shd_str, shape_shd_str), -1)

        # get random float from blur_strength
        sigma = np.random.uniform(self.blur_strength[0], self.blur_strength[1])

        # blur shadow_mask
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigma)

        # add shadow_mask to image
        shadow_mask = 1 - shadow_mask

        image[:, :, :] = image[:, :, :] * shadow_mask

        sample['image'] = image
        sample['label'] = label

        return sample


class CustomBlurTransform:
    def __init__(self, sigma=(0.1, 1.5), prob=0.3):
        self.sigma = sigma
        self.prob = prob

    def __call__(self, sample):

        if np.random.uniform(0, 1) > self.prob:
            return sample

        image = sample['image']
        label = sample['label']

        image_pil = Image.fromarray(np.uint8(image))

        sigma = random.uniform(self.sigma[0], self.sigma[1])

        image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

        image = np.array(image_pil)

        sample['image'] = image
        sample['label'] = label
        return sample


class CustomFlipTransform:
    def __init__(self, flip_type, prob=0.5):
        self.flip_type = flip_type
        self.prob = prob

    def __call__(self, sample):

        if np.random.uniform(0, 1) > self.prob:
            return sample

        image = sample['image']
        label = sample['label']

        if self.flip_type == 'horizontal':
            # Horizontal flip
            image = image[:, ::-1, :]
            label[:, 0] = image.shape[1] - label[:, 0]

            # Swap left and right points
            label[[0, 1]] = label[[1, 0]]
            label[[2, 3]] = label[[3, 2]]
        else:
            # Vertical flip
            image = image[::-1, :, :]
            label[:, 1] = image.shape[0] - label[:, 1]

            # Swap top and bottom points
            label[[0, 3]] = label[[3, 0]]
            label[[1, 2]] = label[[2, 1]]

        sample['image'] = image
        sample['label'] = label
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

class CenterCrop:

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
    
        # crop image
        crop_position_x = (image.shape[1] - crop_size_width) // 2
        crop_position_y = (image.shape[0] - crop_size_height) // 2
        image = image[crop_position_y:crop_position_y+crop_size_height,
                        crop_position_x:crop_position_x+crop_size_width, :]
        
        # crop label
        label[:, 0] = label[:, 0] - crop_position_x
        label[:, 1] = label[:, 1] - crop_position_y

        sample["image"] = image
        sample["label"] = label

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
