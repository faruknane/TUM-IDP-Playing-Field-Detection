import torch
import torchvision
import cv2
import numpy as np
import torchvision.models as models
import torch.nn as nn

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained=True)

        self.backbone = model
        self.final_layer = nn.Conv2d(in_channels=2048, out_channels=12, kernel_size=(1,1))

        self.activation = {}

        self.SetHook()

    def FreezeBackbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def UnfreezeBackbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def SetHook(self):
        self.backbone.layer4.register_forward_hook(self.get_activation('layer4'))

    def forward(self, x):
      self.backbone(x)
      x = self.activation["layer4"]
      x = self.final_layer(x)
      return x
    

class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        model = models.resnet34(pretrained=True)
        model.avgpool = torch.nn.Identity()
        def forward(x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            return x

        model._forward_impl = forward

        self.backbone = model
        self.final_layer = nn.Conv2d(in_channels=model.fc.in_features, out_channels=9, kernel_size=(1,1))

    def forward(self, x):
      x = self.backbone(x)
      x = self.final_layer(x)
      return x
    

class Resnet18(nn.Module):
    def __init__(self, class_count):
        super(Resnet18, self).__init__()

        self.class_count = class_count

        model = models.resnet18(pretrained=True)
        model.avgpool = torch.nn.Identity()


        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=model.fc.in_features, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=1152, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=9*32),
            nn.ReLU(),
        )

        self.l4m_pass = nn.Sequential(
            nn.Conv2d(in_channels=model.fc.in_features, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.l4m_process = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.l4_pass = nn.Sequential(
            nn.Conv2d(in_channels=model.fc.in_features, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.l4_process = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=7, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels=model.fc.in_features, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

        self.classification1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.classification2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.class_count)
        )



        def forward(x):
            # x is [batch_size, 3, 720, 720]
            x = model.conv1(x) # [batch_size, 64, 360, 360]
            x = model.bn1(x) # [batch_size, 64, 360, 360]
            x = model.relu(x) # [batch_size, 64, 360, 360]
            x = model.maxpool(x) # [batch_size, 64, 180, 180]

            l1 = model.layer1(x) # [batch_size, 64, 180, 180]
            l2 = model.layer2(l1) # [batch_size, 128, 90, 90]
            l3 = model.layer3(l2) # [batch_size, 256, 45, 45]
            l4 = model.layer4(l3) # [batch_size, 512, 23, 23]

            l4m = self.maxpool1(l4) # [batch_size, 512, 11, 11]
            l5 = self.layer5(l4m) # [batch_size, 128, 3, 3]

            # flatten l5
            l5_flat = torch.flatten(l5, 1) # [batch_size, 1152]
            d1 = self.dense1(l5_flat) # [batch_size, 9*32]

            # reshape d1 to [batch_size, 32, 3, 3]
            d1 = d1.reshape(-1, 32, 3, 3) # [batch_size, 32, 3, 3]

            # upsample d1 to l4m size dont write 11 11 directly
            d1_up = torch.nn.functional.interpolate(d1, size= (l4m.shape[2], l4m.shape[3]), mode='bilinear', align_corners=True)

            # concatenate l4m and d1_up
            l4m_down = self.l4m_pass(l4m) # [batch_size, 128, 11, 11] 
            l4m_d1 = torch.cat((l4m_down, d1_up), dim=1) # [batch_size, 160, 11, 11]
            l4m_d1 = self.l4m_process(l4m_d1) # [batch_size, 128, 11, 11]

            # upsample l4m_d1 to l4 size dont write 23 23 directly
            l4m_d1_up = torch.nn.functional.interpolate(l4m_d1, size= (l4.shape[2], l4.shape[3]), mode='bilinear', align_corners=True)

            # concatenate l4 and l4m_d1_up
            l4_down = self.l4_pass(l4) # [batch_size, 128, 23, 23]
            l4_d1 = grid_features = torch.cat((l4_down, l4m_d1_up), dim=1) # [batch_size, 256, 23, 23]
            l4_d1 = self.l4_process(l4_d1) # [batch_size, 7, 23, 23]
            

            offset = self.offset_head(l4) # [batch_size, 2, 23, 23]


            # combine last two dimensions in grid_features and l4_d1
            grid_features_flat = torch.flatten(grid_features, 2) # [batch_size, 256, 529]
            l4_d1_flat = torch.flatten(l4_d1, 2) # [batch_size, 7, 529]
            
            # now find max value index in l4_d1_flat in dim=2
            grid_indices = torch.argmax(l4_d1_flat, dim=2) # [batch_size, 7]

            # now find the grid features using grid_indices
            grid_features_flat = torch.transpose(grid_features_flat, 1, 2) # [batch_size, 529, 256]
            grid_features_flat = torch.gather(grid_features_flat, 1, grid_indices.unsqueeze(2).repeat(1,1,256)) # [batch_size, 7, 256]

            # now find the class features
            class_features = self.classification1(grid_features_flat.view(-1, 256)) # [batch_size*7, 64]
            class_features = class_features.view(-1, 7, 64) # [batch_size, 7, 64]
            class_features = torch.mean(class_features, dim=1) # [batch_size, 64]

            # now find the classes
            classes = self.classification2(class_features) # [batch_size, class_count]

            # concatenate l4_d1 and offset
            fin = torch.cat((l4_d1, offset), dim=1) # [batch_size, 7, 23, 23]

            return fin, classes

        model._forward_impl = forward

        self.backbone = model

    def forward(self, x):
      x = self.backbone(x)
      return x

def get_model(model_name, class_count):
    
    if model_name == "resnet18_customized":
        model = Resnet18(class_count=class_count)
    elif model_name == "resnet34_removed_avgpool":
        model = Resnet34()

    return model
