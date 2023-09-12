import torch
import torchvision
import cv2
import numpy as np


def get_model_2():

    # return new resnet18 model
    model = torchvision.models.resnet18(pretrained=True)

    # change the last layer to have 8 outputs
    model.fc = torch.nn.Linear(512, 8)

    return model

def get_model_3():

    # return new mobilenet v3 large
    model = torchvision.models.mobilenet_v3_large(pretrained=True)

    # change the last layer to have 8 outputs
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 8)

    return model


def get_model_4():

    # return new efficientnet b2
    model = torchvision.models.efficientnet_b2(pretrained=True)

    # change the last layer to have 8 outputs
    model.classifier = torch.nn.Linear(model.classifier[-1].in_features, 8)

    return model


def get_model():

    # return new efficientnet b2
    model = torchvision.models.efficientnet_b3(pretrained=True)

    # change the last layer to have 8 outputs but add some layers and dropouts before, dropout 0.2
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.1),
        torch.nn.Linear(model.classifier[-1].in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 8),
    )

    return model
