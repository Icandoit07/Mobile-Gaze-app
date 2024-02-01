import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class EyeTrackerImageModel(nn.Module):
    def __init__(self):
        super(EyeTrackerImageModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0), 
            nn.BatchNorm2d(32, momentum=0.8), 
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2), 
            nn.Dropout(0.08),  

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0), 
            nn.BatchNorm2d(64, momentum=0.8),  
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),  
            nn.Dropout(0.08), 

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.08)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x

class FaceTotalImageModel(nn.Module):
    def __init__(self):
        super(FaceTotalImageModel, self).__init__()
        self.conv = EyeTrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),  
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceTotalGridModel(nn.Module):
    def __init__(self, gridSize = 25):
        super(FaceTotalGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),  
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EyeTrackerModel(nn.Module):
    def __init__(self):
        super(EyeTrackerModel, self).__init__()
        self.eyeModel = EyeTrackerImageModel()
        self.faceModel = FaceTotalImageModel()
        self.gridModel = FaceTotalGridModel()
        
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2 * 128 * 12 * 12, 256), 
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 + 64 + 32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2), 
        )

    def forward(self, x_in):
        xEyeL = self.eyeModel(x_in["left"])
        xEyeR = self.eyeModel(x_in["right"])

        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        xFace = self.faceModel(x_in["face"])
        xGrid = self.gridModel(x_in["grid"])

        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)
        
        return x

if __name__ == '__main__':
    m = EyeTrackerModel()
    feature = {
        "face": torch.zeros(3, 3, 128, 128), 
        "left": torch.zeros(3, 3, 128, 128), 
        "right": torch.zeros(3, 3, 128, 128), 
        "grid": torch.zeros(3, 1, 25, 25)
    }
    a = m(feature)
    print(a) 
