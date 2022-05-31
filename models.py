import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F




# Resnet50_2
def create_Resnet50_2(device):
    """
    Builds resnet_50 network

    Residual Networks : https://arxiv.org/pdf/1512.03385.pdf
    """
    resnet20 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    return resnet20.to(device)




# Wide Resnet50_2
def create_wideResnet50_2(device):
    """
    Builds wideresnet_50_2 network

    Wide Residual Networks : https://arxiv.org/abs/1605.07146
    """
    wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
    return wide_resnet50_2.to(device)




def create_vgg16(device):
    """
    Builds VGG16 network
    """

    vgg16 = torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar10_vgg16_bn', pretrained=True)
    return vgg16.to(device)



class DDSA(nn.Module):
    """Version simplifiée de l'autoencodeur DDSA
    
    DDSA : https://ieeexplore.ieee.org/document/8890816
    """

    def __init__(self):
        super(DDSA, self).__init__()
        
        # encoder
        self.enc1 = nn.Conv2d(3,128,3)
        self.enc2 = nn.Conv2d(128,64,3)
        self.enc3 = nn.Conv2d(64,32,3)

        self.dec1 = nn.ConvTranspose2d(32, 64, 3)
        self.dec2 = nn.ConvTranspose2d(64, 64, 3)
        self.dec3 = nn.ConvTranspose2d(64, 128, 3)
        self.final_layer = nn.ConvTranspose2d(64, 3, 3)

        self.bn = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4608,2048)
        self.linear2 = nn.Linear(2048,2048)
        self.linear3 = nn.Linear(2048,4608)
        
        
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.bn(x)
        x = F.relu(self.enc2(x))
        x = F.dropout(x,p=0.25)
        x, indices = nn.MaxPool2d(2, stride=2,return_indices=True)(x)
        x = F.relu(self.enc3(x))
        # x= nn.MaxPool2d(2, stride=2)(x)



        x = self.flatten(x)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)



        x2 = torch.reshape(x, (x.shape[0], 32, 12, 12))
        x = F.relu(self.dec1(x2))
        # x = nn.MaxUnpool2d(2, stride=2)(x)
        x = nn.MaxUnpool2d(2, stride=2)(x,indices=indices)

        x = F.relu(self.dec2(x))
        # x = F.relu(self.dec3(x))
        x = self.final_layer(x)

        x = F.sigmoid(x)

        return x
    


def create_ddsa(device):
    return  DDSA().to(device)