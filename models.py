## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)    # Output 220 x 220
        self.maxpool = nn.MaxPool2d(2, 2)   # Output 110 x 110 after pooling
        self.drop1 = nn.Dropout2d(0.3)
        
        self.conv2 = nn.Conv2d(32, 64, 4)   # Output 107 x 107
        self.drop2 = nn.Dropout2d(0.3)      # Output 53 x 53 after pooling
        
        self.conv3 = nn.Conv2d(64, 128, 3)  # Output 51 x 51
        self.drop3 = nn.Dropout2d(0.4)      # Output 25 x 25 after pooling
        
        self.conv4 = nn.Conv2d(128, 256, 3) # Output 24 x 24
        self.drop4 = nn.Dropout2d(0.4)      # Output 12 x 12 after pooling
        
        self.fc1 = nn.Linear(36864, 1000)   # Output 
        self.drop5 = nn.Dropout2d(0.5)
        
        self.fc2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout2d(0.6)
        
        self.fc3 = nn.Linear(1000, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.maxpool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
