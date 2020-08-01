## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, image_size=224,depth=1):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
        kernel_size_5= 5
        conv1_channels= 32
        conv1_output_size= image_size - kernel_size_5 +1
        pool1_output_size= int(conv1_output_size/2)
        
        conv2_channels= 48
        conv2_output_size= pool1_output_size - kernel_size_5 + 1
        pool2_output_size= int(conv2_output_size/2)
        
        kernel_size_3= 3
        conv3_channels= 48
        conv3_output_size= pool2_output_size - kernel_size_3 +1
        pool3_output_size= int(conv3_output_size/2)
        
        conv4_channels= 64
        conv4_output_size= pool3_output_size - kernel_size_3 +1
        pool4_output_size= int(conv4_output_size/2)
        
        fc1_channels= 4096
        fc2_channels= 1028
        
        output_channels= 2*68
        
        self.conv1= nn.Conv2d(depth, conv1_channels, kernel_size_5)
        self.conv1_bn= nn.BatchNorm2d(conv1_channels)        
        self.pool1= nn.MaxPool2d(2,2)
        self.dropout1=nn.Dropout2d(p=0.2)
        
        self.conv2= nn.Conv2d(conv1_channels, conv2_channels, kernel_size_5)
        self.conv2_bn= nn.BatchNorm2d(conv2_channels)
        self.pool2= nn.MaxPool2d(2,2)
        self.dropout2= nn.Dropout2d(p=0.2)
        
        self.conv3= nn.Conv2d(conv2_channels, conv3_channels, kernel_size_3)
        self.conv3_bn= nn.BatchNorm2d(conv3_channels)
        self.pool3= nn.MaxPool2d(2,2)
        self.dropout3= nn.Dropout2d(p=0.2)
        
        self.conv4= nn.Conv2d(conv3_channels, conv4_channels, kernel_size_3)
        self.conv4_bn= nn.BatchNorm2d(conv4_channels)
        self.pool4= nn.MaxPool2d(2,2)
        
        self.fc1= nn.Linear(conv4_channels*pool4_output_size*pool4_output_size, fc1_channels)
        self.fc1_bn= nn.BatchNorm1d(fc1_channels)
        
        self.fc1_drop= nn.Dropout(p=0.4)
        
        self.fc2= nn.Linear(fc1_channels, fc2_channels)
        
        self.output= nn.Linear(fc2_channels, output_channels)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = F.relu(self.pool1(self.conv1_bn(self.conv1(x))))
        x = self.dropout1(x)
        x = F.relu(self.pool2(self.conv2_bn(self.conv2(x))))
        x = self.dropout2(x)
        x = F.relu(self.pool3(self.conv3_bn(self.conv3(x))))
        x = self.dropout3(x)
        x = F.relu(self.pool4(self.conv4_bn(self.conv4(x))))
        
        x = x.view(x.size(0),-1)
        
        x= F.relu(self.fc1_bn(self.fc1(x)))
        x= self.fc1_drop(x)
        x= F.relu(self.fc2(x))
        x= self.output(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
