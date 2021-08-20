import torch
import torch.nn as nn
import torch.nn.functional as F
from ilm import fsc_BN3d

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()
        self.relu9 = nn.ReLU()
        
        self.pool_stride_2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=2) # (size, stride)
        self.droput_3d_05 = nn.Dropout3d(p=0.5)
        self.droput_1d_02 = nn.Dropout(p=0.25)
        
        self.conv1 = nn.Conv3d(1, 5, 5, padding=(2, 2, 2)) #(input channels, ouput channels (no. of filters), kernel size)
        self.bn1 = fsc_BN3d(5, 2)
        self.conv2 = nn.Conv3d(5, 10, 5, padding=(2, 2, 2))
        #ReLU
        #pooling
        self.bn2 = fsc_BN3d(10, 2)

        self.conv3 = nn.Conv3d(10, 15, 3, padding=(1, 1, 1))
        self.bn3 = fsc_BN3d(15, 2)
        self.conv4 = nn.Conv3d(15, 20, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.bn4 = fsc_BN3d(20, 4)

        self.conv5 = nn.Conv3d(20, 40, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.bn5 = fsc_BN3d(40, 4)

        self.conv6 = nn.Conv3d(40, 80, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.bn6 = fsc_BN3d(80, 8)

        self.conv7 = nn.Conv3d(80, 160, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.bn7 = fsc_BN3d(160, 16)

        self.conv8 = nn.Conv3d(160, 320, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.bn8 = fsc_BN3d(320, 32)

        #flatten
        #Dropout3d
        
        self.fc1 = nn.Linear(320*2*2*1, 65)
        self.bn9 = nn.BatchNorm1d(65)
        
        self.fc2 = nn.Linear(65, 40)
        self.bn10 = nn.BatchNorm1d(40)
        
        self.fc3 = nn.Linear(40, 20)
        self.bn11 = nn.BatchNorm1d(20)
        #Dropout
        self.fc4 = nn.Linear(20, 3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.pool_stride_2(self.relu1(self.conv2(x)))
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = self.pool_stride_2(self.relu2(self.conv4(x)))
        x = self.bn4(x)
        
        x = self.pool_stride_2(self.relu3(self.conv5(x)))
        x = self.bn5(x)
        
        x = self.pool_stride_2(self.relu4(self.conv6(x)))
        x = self.bn6(x)
        
        x = self.pool_stride_2(self.relu5(self.conv7(x)))
        x = self.bn7(x)
        
        x = self.pool_stride_2(self.relu6(self.conv8(x)))
        x = self.bn8(x)
        
        x = self.droput_3d_05(x)
        
        x = x.view(-1, 1280)
        x = self.relu7(self.fc1(x))
        x = self.bn9(x)
        x = self.relu8(self.fc2(x))
        x = self.bn10(x)
        x = self.relu9(self.fc3(x))
        x = self.bn11(x)
        
        x = self.droput_1d_02(x)
        
        x = self.fc4(x)
        
        return x