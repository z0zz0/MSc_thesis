import torch
import torch.nn as nn
from ilm import ilm_IN

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
        self.ilm_in1 = ilm_IN(channels=5)
        self.conv2 = nn.Conv3d(5, 10, 5, padding=(2, 2, 2))
        #ReLU
        #pooling
        self.ilm_in2 = ilm_IN(channels=10)

        self.conv3 = nn.Conv3d(10, 15, 3, padding=(1, 1, 1))
        self.ilm_in3 = ilm_IN(channels=15)
        self.conv4 = nn.Conv3d(15, 20, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.ilm_in4 = ilm_IN(channels=20)

        self.conv5 = nn.Conv3d(20, 40, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.ilm_in5 = ilm_IN(channels=40)

        self.conv6 = nn.Conv3d(40, 80, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.ilm_in6 = ilm_IN(channels=80)

        self.conv7 = nn.Conv3d(80, 160, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.ilm_in7 = ilm_IN(channels=160)

        self.conv8 = nn.Conv3d(160, 320, 3, padding=(1, 1, 1))
        #ReLU
        #pooling
        self.ilm_in8 = ilm_IN(channels=320)

        #flatten
        #Dropout3d
        
        self.fc1 = nn.Linear(320*2*2*1, 65)
        self.ln9 = nn.LayerNorm(65) ## Layer Norm has the same functionality as Instance Norm when it comes to 1d data. InstanceNorm1d function is not suitable for flattened 1d data
                                    ## LN is thus used in the fully connected layers.
        self.fc2 = nn.Linear(65, 40)
        self.ln10 = nn.LayerNorm(40)
        
        self.fc3 = nn.Linear(40, 20)
        self.ln11 = nn.LayerNorm(20)
        #Dropout
        self.fc4 = nn.Linear(20, 3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.ilm_in1(x)
        
        x = self.pool_stride_2(self.relu1(self.conv2(x)))
        x = self.ilm_in2(x)
        
        x = self.conv3(x)
        x = self.ilm_in3(x)
        
        x = self.pool_stride_2(self.relu2(self.conv4(x)))
        x = self.ilm_in4(x)
        
        x = self.pool_stride_2(self.relu3(self.conv5(x)))
        x = self.ilm_in5(x)
        
        x = self.pool_stride_2(self.relu4(self.conv6(x)))
        x = self.ilm_in6(x)
        
        x = self.pool_stride_2(self.relu5(self.conv7(x)))
        x = self.ilm_in7(x)
        
        x = self.pool_stride_2(self.relu6(self.conv8(x)))
        x = self.ilm_in8(x)
        
        x = self.droput_3d_05(x)
        
        x = x.view(-1, 1280)
        x = self.relu7(self.fc1(x))
        x = self.ln9(x)
        x = self.relu8(self.fc2(x))
        x = self.ln10(x)
        x = self.relu9(self.fc3(x))
        x = self.ln11(x)
        
        x = self.droput_1d_02(x)
        
        x = self.fc4(x)
        
        return x