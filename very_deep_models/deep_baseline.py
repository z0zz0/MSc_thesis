import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.pool_stride_2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=2) # (size, stride)
        self.pool_stride_1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=1) # (size, stride)
        
        self.droput_3d_05 = nn.Dropout3d(p=0.5)
        self.droput_1d_02 = nn.Dropout(p=0.25)
        
        self.conv1 = nn.Conv3d(1, 5, 5, padding=(2, 2, 2)) #(input channels, ouput channels (no. of filters), kernel size)
        self.relu1 = nn.ReLU()
        #pooling 1 stride, 2x2x2.
        
        self.conv2 = nn.Conv3d(5, 5, 5, padding=(2, 2, 2))
        self.relu2 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv3 = nn.Conv3d(5, 5, 3, padding=(1, 1, 1))
        self.relu3 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv4 = nn.Conv3d(5, 5, 3, padding=(1, 1, 1))
        self.relu4 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv5 = nn.Conv3d(5, 5, 3, padding=(1, 1, 1))
        self.relu5 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv6 = nn.Conv3d(5, 10, 3, padding=(1, 1, 1))
        self.relu6 = nn.ReLU()
        #pooling 2 stride, 2x2x2. ### 
        
        self.conv7 = nn.Conv3d(10, 10, 3, padding=(1, 1, 1))
        self.relu7 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv8 = nn.Conv3d(10, 10, 3, padding=(1, 1, 1))
        self.relu8 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv9 = nn.Conv3d(10, 10, 3, padding=(1, 1, 1))
        self.relu9 = nn.ReLU()
        #pooling 1 stride, 2x2x2.
        
        self.conv10 = nn.Conv3d(10, 10, 3, padding=(1, 1, 1))
        self.relu10 = nn.ReLU()
        #pooling 1 stride, 2x2x2.
        
        self.conv11 = nn.Conv3d(10, 10, 3, padding=(1, 1, 1))
        self.relu11 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv12 = nn.Conv3d(10, 20, 3, padding=(1, 1, 1))
        self.relu12 = nn.ReLU()
        #pooling 2 stride, 2x2x2. ###
        
        self.conv13 = nn.Conv3d(20, 20, 3, padding=(1, 1, 1))
        self.relu13 = nn.ReLU()
        #pooling 1 stride, 2x2x2.
        
        self.conv14 = nn.Conv3d(20, 20, 3, padding=(1, 1, 1))
        self.relu14 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv15 = nn.Conv3d(20, 20, 3, padding=(1, 1, 1))
        self.relu15 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv16 = nn.Conv3d(20, 20, 3, padding=(1, 1, 1))
        self.relu16 = nn.ReLU()
        #pooling 1 stride, 2x2x2.
        
        self.conv17 = nn.Conv3d(20, 20, 3, padding=(1, 1, 1))
        self.relu17 = nn.ReLU()
        #pooling 1 stride, 2x2x2. 
        
        self.conv18 = nn.Conv3d(20, 40, 3, padding=(1, 1, 1))
        self.relu18 = nn.ReLU()
        #pooling 2 stride, 2x2x2. ###
        
        self.conv19 = nn.Conv3d(40, 80, 3, padding=(1, 1, 1))
        self.relu19 = nn.ReLU()
        #pooling 2 stride, 2x2x2. ### 
        
        #flatten
        #Dropout3d
        
        self.fc1 = nn.Linear(80*7*7*3, 65)
        self.relu20 = nn.ReLU()
        self.fc2 = nn.Linear(65, 40)
        self.relu21 = nn.ReLU()
        self.fc3 = nn.Linear(40, 20)
        self.relu22 = nn.ReLU()
        #Dropout
        self.fc4 = nn.Linear(20, 3)
    
    def forward(self, x):
        x = self.pool_stride_1(self.relu1(self.conv1(x)))
        
        x = self.pool_stride_1(self.relu2(self.conv2(x)))
        
        x = self.pool_stride_1(self.relu3(self.conv3(x)))
        
        x = self.pool_stride_1(self.relu4(self.conv4(x)))
        
        x = self.pool_stride_1(self.relu5(self.conv5(x)))
        
        x = self.pool_stride_2(self.relu6(self.conv6(x)))
        
        x = self.pool_stride_1(self.relu7(self.conv7(x)))
        
        x = self.pool_stride_1(self.relu8(self.conv8(x)))
        
        x = self.pool_stride_1(self.relu9(self.conv9(x)))
        
        x = self.pool_stride_1(self.relu10(self.conv10(x)))
        
        x = self.pool_stride_1(self.relu11(self.conv11(x)))
        
        x = self.pool_stride_2(self.relu12(self.conv12(x)))
        
        x = self.pool_stride_1(self.relu13(self.conv13(x)))
        
        x = self.pool_stride_1(self.relu14(self.conv14(x)))
        
        x = self.pool_stride_1(self.relu15(self.conv15(x)))
        
        x = self.pool_stride_1(self.relu16(self.conv16(x)))
        
        x = self.pool_stride_1(self.relu17(self.conv17(x)))
        
        x = self.pool_stride_2(self.relu18(self.conv18(x)))
        
        x = self.pool_stride_2(self.relu19(self.conv19(x)))
        
        x = self.droput_3d_05(x)
        
        x = x.view(-1, 11760)
        x = self.relu20(self.fc1(x))
        x = self.relu21(self.fc2(x))
        x = self.relu22(self.fc3(x))
        
        x = self.droput_1d_02(x)
        
        x = self.fc4(x)
        
        return x