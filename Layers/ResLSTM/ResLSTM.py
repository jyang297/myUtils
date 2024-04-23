import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.laplacian import *
from model.warplayer import warp
from model.refine import *
from model.myContext import *
from model.loss import *
from model.myLossset import *

# Attention test
# import model.Attenions as att
import Attenions as att
import Resnet as resnet

class single_conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, stride=1,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels=in_planes,out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN = nn.BatchNorm2d(out_planes)
        self.LeReLU = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.LeReLU(x)
        return x
    
class Extractor_conv(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn1 = single_conv(3,64,3,1,1)
        self.cnn2 = single_conv(64,128)
        self.cnn3 = single_conv(128,128)
        self.cnn4 = single_conv(128,128)
    
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        return x


   
c = 64
class unitConvGRU(nn.Module):
    # Formula:
    # I_t = Sigmoid(Conv(x_t;W_{xi}) + Conv(h_{t-1};W_{hi}) + b_i)
    # F_t = Sigmoid(Conv(x_t;W_{xf}) + Conv(h_{t-1};W_{hi}) + b_i)
    def __init__(self, hidden_dim=128, input_dim=c):
        # 192 = 4*4*12  
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h


class unitConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.conv_f = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_i = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_c = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_o = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, hidden:tuple):
        h_prev, c_prev = hidden
        
        combined = torch.cat([x, h_prev], dim=1)  # concatenate along the channel dimension
        
        f_t = torch.sigmoid(self.conv_f(combined))  # forget gate
        i_t = torch.sigmoid(self.conv_i(combined))  # input gate 
        o_t = torch.sigmoid(self.conv_o(combined))  # output gate
        c_tilde = torch.tanh(self.conv_c(combined))  # candidate cell state
        
        c_t = f_t * c_prev + i_t * c_tilde  # cell state update
        h_t = o_t * torch.tanh(c_t)  # hidden state update

        return h_t, (h_t, c_t)

class fusion_Fimage_hidden(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.bn = nn.BatchNorm2d(num_features=out_planes)
        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.fussConv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.Covn2d(256, 128, kernel_size=3,stride=1,padding=1)
        )
    def forward(self, hidden_state, image_feature):
        fuss = torch.concat([hidden_state, image_feature])
        fuss = self.bn(fuss)
        fuss = self.leakyRelu(fuss)
        return fuss
        

class ResLSTMflowUnit(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.image_extractor = Extractor_conv()
        self.convlstm = unitConvLSTM(hidden_dim=self.hidden_size, input_dim=None)
        self.fison = fusion_Fimage_hidden(in_planes=128+self.hidden_size, out_planes=128)
        self.attention = att.CBAM(in_channel=128)
        self.Res1 = resnet.BasicResNetUnit(in_channels=128, out_channels=128)
        self.Res2 = resnet.NoReluBasicResNetUnit(in_channels=128, out_channels=128)
        self.finalact = nn.LeakyReLU(inplace=True)
        
        
    def forward(self, RRinput, selected_frame, pre_hidden_cell):
        '''
        Unit of ResLSTM. Stack it to finish the backbone.

        Args:
        - selected_frame (torch.Tensor): Tensor of shape (b, n, c, h, w) representing a batch of frame sequences.0<1>2<3>4<5>6
            a) b, n, c, h, w = allframes_N.shape()
        - selected_frame (torch.Tensor): Tensor of shape (b, 1, c, h, w) representing the currently selected frame. 
            1. b, 1, c, h, w = selected_frame
            2. forward:0 -> 2 -> 4
            3. backward:6 -> 4 -> 2
        - hidden_state: Hidden state and cell state for the ConvLSTM.

        '''
        # X_1 = RRinput
        X_2 = self.Res1(RRinput)
        # h_t, (h_t, c_t)
        output, curr_hidden_cell= self.convlstm(X_2, pre_hidden_cell)
        freature_image = self.image_extractor(selected_frame)
        stacked_input = torch.cat([freature_image , output], dim=1)
        fusion_feature = self.fusion(stacked_input)
        X_3= self.attention(fusion_feature)
        X_3_r = self.Res2(X_3)
        X_4 = self.finalact(X_3_r + fusion_feature)
                
            
        return curr_hidden_cell, output, X_4
            
                   
            
class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        self.initializer = Extractor_conv()
        self.forwardLSTM = ResLSTMflowUnit(hidden_size=self.hidden_size)
        self.backwardLSTM = ResLSTMflowUnit(hidden_size=self.hidden_size)
        self.IFnet = IFnet_m()
        
    def forward(self, allframes):
        # allframes 0<1>2<3>4<5>6
        # IFnet module
        b, n, c, h, w = allframes.shape()
        h0 = torch.zeros(n,c,h,w).to(allframes.device)
        c0 = torch.zeros(n,c,h,w).to(allframes.device)
        forward_output_list = []
        backward_output_list = []
        initial_hc = (h0, c0)
        for i in range(3):
            if i == 0:
                # assume allframes:b,n,c,h,w
                f_RRinput = self.initializer(allframes[:,0])
                b_RRinput = self.initializer(allframes[:,-1])
                forward_curr_hidden_cell, forward_output, f_X_4 = self.forwardLSTM(RRinput=f_RRinput, selected_frame=allframes[:,i], pre_hidden_cell=initial_hc)
                backward_curr_hidden_cell, backward_output, b_X_4 = self.backwardLSTM(RRinput=b_RRinput, selected_frame=allframes[:,i], pre_hidden_cell=initial_hc)

            else:
                forward_curr_hidden_cell, forward_output, f_X_4 = self.forwardLSTM(RRinput=f_X_4, selected_frame=allframes[:,i], pre_hidden_cell=forward_curr_hidden_cell)
                backward_curr_hidden_cell, backward_output, b_X_4 = self.backwardLSTM(RRinput=b_X_4, selected_frame=allframes[:,i], pre_hidden_cell=backward_curr_hidden_cell)
                
            forward_output_list.append(forward_output)
            backward_output_list.append(backward_output)

        for i in range(3):
            all_ = self.IFnet(allframes[:,i:i+2], forward_output_list[i], backward_output_list[-i-1])

        return None, all_loss
        