from typing import Optional

import torch
from torch import nn

class ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      bias = False),
            nn.BatchNorm1d(in_channels), 
            nn.PReLU(),
            nn.Conv1d(in_channels = out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      bias = False),
            nn.BatchNorm1d(out_channels)
        )
        self.stage2 = nn.Sequential(
            nn.PReLU(),
            nn.MaxPool1d(3)
        )

    def forward(self, x):
        x1 = self.stage1(x)
        if self.in_channels == self.out_channels:
            x1 = x + x1
        return self.stage2(x1)
        

class TCN_Layer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_channels: int,
                 kernel_size: int,
                 dilation: int,
                 spk_dim: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spk_dim = spk_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = (kernel_size - 1)*dilation//2
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels + spk_dim if spk_dim else in_channels, conv_channels//4, 1),
            nn.PReLU(), nn.GroupNorm(1, conv_channels//4),
            nn.Conv1d(conv_channels//4, conv_channels//4,
                      kernel_size = kernel_size,
                      dilation = dilation,
                      padding = padding,
                      groups = 1),
            nn.PReLU(), nn.GroupNorm(1, conv_channels//4),
            nn.Conv1d(conv_channels//4, in_channels, 1))
        
    def forward(self, x: torch.Tensor):
        x_= self.seq(x)
        if self.spk_dim is not None:
          x_ = x_ + x[:, :self.in_channels, :]
        else:
          x_ = x_ + x
        return x_

class TCN_Block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_channels: int,
                 kernel_size: int,
                 num_layers: int,
                 start_idx: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.num_layers = num_layers
        self.start_idx = start_idx
        dilations = [2**i for i in range(num_layers)]
        for i in range(start_idx, num_layers):
            setattr(self, f'tcn_layer{i}', TCN_Layer(in_channels,
                                                     conv_channels,
                                                     kernel_size = kernel_size,
                                                     dilation = dilations[i]))
    def forward(self, x: torch.Tensor):
        for i in range(self.start_idx, self.num_layers):
            x = getattr(self, f"tcn_layer{i}")(x)
        return x
    
class Stacked_TCNs(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_channels: int,
                 kernel_size: int,
                 num_blks: int,
                 num_tcn_layers: int,
                 return_intermidate: bool = False):
        """
        in_channels, int = C + Dspk
        """
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.num_blks = num_blks
        self.return_intermidate = return_intermidate
        for i in range(num_blks):
            setattr(self, f'tcn_blk{i}', TCN_Block(in_channels,
                                                   conv_channels,
                                                   kernel_size = kernel_size,
                                                   num_layers = num_tcn_layers))

    def forward(self, x: torch.Tensor):
        out = []
        for i in range(self.num_blks):
            x = getattr(self, f'tcn_blk{i}')(x)
            if self.return_intermidate:
                out.append(x)
        if self.return_intermidate:
            return out
        else:
            return [x]
    
class GatedConvFusion(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 spk_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spk_dim = spk_dim
        self.in_proj = nn.Conv1d(in_channels, out_channels//4, 1)
        self.branch1 = nn.Sequential(nn.PReLU(),
                                     nn.GroupNorm(1, out_channels//4, eps = 1e-08),
                                     nn.Conv1d(out_channels//4, out_channels//4,
                                               kernel_size = 3,
                                               padding = 1,
                                               groups = 1))
        self.branch2 = nn.Sequential(nn.PReLU(),
                                     nn.GroupNorm(1, out_channels//4, eps = 1e-08),
                                     nn.Conv1d(out_channels//4, out_channels//4,
                                               kernel_size = 3,
                                               padding = 1,
                                               groups = 1))
        self.out_proj = nn.Sequential(nn.PReLU(),
                                      nn.GroupNorm(1, out_channels//4, eps = 1e-08),
                                      nn.Conv1d(out_channels//4, in_channels, 1))
        self.linear = nn.Linear(spk_dim, in_channels)
    
    def forward(self, 
                x: torch.Tensor, 
                aux: torch.Tensor):
        """
        x - torch.Tensor, shape (N, C, T)
        aux - torch.Tensor, shape (N, Dspk) 
        """
        x_ = x*self.linear(aux).unsqueeze(-1)
        x_ = self.in_proj(x_)
        x_ = self.out_proj(torch.sigmoid(self.branch1(x_))*self.branch2(x_))
        return x + x_
    
class Fusion_Stacked_TCNs(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 conv_channels: int,
                 spk_dim: int,
                 kernel_size: int,
                 num_tcn_layers: int,
                 num_blks: int):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spk_dim = spk_dim
        self.kernel_size = kernel_size
        self.num_tcn_layers = num_tcn_layers
        self.num_blks = num_blks
        for i in range(num_blks):
            setattr(self, f'fusion{i}', GatedConvFusion(in_channels, 
                                                        conv_channels,
                                                        spk_dim))
            setattr(self, f'tcn_blk{i}', TCN_Block(in_channels,
                                                   conv_channels,
                                                   kernel_size = kernel_size,
                                                   num_layers = num_tcn_layers,
                                                   start_idx = 1))
            
    def forward(self, x: torch.Tensor, aux: torch.Tensor):
        """
        x - torch.Tensor, shape (N*(k+1), C, T)
        aux - torch.Tensor, shape (N, k+1, Dspk) 
        """
        NK, C, T = x.shape
        N, K = aux.shape[:2]
        aux = aux.reshape(NK, -1)
        for i in range(self.num_blks):
            x = getattr(self, f'fusion{i}')(x, aux)
            x = getattr(self, f'tcn_blk{i}')(x)
        return x

class DownSample(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = 2

        padding1 = int((kernel_size - 1)*self.dilation/2)
        padding2 = int((kernel_size - stride + 1)/2)
        self.dilated_conv1 = nn.Conv1d(in_channels = in_channels,
                                       out_channels = in_channels,
                                       kernel_size = kernel_size,
                                       padding = padding1,
                                       dilation = self.dilation,
                                       bias = False)
        self.bn11 = nn.BatchNorm1d(in_channels)
        self.prelu1 = nn.PReLU()

        self.conv1 = nn.Conv1d(in_channels = in_channels,
                               out_channels = in_channels*2,
                               kernel_size = 1)
        self.glu1 = nn.GLU(dim = 1)
        self.bn12 = nn.BatchNorm1d(in_channels)

        self.conv2 = nn.Conv1d(in_channels = in_channels,
                               out_channels = in_channels,
                               kernel_size = 1,
                               bias = False)
        self.bn2 = nn.BatchNorm1d(in_channels)

        self.conv = nn.Conv1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              padding = padding2,
                              stride = stride)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor):
        xb = x.clone()
        # brach 1
        x = self.prelu1(self.bn11(self.dilated_conv1(x)))
        x = self.bn12(self.glu1(self.conv1(x)))
        # brach 2
        xb = self.bn2(self.conv2(xb))

        x = x + xb
        x = self.prelu(self.conv(x))
        return x

class UpSample(nn.Module):
    def __init__(self,
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = 2

        padding1 = int((kernel_size - 1)*self.dilation/2)
        padding2 = int((kernel_size - stride + 1)/2)
        self.dilated_conv1 = nn.Conv1d(in_channels = in_channels,
                                       out_channels = in_channels,
                                       kernel_size = kernel_size,
                                       padding = padding1,
                                       dilation = self.dilation,
                                       bias = False)
        self.bn11 = nn.BatchNorm1d(in_channels)
        self.prelu1 = nn.PReLU()

        self.conv1 = nn.Conv1d(in_channels = in_channels,
                               out_channels = in_channels*2,
                               kernel_size = 1)
        self.glu1 = nn.GLU(dim = 1)
        self.bn12 = nn.BatchNorm1d(in_channels)

        self.conv2 = nn.Conv1d(in_channels = in_channels,
                               out_channels = in_channels,
                               kernel_size = 1,
                               bias = False)
        self.bn2 = nn.BatchNorm1d(in_channels)

        self.conv = nn.ConvTranspose1d(in_channels = in_channels,
                                       out_channels = out_channels,
                                       kernel_size = kernel_size,
                                       padding = padding2,
                                       stride = stride,
                                       output_padding = 1)
        self.prelu = nn.PReLU()

    def cat(self, x, aux):
        out = torch.full([x.shape[0], x.shape[1] + aux.shape[1], max(x.shape[2], aux.shape[2])], 0,
                         device = x.device, dtype = x.dtype)
        out[:, :x.shape[1], :x.shape[2]] = x
        out[:, x.shape[1]:, :aux.shape[2]] = aux
        return out

    def forward(self, x: torch.Tensor, aux: torch.Tensor):
        x = self.cat(x, aux)
        
        xb = x.clone()
        # brach 1
        x = self.prelu1(self.bn11(self.dilated_conv1(x)))
        x = self.bn12(self.glu1(self.conv1(x)))
        # brach 2
        xb = self.bn2(self.conv2(xb))

        x = x + xb
        x = self.prelu(self.conv(x))
        return x