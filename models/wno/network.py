import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT, IDWT

class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, shape, device='cuda'):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse DWT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.shape = shape
        self.device = device

        self.dwt = DWT(J=self.level, mode='symmetric', wave='db4').to(device)
        self.idwt = IDWT(mode='symmetric', wave='db4').to(device)
        
        dummy_data = torch.randn(1, in_channels, *shape).to(device)
        self.mode_data, _ = self.dwt(dummy_data)
        self.modes1 = self.mode_data.shape[-2]
        self.modes2 = self.mode_data.shape[-1]

        # Parameter initialization
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        input_shape = x.shape[-2:]

        # Adjust the wavelet decomposition level dynamically if necessary
        if input_shape != self.shape:
            factor_w = int(torch.log2(torch.tensor(input_shape[-2] / self.shape[-2])).item())
            factor_h = int(torch.log2(torch.tensor(input_shape[-1] / self.shape[-1])).item())
            level_adjustment = max(factor_w, factor_h)

            dwt = DWT(J=self.level + level_adjustment, mode='symmetric', wave='db4').to(self.device)
        else:
            dwt = self.dwt

        x_ft, x_coeff = dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = self.mul2d(x_ft, self.weights1)
        
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:, :, 0, :, :] = self.mul2d(x_coeff[-1][:, :, 0, :, :].clone(), self.weights2)
        x_coeff[-1][:, :, 1, :, :] = self.mul2d(x_coeff[-1][:, :, 1, :, :].clone(), self.weights3)
        x_coeff[-1][:, :, 2, :, :] = self.mul2d(x_coeff[-1][:, :, 2, :, :].clone(), self.weights4)
        
        # Return to physical space
        x = self.idwt((out_ft, x_coeff))
        return x
    
    
class WNO2d(nn.Module):
    def __init__(self, width, level, shape, in_channels, out_channels, device='cuda'):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of the last layer using self.fc1 and self.fc2.
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, channels, height, width)
        output: the solution 
        output shape: (batchsize, channels, height, width)
        """
        self.level = level
        self.width = width
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.padding = 1
        self.fc0 = nn.Linear(in_channels, self.width)
        self.pad = shape[0] % 2 != 0

        self.conv0 = WaveConv2d(self.width, self.width, self.level, shape, device)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, shape, device)
        self.conv2 = WaveConv2d(self.width, self.width, self.level, shape, device)
        self.conv3 = WaveConv2d(self.width, self.width, self.level, shape, device)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 192)
        self.fc2 = nn.Linear(192, out_channels)

    def forward(self, x):

        batch_size, channels, height, width = x.shape
        
        # Flatten spatial dimensions and apply the first fully connected layer
        x = x.view(batch_size * height * width, channels)  # Flatten spatial dimensions for fc0
        
        # Pass through fc0 and reshape back to [batch, height, width, width] after fc0
        x = self.fc0(x)
        x = x.view(batch_size, height, width, self.width)
        x = x.permute(0, 3, 1, 2)  # Rearrange to [batch, width, height, width]

        if self.pad:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        if self.pad:
            x = x[..., :-self.padding, :-self.padding]

        # Reshape for fc1: [batch_size, height * width, width]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.width)
        
        x = self.fc1(x)  # Shape should now be [batch_size, height * width, fc1_out_features]
        x = F.gelu(x)
        x = self.fc2(x)  # Final output should be [batch_size, height * width, out_channels]
        
        # Reshape back to [batch_size, out_channels, height, width]
        x = x.view(batch_size, height, width, self.out_channels).permute(0, 3, 1, 2)
        
        return x
