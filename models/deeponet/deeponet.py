import torch.nn as nn
from models.base import BaseLightningModule
from models.deeponet.network import DeepONet2D

class DeepONet(BaseLightningModule):
    def __init__(self, input_channels_func, input_channels_loc, out_channels, branch_net_layers, trunk_net_layers, modes, loss=nn.MSELoss(), lr=1e-4, plot_path='./plots/', log_file='DeepONet_log.txt'):
        super(DeepONet, self).__init__(lr=lr, plot_path=plot_path, log_file=log_file)
        self.input_channels_func = input_channels_func
        self.input_channels_loc = input_channels_loc
        self.output_channels = out_channels
        self.branch_net_layers = branch_net_layers
        self.trunk_net_layers = trunk_net_layers
        self.modes = modes
        self.loss = loss
        self.model = DeepONet2D(input_channels_func = self.input_channels_func, input_channels_loc = self.input_channels_loc, 
                                output_channels = self.output_channels, branch_net_layers = self.branch_net_layers,
                                trunk_net_layers = self.trunk_net_layers, modes = self.modes)


    def forward(self, x):
        x1 = x[:,:self.input_channels_func,:,:] # x[:,:,:,:3]# function values [Re, SDF, mask]
        x2 = x[:,self.input_channels_func:,:,:] #x[:,:,:,3:] # grid points [x,y]
        return self.model(x1, x2)

 
