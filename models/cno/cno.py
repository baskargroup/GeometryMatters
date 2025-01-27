from models.base import BaseLightningModule
import torch.nn as nn
from models.cno.CNO.CNOModule import CNO


class cno(BaseLightningModule):
    def __init__(self, in_channels, out_channels, n_layers, in_size, out_size, N_res =1, loss=nn.MSELoss(), lr=1e-3, plot_path='./plots/', log_file='cno_log.txt'):
        super(cno, self).__init__(lr=lr, plot_path=plot_path, log_file=log_file)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.in_size = in_size
        self.out_size = out_size
        self.N_res = N_res
        self.loss = loss
        self.model = CNO(in_dim = self.in_channels, in_size = self.in_size, N_layers = self.n_layers, 
                       out_dim = self.out_channels, out_size = self.out_size, N_res = self.N_res)

    def forward(self, x):
        return self.model(x)

 