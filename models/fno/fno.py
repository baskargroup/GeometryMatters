import torch.nn as nn
from models.base import BaseLightningModule
from neuralop.models import TFNO


class FNO(BaseLightningModule):
    def __init__(self, n_modes, in_channels, out_channels, hidden_channels, projection_channels, n_layers, loss=nn.MSELoss(), lr=1e-4, plot_path='./plots/', log_file='fno_log.txt'):
        super(FNO, self).__init__(lr=lr, plot_path=plot_path, log_file=log_file)
        self.n_modes = n_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.projection_channels = projection_channels
        self.n_layers = n_layers
        self.loss = loss
        self.model = TFNO(n_modes=self.n_modes, hidden_channels=self.hidden_channels,
                        in_channels=self.in_channels, out_channels=self.out_channels,
                        projection_channels=self.projection_channels, n_layers=self.n_layers)

    def forward(self, x):
        return self.model(x)