import torch.nn as nn
from models.base import BaseLightningModule
from models.wno.network import WNO2d


class WNO(BaseLightningModule):
    def __init__(self, in_channels, out_channels, shape, level, width, loss=nn.MSELoss(), lr=1e-3, plot_path='./plots/wno', log_file='WNO_log.txt'):
        super(WNO, self).__init__(lr=lr, plot_path=plot_path, log_file=log_file)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = tuple(shape)
        self.level = level
        self.width = width
        self.loss = loss
        self.model = WNO2d(in_channels=self.in_channels, out_channels=self.out_channels, width=self.width, level=self.level, shape=self.shape)

    def forward(self, x):
        return self.model(x)
