import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class BaseLightningModule(pl.LightningModule):
    def __init__(self, lr, plot_path, log_file, **kwargs):
        super(BaseLightningModule, self).__init__()
        self.lr = lr
        self.plot_path = plot_path
        self.log_file = log_file

    def forward(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch):
        x, y = batch
        # apply object boundary to prediction before computing loss
        y_hat = self(x)
        for i in range(y_hat.shape[1]):
            y_hat[:,i,:,:] = torch.where(x[:,-1,:,:] > 0.,
                                         y_hat[:,i,:,:],
                                         torch.zeros_like(y_hat[:,i,:,:].type_as(y_hat))
                                         )

        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        losses = self.custom_loss(y_hat, y, x)
        return losses[0]  # Returning the full MSE loss for monitoring
    
    def custom_loss(self, y_hat, y, x):
        # Apply BCs, find area of domain outside of object, collect losses
        # for mask input type this will be an identity op
        node_count = torch.where(x[:,-1,:,:] > 0., 
                          torch.ones_like(x[:,-1,:,:]).type_as(x), 
                          torch.zeros_like(x[:,-1,:,:]).type_as(x)
                          ).sum((1,2))
        
        y_hat = self(x)
        for i in range(y_hat.shape[1]):
            y_hat[:,i,:,:] = torch.where(x[:,-1,:,:] > 0.,
                                         y_hat[:,i,:,:],
                                         torch.zeros_like(y_hat[:,i,:,:].type_as(y_hat))
                                         )
            
        loss = F.mse_loss(y_hat, y, reduction='none')
        losses = [(loss.sum((1,2,3)) / node_count).mean().item()]
        for idx in range(y_hat.shape[1]):
            losses.append((loss[:,idx].sum((1,2)) / node_count).mean().item())
        
        if y_hat.shape[1] == 3:    # uvp for ns only
            loss_names = ['full', 'u', 'v', 'p']
        if y_hat.shape[1] == 4:    # uvpt for ns+ht
            loss_names = ['full', 'u', 'v', 'p', 't']
        for i, name in enumerate(loss_names):
            self.log(f'val_loss_{name}', losses[i], on_epoch=True)
        return losses
