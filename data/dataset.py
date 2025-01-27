import torch
from torch.utils.data import Dataset
import numpy as np

class LidDrivenDataset(Dataset):
    """
    Custom dataset for loading and processing Lid Driven Cavity problem data from .npz files.
    """
    def __init__(self, file_path_x, file_path_y, data_type=None, inputs=None, geometric_deeponet=False):
        """
        Initializes the dataset with the paths to the .npz files and an optional transform.
        
        Args:
            file_path_x (str): Path to the .npz file containing the input data.
            file_path_y (str): Path to the .npz file containing the target data.
        """
        # Load data from .npz files
        x = np.load(file_path_x)['data']  # shape [num_samples, [Re, SDF, Mask], 512, 512]
        y = np.load(file_path_y)['data']  # shape [num_samples, [u, v, p, c_d, c_l], 512, 512]
    
        if not geometric_deeponet:
            if inputs == 'sdf':    
                x = x[:,:2,:,:]
                y = y[:,:3,:,:]  # only selecting u, v, p channels (actual field solutions, excluding c_d, c_l)
            elif inputs == 'mask':
                x = x[:,[0,-1],:,:]
                x[:,-1,:,:] = x[:,-1,:,:] / 255.  # the mask wasn't normalized when creating dataset 
                y = y[:,:3,:,:]  # only selecting u, v, p channels (actual field solutions, excluding c_d, c_l)
                
        if data_type == 'field':
            self.collocation = False
        elif data_type == 'collocation':
            self.collocation = True
            self.resolution = x.shape[-1]

        # Convert numpy arrays to PyTorch tensors
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the sample and its label at the specified index.
        
        Args:Returns:
            tuple: (sample, target) where sample is the input data and target is the expected output.
        """
        sample = self.x[idx]
        target = self.y[idx]
        
        if self.collocation:
            grid = self.get_grid()
            sample = torch.cat((sample, grid), dim=0)
        
        return sample, target
    
    def get_grid(self):
        # Create the uniform grid for x and y locations of the input
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, self.resolution), 
            np.linspace(0, 1, self.resolution)
        )
        # Stack the grids along the last dimension to get (res, res, 2) shape
        grid = np.stack([grid_x, grid_y], axis=-1)
        # Transpose to get shape (2, res, res) instead of (res, res, 2)
        grid = grid.transpose(2, 0, 1)
        # Convert to PyTorch tensor
        grid = torch.tensor(grid, dtype=torch.float)
        return grid
