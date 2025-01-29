import os
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import Trainer
from models.fno.fno import FNO
from models.cno.cno import cno
from models.uno.uno import UNO
from models.wno.wno import WNO
from models.deeponet.deeponet import DeepONet
from models.geometric_deeponet.geometric_deeponet import GeometricDeepONet
from data.partial_dataset2 import LidDrivenDataset
from residual_m3_new import ResidualLoss

torch.cuda.empty_cache()


def load_model(model_name, checkpoint_path, config):
    params = {k: v for k, v in config.model.items()}
    model_dict = {
        'fno': FNO,
        'cno': cno,
        'uno': UNO,
        'wno': WNO,
        'deeponet': DeepONet,
        'geometric-deeponet': GeometricDeepONet
    }
    if model_name not in model_dict:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_dict[model_name].load_from_checkpoint(checkpoint_path, **params)

def plot_velocity_error_and_residuals(idx, error_x, error_y, residual_x, residual_y, plot_dir, sdf):
    """
    Plot the error in velocity derivatives and residuals in x and y directions in log scale with SDF mask applied.
    """
    # Extract the first channel for plotting (convert to 2D)
    error_x_2d = error_x[0]
    error_y_2d = error_y[0]
    residual_x_2d = residual_x[0]
    residual_y_2d = residual_y[0]

    # Add a small value to avoid log of zero
    epsilon = 1e-10
    error_x_log = np.log10(error_x_2d + epsilon)
    error_y_log = np.log10(error_y_2d + epsilon)
    residual_x_log = np.log10(residual_x_2d + epsilon)
    residual_y_log = np.log10(residual_y_2d + epsilon)

    # Adjust the SDF mask to match the error shapes (trim edges if necessary)
    sdf_mask = sdf[0, :error_x_log.shape[0], :error_x_log.shape[1]].cpu().numpy() > 0

    # Ensure the mask shape matches the error array shapes exactly
    sdf_mask = sdf_mask[0]
    sdf_mask = sdf_mask[:error_x_log.shape[0], :error_x_log.shape[1]]

    # Apply the SDF mask
    error_x_masked = np.ma.masked_where(~sdf_mask, error_x_log)
    error_y_masked = np.ma.masked_where(~sdf_mask, error_y_log)
    residual_x_masked = np.ma.masked_where(~sdf_mask, residual_x_log)
    residual_y_masked = np.ma.masked_where(~sdf_mask, residual_y_log)

    # X-direction velocity error in log scale with SDF mask applied
    plt.figure()
    plt.imshow(error_x_masked, cmap='jet', origin='lower', vmin=-8, vmax=-3)
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, f'velocity_error_x_log_{idx}.png'))
    plt.close()

    # Y-direction velocity error in log scale with SDF mask applied
    plt.figure()
    plt.imshow(error_y_masked, cmap='jet', origin='lower', vmin=-8, vmax=-3)
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, f'velocity_error_y_log_{idx}.png'))
    plt.close()

    # X-direction residual in log scale with SDF mask applied
    plt.figure()
    plt.imshow(residual_x_masked, cmap='jet', origin='lower', vmin=-8, vmax=-3)
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, f'residual_x_log_{idx}.png'))
    plt.close()

    # Y-direction residual in log scale with SDF mask applied
    plt.figure()
    plt.imshow(residual_y_masked, cmap='jet', origin='lower', vmin=-8, vmax=-3)
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, f'residual_y_log_{idx}.png'))
    plt.close()


def main(model_name, config_path, checkpoint_path):
    config = OmegaConf.load(config_path)

    plot_dir = config.model.plot_path
    os.makedirs(plot_dir, exist_ok=True)

    model = load_model(model_name, checkpoint_path, config)
    model = model.cuda()

    test_dataset = LidDrivenDataset(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        data_type=config.data.type,
        equation=config.data.equation, 
        inputs=config.data.inputs
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=6
    )

    residual_loss_fn = ResidualLoss(domain_size=512, device=torch.device("cuda")).to("cuda")

    model.eval()
    sample_to_plot = [77, 119]
    total_deriv_errors = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            x_batch, y_true = batch
            x_batch = x_batch.to("cuda")
            y_true = y_true.to("cuda")

            y_pred = model(x_batch)
            # Extract SDF from the input batch
            sdf = x_batch[:, 1:2, :, :]
        
            # Compute errors and residuals using the residual function
            error_x, error_y, residual_x, residual_y, deriv_error = residual_loss_fn(x_batch, y_pred, y_true)

            # Collect the total derivative error
            total_deriv_errors.append(deriv_error.item())
            
            # Only plot selected samples
            if idx in sample_to_plot:
                plot_velocity_error_and_residuals(idx, error_x.cpu().numpy(), error_y.cpu().numpy(),
                                                  residual_x.cpu().numpy(), residual_y.cpu().numpy(), 
                                                  plot_dir, sdf)

    # Save the derivative errors to a text file
    deriv_error_file = os.path.join(plot_dir, 'deriv_errors.txt')
    with open(deriv_error_file, 'w') as f:
        f.write(f"Average Derivative Error: {np.mean(total_deriv_errors):.6f}\n")
        f.write(f"Total Samples: {len(total_deriv_errors)}\n")
 
    print(f"Plots saved in {plot_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process different models for prediction and plotting.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to load (fno, cno, uno, wno, deeponet, pod-deeponet, geometric-deeponet).')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')

    args = parser.parse_args()
    main(args.model, args.config, args.checkpoint)
