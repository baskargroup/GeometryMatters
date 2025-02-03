import os
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models.scot.scOT import scOT
from data.dataset import LidDrivenDataset
from residual_m3_function import ResidualLoss

torch.cuda.empty_cache()

def load_model(model_name, checkpoint_path, config):
    base_model_name = model_name.split('-')[0].lower()
    params = {k: v for k, v in config.model.items()}
    pretrained_path = params.pop('pretrained_path', None)

    if base_model_name in ['scot', 'poseidon']:
        model = scOT.load_from_checkpoint(
            checkpoint_path,
            **params,
            pretrained_path=pretrained_path
        )
    else:
        raise ValueError(f"Unknown base model name: {base_model_name}")
    
    return model

domain_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
residual_loss_fn = ResidualLoss(domain_size=domain_size, device=device).to(device)

# Define the custom loss function with both relative L2 errors and MSE
def custom_loss(yhat, y, x_full):
    mses = torch.zeros(4).to(yhat.device)  # Now 4 values: 2 for L2, 2 for MSE

    # SDF is always the second channel in the original full dataset
    sdf = x_full[:, 1, :, :]  # Extract SDF from the second channel of x_full

    # Define conditions based on SDF values
    condition = (sdf >= 0)
    condition_near = (sdf >= 0) & (sdf <= 0.2)

    condition = condition.unsqueeze(1).expand_as(yhat)
    condition_near = condition_near.unsqueeze(1).expand_as(yhat)

    masked_elements_count = condition.sum().item()
    masked_elements_near_count = condition_near.sum().item()

    if masked_elements_count == 0 or masked_elements_near_count == 0:
        raise ValueError("Count of elements outside or near the object are zero.")

    y_outside_obj = torch.where(condition.to(y.device), y, torch.tensor(0.0, device=y.device))
    y_near_obj = torch.where(condition_near.to(y.device), y, torch.tensor(0.0, device=y.device))

    yhat_outside_obj = torch.where(condition.to(yhat.device), yhat, torch.tensor(0.0, device=yhat.device))
    yhat_near_obj = torch.where(condition_near.to(yhat.device), yhat, torch.tensor(0.0, device=yhat.device))

    # Calculate relative L2 errors (already in the original code)
    mses[0] = torch.norm((yhat_outside_obj - y_outside_obj), p=2) / torch.norm(y_outside_obj, p=2)
    mses[1] = torch.norm((yhat_near_obj - y_near_obj), p=2) / torch.norm(y_near_obj, p=2)

    # Add MSE loss calculations
    mses[2] = torch.mean((yhat_outside_obj - y_outside_obj) ** 2)  # MSE for outside object
    mses[3] = torch.mean((yhat_near_obj - y_near_obj) ** 2)  # MSE for near object

    return mses

def save_individual_plot(label, field, idx, plot_dir, vmin=None, vmax=None):
    plt.figure()
    plt.imshow(field, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, f'{label}_{idx}.png'))
    plt.close()

def plot_ldc_like(y, y_hat, x, idx, plot_path, individual_plot_dir, r1_field, r2_field, div_field):
    condition_full = (x[:, 1, :, :] > 0).squeeze().cpu().numpy()

    u_true_full = np.ma.masked_where(~condition_full, y[0, 0, :, :].cpu().numpy())
    v_true_full = np.ma.masked_where(~condition_full, y[0, 1, :, :].cpu().numpy())
    p_true_full = np.ma.masked_where(~condition_full, y[0, 2, :, :].cpu().numpy())
    u_pred_full = np.ma.masked_where(~condition_full, y_hat[0, 0, :, :].cpu().numpy())
    v_pred_full = np.ma.masked_where(~condition_full, y_hat[0, 1, :, :].cpu().numpy())
    p_pred_full = np.ma.masked_where(~condition_full, y_hat[0, 2, :, :].cpu().numpy())

    u_error = np.ma.masked_where(~condition_full, np.abs(u_true_full - u_pred_full))
    v_error = np.ma.masked_where(~condition_full, np.abs(v_true_full - v_pred_full))
    p_error = np.ma.masked_where(~condition_full, np.abs(p_true_full - p_pred_full))

    epsilon = 1e-10
    u_log_error = np.log10(u_error + epsilon)
    v_log_error = np.log10(v_error + epsilon)
    p_log_error = np.log10(p_error + epsilon)

    save_individual_plot('u_truth', u_true_full, idx, individual_plot_dir)
    save_individual_plot('u_pred', u_pred_full, idx, individual_plot_dir)
    save_individual_plot('u_error_log', u_log_error, idx, individual_plot_dir)
    save_individual_plot('v_truth', v_true_full, idx, individual_plot_dir)
    save_individual_plot('v_pred', v_pred_full, idx, individual_plot_dir)
    save_individual_plot('v_error_log', v_log_error, idx, individual_plot_dir)
    save_individual_plot('p_truth', p_true_full, idx, individual_plot_dir)
    save_individual_plot('p_pred', p_pred_full, idx, individual_plot_dir)
    save_individual_plot('p_error_log', p_log_error, idx, individual_plot_dir)

    condition_residuals = condition_full[:-1, :-1]
    r1_log_plot = np.ma.masked_where(~condition_residuals, np.log10(np.abs(r1_field[0].cpu().numpy()) + epsilon))
    r2_log_plot = np.ma.masked_where(~condition_residuals, np.log10(np.abs(r2_field[0].cpu().numpy()) + epsilon))
    r3_log_plot = np.ma.masked_where(~condition_residuals, np.log10(np.abs(div_field[0].cpu().numpy()) + epsilon))
    save_individual_plot('r_x_log', r1_log_plot, idx, individual_plot_dir, vmin=-7, vmax=0)
    save_individual_plot('r_y_log', r2_log_plot, idx, individual_plot_dir, vmin=-7, vmax=0)
    save_individual_plot('div_log', r3_log_plot, idx, individual_plot_dir, vmin=-7, vmax=0)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    labels = ['u', 'v', 'p']
    data_true = [u_true_full, v_true_full, p_true_full]
    data_pred = [u_pred_full, v_pred_full, p_pred_full]
    data_error = [u_log_error, v_log_error, p_log_error]

    for i, (true, pred, error, label) in enumerate(zip(data_true, data_pred, data_error, labels)):
        vmin = true.min()
        vmax = true.max()
        im_true = axs[0, i].imshow(true, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f'{label} Ground Truth')
        divider = make_axes_locatable(axs[0, i])
        cax_true = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_true, cax=cax_true)
        im_pred = axs[1, i].imshow(pred, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        axs[1, i].set_title(f'{label} Prediction')
        divider = make_axes_locatable(axs[1, i])
        cax_pred = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_pred, cax=cax_pred)
        im_error = axs[2, i].imshow(error, cmap='jet', origin='lower', vmin=-6, vmax=0)
        axs[2, i].set_title(f'{label} Error')
        divider = make_axes_locatable(axs[2, i])
        cax_error = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_error, cax=cax_error)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def main(model_name, config_path, checkpoint_path):
    config = OmegaConf.load(config_path)

    plot_dir = config.model.plot_path
    os.makedirs(plot_dir, exist_ok=True)
    individual_plot_dir = os.path.join(plot_dir, 'individual_plots')
    os.makedirs(individual_plot_dir, exist_ok=True)

    model = load_model(model_name, checkpoint_path, config)
    model = model.cuda()

    test_dataset_SDF = LidDrivenDataset(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        data_type=config.data.type,
        inputs='sdf'
    )


    test_dataset = LidDrivenDataset(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        data_type=config.data.type,
        inputs=config.data.inputs
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=6
    )

    trainer = Trainer(precision="16-mixed")
    model.eval()

    all_losses = []
    all_residual_mom_losses = []
    all_residual_cont_losses = []
    residual_data = []

    #sample_to_plot = [77, 83, 91, 107]
    sample_to_plot = [77, 119] 
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            x_batch, y = batch
            x_batch = x_batch.to(device)
            y = y.to(device)

            y_hat = model(x_batch)
            losses = custom_loss(y_hat, y[:, :3, :, :], test_dataset_SDF.x[idx:idx+1])
            r1_field, r2_field, div_field, r_Total, div_Total = residual_loss_fn(x_batch, y_hat)

            all_losses.append(losses.cpu().numpy())
            all_residual_mom_losses.append(r_Total.cpu().numpy())
            all_residual_cont_losses.append(div_Total.cpu().numpy())
            residual_data.append([idx, r_Total.item(), div_Total.item()])

            if idx in sample_to_plot:
                plot_path = os.path.join(plot_dir, f'prediction_{idx}.png')
                plot_ldc_like(y, y_hat, x_batch, idx, plot_path, individual_plot_dir, r1_field, r2_field, div_field)

    average_losses = np.mean(all_losses, axis=0)
    average_resid_mom = np.mean(np.abs(all_residual_mom_losses))
    average_resid_cont = np.mean(np.abs(all_residual_cont_losses))

    average_loss_file = os.path.join(plot_dir, 'losses_average.txt')
    with open(average_loss_file, 'w') as f:
        f.write('Average MSE (Outside Object): ' + f'{average_losses[2]:.6f}' + '\n')
        f.write('Average MSE (Near Object): ' + f'{average_losses[3]:.6f}' + '\n')
        f.write('Average L2 Loss (Outside Object): ' + f'{average_losses[0]:.6f}' + '\n')
        f.write('Average L2 Loss (Near Object): ' + f'{average_losses[1]:.6f}' + '\n')
        f.write('Average momentum residual (Outside Object): ' + f'{average_resid_mom:.6f}' + '\n')
        f.write('Average continuity residual (Outside Object): ' + f'{average_resid_cont:.6f}' + '\n')

    residual_df = pd.DataFrame(residual_data, columns=['Sample ID', 'Momentum Residual', 'Continuity Residual'])
    residual_df.to_csv(os.path.join(plot_dir, 'residuals_per_sample.csv'), index=False)

    plt.figure()
    plt.plot(residual_df['Sample ID'], residual_df['Momentum Residual'], label='Momentum Residual', color='b')
    plt.xlabel('Sample ID')
    plt.ylabel('Momentum Residual')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, 'momentum_residuals.png'))
    plt.close()

    plt.figure()
    plt.plot(residual_df['Sample ID'], residual_df['Continuity Residual'], label='Continuity Residual', color='r')
    plt.xlabel('Sample ID')
    plt.ylabel('Continuity Residual')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, 'continuity_residuals.png'))
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process scOT or Poseidon models for prediction and plotting.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to load (scot-T, scot-B, scot-L, poseidon-T, poseidon-B, poseidon-L).')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')

    args = parser.parse_args()
    main(args.model, args.config, args.checkpoint)
