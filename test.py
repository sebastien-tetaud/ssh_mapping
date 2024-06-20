import torch
import xarray as xr
import numpy as np
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from datasets import TestDataset
import os
from models import AutoencoderCNN
from train_logs import log_prediction_plot
from utils import count_model_parameters
import yaml
import diags

def load_model(checkpoint_path, model_architecture, device='cuda', model_eval=False):
    """
    Load a specified model from a checkpoint and prepare it for use.

    Parameters:
    - checkpoint_path (str): Path to the checkpoint file (.pth) containing the model's state dictionary.
    - model_name (str): The name of the model to load. Supported models include:
      - 'AutoencoderCNN': Loads an instance of the AutoencoderCNN model.
      Add more models here as needed.
    - device (str, optional): The device on which to load the model. Options are 'cuda' for GPU or 'cpu'.
      Default is 'cuda'. If 'cuda' is chosen but not available, it falls back to 'cpu'.
    - model_eval (bool, optional): If True, sets the model to evaluation mode (`model.eval()`).
      Default is False.

    Returns:
    - model (torch.nn.Module): The loaded model, moved to the specified device.
    - device (torch.device): The device on which the model is loaded.

    Raises:
    - ValueError: If an unsupported model_name is provided.

    Example usage:
    ```python
    model, device = load_model(
        checkpoint_path="/path/to/checkpoint.pth",
        model_name="AutoencoderCNN",
        device='cuda',
        model_eval=True
    )
    print(f"Model loaded on device: {device}")
    ```
    """
    if model_architecture == "AutoencoderCNN":
        # Initialize the model
        model = AutoencoderCNN()
    else:
        raise ValueError(f"Unsupported model_name: {model_architecture}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the model state
    model.load_state_dict(checkpoint)

    if model_eval:
        # Set the model to evaluation mode
        model.eval()
        logger.info("Model set to evaluation mode")

    # Move the model to the specified device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, device

def prediction(test_dataloader, device, model, ymin, ymax):

    list_pred = []

    for data in tqdm(test_dataloader, ncols=100, colour='#FF33EC'):
    
        inputs, _ = data
        inputs = inputs.to(device, dtype=torch.float)
        with torch.no_grad():
            pred = model(inputs)
        pred = torch.squeeze(pred,0)
        pred = pred.detach().cpu().numpy()[0,:,:]

        # Back to real values before normalization 
        # We should create a specific function for normalization and associated de-normalization
        pred = (pred - 0.01) * (ymax - ymin) + ymin
        
        # Append to list
        list_pred.append(pred)

    return np.asarray(list_pred)


def test(config_file, checkpoint_path, prediction_dir):

    model_architecture = config_file["model_architecture"]
    test_start = config_file["test_start"]
    test_end = config_file["test_end"]
    inputs_path = config_file["inputs_path"]
    target_path = config_file["target_path"]
    name_var_inputs = config_file["name_var_inputs"]
    name_var_target = config_file["name_var_target"]
    name_diag = config['name_diag']
    write_netcdf = config['write_netcdf']
    animate = config['animate']
    compute_metrics = config['compute_metrics']

    test_dir = os.path.join(prediction_dir,"test_inference")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    model, device = load_model(checkpoint_path, model_architecture, device='cuda', model_eval=True)
    nb_parameters = count_model_parameters(model=model)
    logger.info(f"Model is on Cuda: {next(model.parameters()).is_cuda}")
    logger.info("Number of parameters {}: ".format(nb_parameters))

    ds_inputs = xr.open_dataset(inputs_path)
    ds_target = xr.open_dataset(target_path)

    ds_inputs = ds_inputs.sel(time=slice(test_start, test_end))[name_var_inputs]
    ds_target = ds_target.sel(time=slice(test_start, test_end))[name_var_target]

    test_dataset = TestDataset(ds_inputs=ds_inputs,ds_target=ds_target)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Run prediction and stack in an array
    pred = prediction(test_dataloader, device, model, ds_target.min().values, ds_target.max().values)

    # Create dataset
    ds_pred = ds_target.copy()
    ds_pred.data = pred

    if write_netcdf:
        ds_pred.to_netcdf(f'{test_dir}/pred.nc')

    # Run diagnostics
    diag = getattr(diags, name_diag)(ds_inputs, ds_pred, ds_target, test_dir)
    if animate:
        logger.info('Diagnostics: animate')
        diag.animate()
    if compute_metrics:
        logger.info('Diagnostics: compute_metrics')
        diag.compute_metrics()
        diag.Leaderboard()


if __name__ == '__main__':

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.info(exc)

    test(config, 'training_inference/2024_06_20_08_55_57/best.pth', 'training_inference/2024_06_20_08_55_57')

