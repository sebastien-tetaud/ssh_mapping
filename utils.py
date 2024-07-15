import os
import random
import json
import numpy as np
import seaborn as sns
import torch
from bokeh.layouts import gridplot
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import models
from loguru import logger
import yaml




def load_config(file_path: str) -> dict:
    """
    Load YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing configuration information.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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

    model = getattr(models, model_architecture)()

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


def save_config(prediction_dir, config):
    """
    Save the configuration dictionary to a JSON file in the specified directory.

    Parameters:
    - prediction_dir (str): The directory where the config.json file will be saved.
    - config (dict): The configuration dictionary to save.
    """

    # Define the path to the config file
    config_path = os.path.join(prediction_dir, "config.json")

    # Write the config dictionary to the JSON file
    with open(config_path, 'w') as fp:
        json.dump(config, fp, indent=4)  # indent=4 for pretty printing

    logger.info(f"Configuration saved to {config_path}")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(y_pred, y_true, save_path):
    """
    Plot confusion matrices using Seaborn's heatmap for two predicted sets.

    Args:

    - y_pred (array-like): Predicted labels for the first set.
    - y_true (array-like): True labels.
    """


    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))


def plot_loss_metrics(metrics, save_path):

    """
    Generate a plot displaying 'loss_train' and 'loss_eval' metrics against epochs.

    Parameters:
    metrics (dict): A dictionary containing metrics for different epochs.
                    Each key represents an epoch number, and the corresponding value is a dictionary
                    containing metrics such as 'loss_train' and 'loss_eval'.

    Returns:
    None: Displays a plot showing 'loss_train' and 'loss_eval' against the epochs.
    """
    epochs = list(metrics.keys())
    loss_train = [metrics[epoch]['loss_train'] for epoch in epochs]
    loss_eval = [metrics[epoch]['loss_eval'] for epoch in epochs]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_train, label='Loss Train')
    plt.plot(epochs, loss_eval, label='Loss Eval')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,"loss.png"))  # Save the plot as an image
    plt.show()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_model_parameters(model):
    """
    Count number of parameters in a Pytorch Model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits

        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits

    size_model = size_model / 8e6
    print(f"model size: {size_model:.2f} MB")
    return size_model


class Dashboard:

    """
    Generates and saves a dashboard based on a given dataframe.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the data for the dashboard.

    Attributes:
        dataframe (pandas.DataFrame): The dataframe containing the data for the dashboard.

    Methods:
        generate_dashboard: Generates the dashboard plots.
        save_dashboard: Saves the generated dashboard to a specified directory path.
    """
    def __init__(self, df):

        """
        Initializes the Dashboard instance.

        Args:
            dataframe (pandas.DataFrame): The dataframe containing the data for the dashboard.
        """
        self.df = df

    def generate_dashboard(self):

        """
        Generates individual plots for each metric in the dataframe and combines them into a grid layout.

        Returns:
            bokeh.layouts.gridplot: The grid layout of plots representing the dashboard.

        Raises:
            IndexError: If there are not enough metrics available for plotting.
        """

        metrics = list(self.df.columns)
        plots = []
        colors = Category10[10]  # Change the number based on the number of metrics

        # Generate individual plots with a given color palette
        for i, metric in enumerate(metrics):
            p = figure(title=metric, x_axis_label='Epoch', y_axis_label=metric, width=800, height=300)
            p.line(x=self.df.index, y=self.df[metric], legend_label=metric, color=colors[i],line_width=4)
            plots.append(p)

        # Create grid layout
        self.fig = gridplot(plots, ncols=2)

        return self.fig


    def save_dashboard(self, directory_path):

            """
            Saves the generated dashboard to the specified directory path.

            Args:
                fig (bokeh.layouts.gridplot): The grid layout of plots representing the dashboard.
                directory_path (str): The path to the directory where the dashboard should be saved.
            """

            filename = os.path.join(directory_path,'validation_metrics_log.html')
            output_file(filename=filename, title='validation metrics log')
            save(self.fig, filename)