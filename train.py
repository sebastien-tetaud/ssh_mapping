import copy
import datetime
import os
import yaml
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from tqdm import tqdm
from datasets import TrainDataset, EvalDataset
from utils import *
from sklearn.metrics import accuracy_score, f1_score

from loguru import logger
import random
import numpy as np

import warnings
warnings.simplefilter('ignore')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torchvision.models as models


class ModifiedMobileNetV2(nn.Module):
    def __init__(self):
        super(ModifiedMobileNetV2, self).__init__()
        # Load the pre-trained MobileNetV2 model
        self.mobilenet_v2 = models.mobilenet_v2(weights="IMAGENET1K_V1")

        # Remove the classifier
        self.features = self.mobilenet_v2.features

        # Add a final convolutional layer to output a single channel with the same spatial dimensions
        self.final_conv = nn.Conv2d(in_channels=1280, out_channels=1, kernel_size=1)

        # Optional: add upsampling layer to ensure output matches input dimensions (if needed)
        self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.features(x)  # Get feature maps
        x = self.final_conv(x)  # Apply the final conv layer
        x = self.upsample(x)
        return x





def main(config):
    """Main function for training and evaluating the model.

    Args:
        config (dict): Dictionary of configurations.
    """

    # load conf file for training
    PREDICTION_DIR = config['prediction_dir']
    REGULARIZATION = config['regularization']
    MODEL_ARCHITECTURE = config['model_architecture']
    DATA_AUGMENTATION = config['data_augmentation']
    SEED = config['seed']
    LR = float(config['lr'])
    BATCH_SIZE = config['batch_size']
    NUM_EPOCHS = config['epochs']
    LOG_DIR = config['log_dir']
    TBP = config['tbp']
    GPU_DEVICE = config['gpu_device']
    LOSS_FUNC = config['loss']
    AUTO_EVAL = config['auto_eval']

    seed_everything(seed=SEED)
    start_training_date = datetime.datetime.now()
    logger.info("start training session '{}'".format(start_training_date))
    date = start_training_date.strftime('%Y_%m_%d_%H_%M_%S')

    TENSORBOARD_DIR = 'tensorboard'
    tensorboard_path = os.path.join(LOG_DIR, TENSORBOARD_DIR)
    logger.info("Tensorboard path: {}".format(tensorboard_path))
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)

    prediction_dir = os.path.join(PREDICTION_DIR, '{}'.format(date))
    os.makedirs(prediction_dir)
    log_filename = os.path.join(prediction_dir, "train.log")
    logger.add(log_filename, backtrace=False, diagnose=True)

    cudnn.benchmark = True
    if TBP is not None:

        logger.info("starting tensorboard")
        logger.info("------")

        command = f'tensorboard --logdir {tensorboard_path} --port {TBP} --host localhost --load_fast=true'

        train_tensorboard_writer = SummaryWriter(
            os.path.join(tensorboard_path, 'train'), flush_secs=30)
        val_tensorboard_writer = SummaryWriter(
            os.path.join(tensorboard_path, 'val'), flush_secs=30)
        writer = SummaryWriter()
    else:
        logger.exception("An error occurred: {}", "no tensorboard")
        tensorboard_process = None
        train_tensorboard_writer = None
        val_tensorboard_writer = None



    model = ModifiedMobileNetV2()

    logger.info("Number of GPU(s) {}: ".format(torch.cuda.device_count()))
    logger.info("GPU(s) in used {}: ".format(GPU_DEVICE))
    logger.info("------")
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device='cuda')
    nb_parameters = count_model_parameters(model=model)
    logger.info("Number of parameters {}: ".format(nb_parameters))

    # Define Optimizer
    if LOSS_FUNC == "MSE":
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    import xarray as xr
    ds_inputs = xr.open_dataset("data_inputs.nc")
    ds_target = xr.open_dataset("data_target.nc")


    ds_input_train = ds_inputs['ssh'][:300]
    ds_input_valid = ds_inputs['ssh'][300:]

    ds_target_train = ds_target['ssh'][:300]
    ds_target_valid = ds_target['ssh'][300:]

    logger.info("Number of Training data {0:d}".format(len(ds_target_train)))
    logger.info("------")
    logger.info("Number of Validation data {0:d}".format(len(ds_target_valid)))
    logger.info("------")
    train_dataset = TrainDataset(ds_inputs=ds_input_train,ds_target=ds_target_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    eval_dataset = EvalDataset(ds_inputs=ds_input_valid,ds_target=ds_target_valid)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = 0.0
    step = 0
    metrics_dict = {}

    for epoch in range(NUM_EPOCHS):

        train_losses = AverageMeter()
        eval_losses = AverageMeter()
        model.train()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % BATCH_SIZE), ncols = 100, colour='#3eedc4') as t:
            t.set_description('epoch: {}/{}'.format(epoch, NUM_EPOCHS - 1))

            for data in train_dataloader:

                optimizer.zero_grad()
                inputs, targets = data
                logger.info(inputs.shape)
                logger.info(targets.shape)
                inputs = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.float)
                preds = model(inputs)
                loss_train = criterion(preds.to(torch.float32), targets.to(torch.float32))

                loss_train.backward()
                optimizer.step()

                train_losses.update(loss_train.item(), len(inputs))
                # train_log(step=step, loss=loss, tensorboard_writer=train_tensorboard_writer, name="Training")
                t.set_postfix(loss='{:.6f}'.format(train_losses.avg))
                t.update(len(inputs))

                step += 1

        model.eval()

        targets = []
        preds = []
        # Model Evaluation
        for index, data in enumerate(eval_dataloader):

            inputs, target = data
            inputs = inputs.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            with torch.no_grad():

                pred = model(inputs)
                eval_loss = criterion(pred.to(torch.float32), target.to(torch.float32))

            eval_losses.update(eval_loss.item(), len(inputs))
    #         target = torch.squeeze(target,0)
    #         target = target.detach().cpu().numpy()
    #         pred = torch.squeeze(pred,0)
    #         pred = pred.detach().cpu().numpy()
    #         pred = pred.round()
    #         targets.append(target)
    #         preds.append(pred)

    #     acc = accuracy_score(targets, preds)
    #     f1 = f1_score(targets, preds)
    #     metrics_dict[epoch] = { "F1": f1, "Accuracy": acc,
    #                             'loss_train': train_losses.avg,
    #                             'loss_eval': eval_losses.avg}

    #     plot_loss_metrics(metrics=metrics_dict, save_path=prediction_dir)

    #     df_metrics = pd.DataFrame(metrics_dict).T
    #     df_mean_metrics = df_metrics.mean()
    #     df_mean_metrics = pd.DataFrame(df_mean_metrics).T

    #     if epoch == 0:

    #         df_val_metrics = pd.DataFrame(columns=df_mean_metrics.columns)
    #         df_val_metrics = pd.concat([df_val_metrics, df_mean_metrics])

    #     else:
    #         df_val_metrics = pd.concat([df_val_metrics, df_mean_metrics])
    #         df_val_metrics = df_val_metrics.reset_index(drop=True)

    #     dashboard = Dashboard(df_val_metrics)
    #     dashboard.generate_dashboard()
    #     dashboard.save_dashboard(directory_path=prediction_dir)
    #     # Access the mean values
    #     logger.info(f'Epoch {epoch} Eval {LOSS_FUNC} - Loss: {eval_losses.avg} - Acc {acc} - F1 {f1}')

    #     # Save best model
    #     if epoch == 0:

    #         best_epoch = epoch
    #         best_f1 = f1
    #         best_loss = eval_losses.avg
    #         best_weights = copy.deepcopy(model.state_dict())

    #     elif f1 > best_f1:

    #         best_epoch = epoch
    #         best_f1 = f1
    #         best_loss = eval_losses.avg
    #         best_weights = copy.deepcopy(model.state_dict())


    # logger.info(f'best epoch: {best_epoch}, best F1-score: {best_f1} loss: {best_loss}')

    # torch.save(best_weights, os.path.join(prediction_dir, 'best.pth'))
    # logger.info('Training Done')
    # logger.info('best epoch: {}, {} loss: {:.2f}'.format( best_epoch, LOSS_FUNC, best_loss))
    # # Measure total training time
    # end_training_date = datetime.datetime.now()
    # training_duration = end_training_date - start_training_date
    # logger.info('Training Duration: {}'.format(str(training_duration)))
    # df_val_metrics['Training_duration'] = training_duration
    # df_val_metrics['nb_parameters'] = nb_parameters
    # model_size = estimate_model_size(model)
    # logger.info("model size: {}".format(model_size))
    # df_val_metrics['model_size'] = model_size
    # df_val_metrics.to_csv(os.path.join(prediction_dir, 'valid_metrics_log.csv'))

    # if AUTO_EVAL:

    #     from eval import auto_eval
    #     model_path = os.path.join(prediction_dir, 'best.pth')
    #     preds_eval, targets_eval = auto_eval(model_path=model_path,
    #                                         model_architecture=MODEL_ARCHITECTURE,
    #                                         save_path=prediction_dir)
    #     plot_confusion_matrix(preds_eval,targets_eval,prediction_dir)


if __name__ == '__main__':

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.info(exc)

    main(config)
