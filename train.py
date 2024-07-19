import copy
import datetime
import os

import pandas as pd  # type: ignore
import torch  # type: ignore
import torch.backends.cudnn as cudnn  # type: ignore
import torch.optim as optim  # type: ignore
import xarray as xr  # type: ignore
from loguru import logger  # type: ignore
from sklearn.metrics import root_mean_squared_error  # type: ignore
from torch import nn  # type: ignore
from torch.utils.data.dataloader import DataLoader  # type: ignore
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm  # type: ignore

import models
from datasets import TrainDataset3D, create_3d_datasets
from utils import (AverageMeter, count_model_parameters, estimate_model_size,
                   load_config, plot_loss_metrics, save_config,
                   seed_everything)



def main():
    """Main function for training and evaluating the model.
    """

    config = load_config(file_path="config.yaml")
    # load conf file for training
    PREDICTION_DIR = config['prediction_dir']
    SEED = config['seed']
    LR = float(config['lr'])
    BATCH_SIZE = config['batch_size']
    NUM_EPOCHS = config['epochs']
    LOG_DIR = config['log_dir']
    TBP = config['tbp']
    GPU_DEVICE = config['gpu_device']
    LOSS_FUNC = config['loss']
    INPUTS_PATH = config['inputs_path']
    DEPTH = config['depth']
    TARGET_PATH = config['target_path']
    DATA_SPLIT = config['data_split']
    TRAIN_START = config['train_start']
    TRAIN_END = config['train_end']
    MODEL_ARCHITECTURE = config['model_architecture']


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
    train_dir = os.path.join(prediction_dir, "train_inference")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
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

    model = getattr(models, MODEL_ARCHITECTURE)()

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

    ds_inputs = xr.open_dataset(INPUTS_PATH)
    ds_target = xr.open_dataset(TARGET_PATH)
    # slice data related to the challange
    ds_inputs = ds_inputs.sel(time=slice(TRAIN_START, TRAIN_END))
    ds_target = ds_target.sel(time=slice(TRAIN_START, TRAIN_END))

    total_samples = len(ds_inputs['ssh'])
    # Calculate the number of samples for the 80/20 split
    train_samples = int(DATA_SPLIT[0]/100 * total_samples)
    # Perform the split
    ds_input_train = ds_inputs['ssh'][:train_samples]
    ds_input_valid = ds_inputs['ssh'][train_samples:]

    ds_target_train = ds_target['ssh'][:train_samples]
    ds_target_valid = ds_target['ssh'][train_samples:]

    # Create 3D training and validation datasets
    train_inputs_3d, train_target_3d = create_3d_datasets(ds_input_train, ds_target_train, depth=6)
    valid_inputs_3d, valid_target_3d = create_3d_datasets(ds_input_valid, ds_target_valid, depth=6)

    logger.info("Number of Training data {0:d}".format(len(ds_target_train)))
    logger.info("------")
    logger.info("Number of Validation data {0:d}".format(len(ds_target_valid)))
    logger.info("------")
    # train_dataset = TrainDataset(ds_inputs=ds_input_train, ds_target=ds_target_train)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # eval_dataset = EvalDataset(ds_inputs=ds_input_valid,ds_target=ds_target_valid)
    # eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Instantiate the datasets
    train_dataset = TrainDataset3D(train_inputs_3d, train_target_3d)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    valid_dataset = TrainDataset3D(valid_inputs_3d, valid_target_3d)
    eval_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

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
                inputs = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.float)
                preds = model(inputs)

                preds = preds[:, :, 3, :, :]

                loss_train = torch.sqrt(criterion(preds.to(torch.float32), targets.to(torch.float32)))
                loss_train.backward()
                optimizer.step()
                train_losses.update(loss_train.item(), len(inputs))
                # from train_logs import train_log
                # train_log(step=step, loss=loss_train, tensorboard_writer=train_tensorboard_writer, name="Training")
                t.set_postfix(loss='{:.6f}'.format(train_losses.avg))
                t.update(len(inputs))
                step += 1

        model.eval()
        targets = []
        preds = []

        for index, data in enumerate(eval_dataloader):
            inputs, target = data
            inputs = inputs.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            # inputs, masks, target = data
            # inputs = inputs.to(device, dtype=torch.float)
            # target = target.to(device, dtype=torch.float)
            # masks = masks.to(device, dtype=torch.float)


            with torch.no_grad():

                # pred = model(inputs, masks)
                pred = model(inputs)
                pred = pred[:, :, 3, :, :]
                eval_loss = torch.sqrt(criterion(pred.to(torch.float32), target.to(torch.float32)))

            eval_losses.update(eval_loss.item(), len(inputs))
            target = torch.squeeze(target, 0)
            target = target.detach().cpu().numpy()[0,:,:]
            pred = torch.squeeze(pred, 0)
            pred = pred.detach().cpu().numpy()[0,:,:]

            inputs = torch.squeeze(inputs, 0)
            inputs = inputs.detach().cpu().numpy()[0,:,:]

            from train_logs import log_prediction_plot
            if index==1:

                log_prediction_plot(inputs, pred, target, epoch, train_dir)

            targets.append(target.flatten())
            # pred = pred[3,:,:]
            preds.append(pred.flatten())

        train_tensorboard_writer.add_scalar('Loss/Validation', eval_losses.avg, epoch)
        val_tensorboard_writer.add_scalar('Loss/Trainin', train_losses.avg, epoch)

        rmse = root_mean_squared_error(targets, preds)
        metrics_dict[epoch] = { "rmse": rmse,
                                'loss_train': train_losses.avg,
                                'loss_eval': eval_losses.avg}
        plot_loss_metrics(metrics=metrics_dict, save_path=prediction_dir)

        df_metrics = pd.DataFrame(metrics_dict).T
        df_mean_metrics = df_metrics.mean()
        df_mean_metrics = pd.DataFrame(df_mean_metrics).T

        if epoch == 0:

            df_val_metrics = pd.DataFrame(columns=df_mean_metrics.columns)
            df_val_metrics = pd.concat([df_val_metrics, df_mean_metrics])

        else:
            df_val_metrics = pd.concat([df_val_metrics, df_mean_metrics])
            df_val_metrics = df_val_metrics.reset_index(drop=True)

        # dashboard = Dashboard(df_val_metrics)
        # dashboard.generate_dashboard()
        # dashboard.save_dashboard(directory_path=prediction_dir)

        logger.info(f'Epoch {epoch} Eval {LOSS_FUNC} - Loss: {eval_losses.avg} - rmse {rmse}')

        # Save best model
        if epoch == 0:

            best_epoch = epoch
            best_rmse = rmse
            best_loss = eval_losses.avg
            best_weights = copy.deepcopy(model.state_dict())

        if rmse < best_rmse:

            best_epoch = epoch
            best_rmse = rmse
            best_loss = eval_losses.avg
            best_weights = copy.deepcopy(model.state_dict())

    logger.info(f'best epoch: {best_epoch}, best RMSE: {best_rmse} loss: {best_loss}')
    model_path = os.path.join(prediction_dir, 'best.pth')
    torch.save(best_weights, model_path)
    logger.info('Training Done')
    logger.info('best epoch: {}, {} loss: {:.2f}'.format( best_epoch, LOSS_FUNC, best_loss))
    end_training_date = datetime.datetime.now()
    training_duration = end_training_date - start_training_date
    logger.info('Training Duration: {}'.format(str(training_duration)))
    df_val_metrics['Training_duration'] = training_duration
    df_val_metrics['nb_parameters'] = nb_parameters
    model_size = estimate_model_size(model)
    logger.info("model size: {}".format(model_size))
    df_val_metrics['model_size'] = model_size
    df_val_metrics.to_csv(os.path.join(prediction_dir, 'valid_metrics_log.csv'))

    save_config(prediction_dir=prediction_dir, config=config)
    from test import test
    test(config_file=config, checkpoint_path=model_path, prediction_dir=prediction_dir)

if __name__ == '__main__':

    main()
