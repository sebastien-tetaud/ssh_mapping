import numpy as np
import matplotlib.pyplot as plt

def log_prediction_plot(inputs, pred, target, epoch, prediction_dir):

    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,5))
    fig.suptitle(f'Epoch {epoch}')
    inputs[inputs==0.001] = np.nan
    im = ax1.pcolormesh(inputs, vmin=0,vmax=1)
    plt.colorbar(im)
    ax1.set_title('Input')
    im = ax2.pcolormesh(pred, vmin=0,vmax=1)
    plt.colorbar(im)
    ax2.set_title('Prediction')
    im = ax3.pcolormesh(target, vmin=0,vmax=1)
    plt.colorbar(im)
    ax3.set_title('Target')
    plt.savefig(f'{prediction_dir}/input_epoch{epoch}.png')




