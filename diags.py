import os 
import xarray as xr 
import numpy as np 
import datetime
import matplotlib.pylab as plt 
import imageio
from loguru import logger
import xrft
import pandas as pd
from matplotlib.ticker import ScalarFormatter

class Diag_OSSE():

    def __init__(self, ds_input, ds_pred, ds_target, dir_output):

        self.ds_input = ds_input
        self.ds_pred = ds_pred
        self.ds_target = ds_target
        self.dir_output = dir_output

    def animate(self, fps=2):

        image_paths = []
        # Temporary Directory to save PNGs
        tmp_dir = '{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(tmp_dir, exist_ok=True)

        vmin = np.nanmin(self.ds_target)
        vmax = np.nanmax(self.ds_target)

        for t in range(self.ds_input.shape[0]):

            fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,5))
            fig.suptitle(self.ds_input.time[t].values)
            self.ds_input[t].plot(ax=ax1, vmin=vmin, vmax=vmax)
            ax1.set_title('Input')
            self.ds_pred[t].plot(ax=ax2, vmin=vmin, vmax=vmax)
            ax2.set_title('Prediction')
            self.ds_target[t].plot(ax=ax3, vmin=vmin, vmax=vmax)
            ax3.set_title('Target')
            plt.savefig(f'{tmp_dir}/frame_{t}.png',bbox_inches='tight')
            image_paths.append(f'{tmp_dir}/frame_{t}.png')

        with imageio.get_writer(f'{self.dir_output}/anim.gif', mode='I', duration=1/fps) as writer:
            for image_path in image_paths:
                image = imageio.imread(image_path)
                writer.append_data(image)
        
            # Clean up the temporary images
            for image_path in image_paths:
                os.remove(image_path)
            os.rmdir(tmp_dir)

        return
    
    def compute_metrics(self, plot=True):

        # RMSE 
        self._rmse_score(plot=plot)

        # SPECTRAL
        self._psd_score(plot=plot)
    
    def Leaderboard(self):

        data = [['Hiyeyo', 
                np.round(self.leaderboard_rmse,2), 
                np.round(self.reconstruction_error_stability_metric, 2), 
                np.round(self.shortest_spatial_wavelength_resolved,2), 
                np.round(self.shortest_temporal_wavelength_resolved,2)]]
        
        Leaderboard = pd.DataFrame(data, 
                                columns=['Method', 
                                         "µ(RMSE) ", 
                                         "σ(RMSE)", 
                                        'λx (degree)', 
                                        'λt (days)'])
        
        with open(f'{self.dir_output}/metrics.txt', 'w') as f:
            dfAsString = Leaderboard.to_string()
            f.write(dfAsString)
        
    def _rmse_score(self, plot=True):

        logger.info('Compute RMSE-based scores...')

        # RMSE(t) 
        rmse_t = 1 - (((self.ds_pred - self.ds_target)**2).mean(dim=('lon', 'lat')))**0.5/(((self.ds_target)**2).mean(dim=('lon', 'lat')))**0.5
        
        # RMSE(x, y) 
        rmse_xy = 1 - ((self.ds_pred - self.ds_target)**2).mean(axis=0)**0.5 /(((self.ds_target)**2).mean(axis=0))**0.5

        rmse_t = rmse_t.rename('rmse_t')
        rmse_xy = rmse_xy.rename('rmse_xy')

        # Plot
        if plot:
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
            rmse_t.plot(ax=ax1)
            ax1.set_ylabel('RMSE Score [%]')
            im2 = rmse_xy.plot(ax=ax2,cmap='Reds_r')
            cbar = plt.colorbar(im2)
            cbar.ax.set_ylabel('RMSE Score [%]')
            fig.savefig(f'{self.dir_output}/rmse.png',dpi=100)


        # Temporal stability of the error
        self.reconstruction_error_stability_metric = rmse_t.std().values

        # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
        self.leaderboard_rmse = (1.0 - (((self.ds_pred - self.ds_target) ** 2).mean()) ** 0.5 / (
            ((self.ds_target) ** 2).mean()) ** 0.5).values

        logger.info(f'RMSE score = {np.round(self.leaderboard_rmse, 2)} +/- {np.round(self.reconstruction_error_stability_metric, 2)}',)



    def _psd_score(self, plot=True):
        
        logger.info('Compute PSD-based scores...')
    
        
        # Compute error = SSH_reconstruction - SSH_true
        err = (self.ds_pred - self.ds_target)
        err = err.chunk({"lat":1, 'time': err['time'].size, 'lon': err['lon'].size})
        # make time vector in days units 
        err['time'] = (err.time - err.time[0]) / np.timedelta64(1, 'D')
        
        # Rechunk SSH_true
        signal = self.ds_target.chunk({"lat":1, 'time': self.ds_target['time'].size, 'lon': self.ds_target['lon'].size})
        # make time vector in days units
        signal['time'] = (signal.time - signal.time[0]) / np.timedelta64(1, 'D')
    
        # Compute PSD_err and PSD_signal
        psd_err = xrft.power_spectrum(err, dim=['time', 'lon'], detrend='constant', window=True).compute()
        psd_signal = xrft.power_spectrum(signal, dim=['time', 'lon'], detrend='constant', window=True).compute()
        
        # Averaged over latitude
        mean_psd_signal = psd_signal.mean(dim='lat').where((psd_signal.freq_lon > 0.) & (psd_signal.freq_time > 0), drop=True)
        mean_psd_err = psd_err.mean(dim='lat').where((psd_err.freq_lon > 0.) & (psd_err.freq_time > 0), drop=True)
        
        # return PSD-based score
        psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)

        # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score
        try:
            level = [0.5]
            cs = plt.contour(1./psd_based_score.freq_lon.values,1./psd_based_score.freq_time.values, psd_based_score, level)
            x05, y05 = cs.collections[0].get_paths()[0].vertices.T
            plt.close()
            self.shortest_spatial_wavelength_resolved = np.min(x05)
            self.shortest_temporal_wavelength_resolved = np.min(y05)
        except:
            self.shortest_spatial_wavelength_resolved = np.nan
            self.shortest_temporal_wavelength_resolved = np.nan

        logger.info(f'Effective spatial resolution = {np.round(self.shortest_spatial_wavelength_resolved, 2)}°')
        logger.info(f'Effective temporal resolution = {np.round(self.shortest_temporal_wavelength_resolved, 2)}days')

        # Plot
        if plot:
            fig, ax =  plt.subplots(1, 1, figsize=(10, 5))
            data = (psd_based_score.values)
            ax.invert_yaxis()
            ax.invert_xaxis()
            c1 = ax.contourf(1./(psd_based_score.freq_lon), 1./psd_based_score.freq_time, data,
                            levels=np.arange(0,1.1, 0.1), cmap='RdYlGn', extend='both')
            ax.set_xlabel('spatial wavelength (degree_lon)', fontsize=15)
            ax.set_ylabel('temporal wavelength (days)', fontsize=15)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(linestyle='--', lw=1, color='w')
            ax.tick_params(axis='both', labelsize=18)

            c2 = ax.contour(1./(psd_based_score.freq_lon), 1./psd_based_score.freq_time, data, levels=[0.5], linewidths=2, colors='k')
            
            cbar = fig.colorbar(c1, ax=ax, pad=0.01)
            cbar.add_lines(c2)

            bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="k", lw=2)
            ax.annotate('Resolved scales',
                            xy=(1.12, 0.8),
                            xycoords='axes fraction',
                            xytext=(1.12, 0.55),
                            bbox=bbox_props,
                            arrowprops=
                                dict(facecolor='black', shrink=0.05),
                                horizontalalignment='left',
                                verticalalignment='center')

            ax.annotate('UN-resolved scales',
                            xy=(1.12, 0.2),
                            xycoords='axes fraction',
                            xytext=(1.12, 0.45),
                            bbox=bbox_props,
                            arrowprops=
                            dict(facecolor='black', shrink=0.05),
                                horizontalalignment='left',
                                verticalalignment='center')
            
            fig.savefig(f'{self.dir_output}/psd.png',dpi=100)

    
        return
    
    