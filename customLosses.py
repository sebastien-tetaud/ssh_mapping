
import torch
import torch.nn as nn
import matplotlib.pylab as plt 


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,predictions, targets):
        return torch.sqrt(self.mse(predictions, targets))
    
class QGLoss(nn.Module):

    def __init__(self, lon, lat, dt, SSH=None, c=2.7, device='cpu', tint=24*3600, alpha=.5, Nh=1):
        super(QGLoss, self).__init__()

        from qgm import Qgm
        self.qgm = Qgm(lon=lon, lat=lat, dt=dt, SSH=SSH, c=c, device=device)
        self.tint = tint
        self.alpha = alpha
        self.Nh = Nh

    def forward(self, predictions, targets):
        loss_qg = 0 
        for h in range(1,self.Nh+1):
            predictions_forward = self.qgm.forward(predictions.squeeze(1), targets.squeeze(1), tint=h*self.tint)
            loss_qg += torch.sqrt(torch.mean((predictions_forward[:-h] - predictions.squeeze(1)[h:]) ** 2))

        loss_mse = torch.sqrt(torch.mean((predictions - targets) ** 2))

        return self.alpha * loss_mse + (1-self.alpha) * loss_qg / self.Nh 
    
