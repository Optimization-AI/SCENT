import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')
from collections import Counter
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from myutils import *
from sklearn.decomposition import PCA
import torch

class SENT(nn.Module):

    def __init__(self, alpha_t=2.0):
        super(SENT, self).__init__()
        self.b = torch.tensor(0.0)
        self.w = torch.tensor(0.0)
        self.bu = torch.tensor(0.0)
        self.alpha_t = torch.tensor(-alpha_t)
        self.criterion = MSELoss(reduction='none')
        self.gamma_t = torch.tensor(1.0)

    def forward(self, output, target, cls_weights, myLambda):

        loss = self.criterion(output, target, cls_weights)
        loss_max = loss.detach().max()
        expLoss1 = torch.exp((loss-loss_max) / myLambda)
        expLoss1 = torch.mean(expLoss1)

        if self.bu == 0.0:
            self.bu = torch.log(expLoss1.detach()) +  loss_max/myLambda  

        log_alpha = self.alpha_t         
        s = log_alpha + self.bu                    
        log_denom = torch.nn.functional.softplus(s)

        b_new = self.bu - log_denom                
        w_new = (s - log_denom) + (torch.log(expLoss1) + loss_max/myLambda)
        bu_new = torch.logaddexp(b_new, w_new)    

        # Store DRO loss for logging: λ * log(E[exp(loss/λ)])
        self.dro_loss = myLambda * torch.log(expLoss1)

        self.b = b_new.detach()
        self.w = w_new.detach()
        self.bu = bu_new.detach()

        log_alpha = self.alpha_t 
        x = log_alpha + self.bu
        self.gamma_t = torch.sigmoid(x)

        t = torch.max(loss_max/myLambda, self.bu)
        log_numerator = loss / myLambda - t
        log_denominator = self.bu - t

        v = log_numerator - log_denominator
        new_loss = torch.exp(v)

        final_loss = torch.mean(new_loss.detach() * loss)

        return final_loss

class SOX(nn.Module):

    def __init__(self, gamma=0.95):
        super(SOX, self).__init__()
        self.b = torch.tensor(0.0)
        self.w = torch.tensor(0.0)
        self.bu = torch.tensor(0.0)
        self.gamma = torch.tensor(gamma)
        self.criterion = MSELoss(reduction='none')
        self.alpha_t = torch.tensor(0.0)

    def forward(self, output, target, cls_weights, myLambda):

        loss = self.criterion(output, target, cls_weights)

        loss_max = loss.detach().max()
        expLoss1 = torch.exp((loss-loss_max) / myLambda)
        expLoss1 = torch.mean(expLoss1)

        if self.bu == 0.0:
            self.bu = torch.log(expLoss1.detach()) +  loss_max/myLambda  

        b_new = torch.log(1-self.gamma) + self.bu
        w_new = torch.log(self.gamma) + torch.log(expLoss1) +  loss_max/myLambda  
        bu_new = torch.max(b_new, w_new) - torch.log(torch.sigmoid(torch.abs(b_new-w_new)))   

        self.dro_loss = myLambda * torch.log(expLoss1)

        self.b = b_new.detach()
        self.w = w_new.detach()
        self.bu = bu_new.detach()

        t = torch.max(loss_max/myLambda, self.bu)
        new_loss = torch.exp(loss / myLambda - t) / torch.exp(self.bu - t)

        final_loss = torch.mean(new_loss.detach() * loss)

        return final_loss

class U_MAX(nn.Module):

    def __init__(self, 
                delta=0.0,
                lr_dual=1e-1):

        super(U_MAX, self).__init__()
        self.b = torch.tensor(0.0)
        self.w = torch.tensor(0.0)
        self.bu = torch.tensor(0.0)
        self.criterion = MSELoss(reduction='none')
        self.alpha_t = torch.tensor(0.0)
        self.lr_dual = lr_dual
        self.delta = delta

    def forward(self, output, target, cls_weights, myLambda):
    
        lr_dual = self.lr_dual

        loss = self.criterion(output, target, cls_weights)
        loss_max = loss.detach().max()
        expLoss1 = torch.exp((loss-loss_max) / myLambda)
        expLoss1 = torch.mean(expLoss1)

        if self.bu == 0.0:
            self.bu = torch.log(expLoss1.detach()) +  loss_max/myLambda  
            
        log_exp_sum = torch.log(expLoss1.detach())
        threshold = log_exp_sum + loss_max/myLambda - self.delta
        if self.bu <= threshold:
            self.bu = log_exp_sum + loss_max/myLambda

        log_numerator = torch.log(expLoss1) + loss_max/myLambda  
        log_denominator = self.bu
        v = log_numerator - log_denominator

        new_loss = torch.exp(v)

        self.bu += myLambda * lr_dual * (new_loss.detach() - 1)

        t = torch.max(loss_max / myLambda, self.bu)
        log_numerator = loss / myLambda - t
        log_denominator = self.bu - t
        v = log_numerator - log_denominator
        new_loss = torch.exp(v)

        final_loss = torch.mean(new_loss.detach() * loss)

        return final_loss

    
class BSGD(nn.Module):

    def __init__(self):
        super(BSGD, self).__init__()
        self.criterion = MSELoss(reduction='none')

    def forward(self, output, target, cls_weights, myLambda):

        loss = self.criterion(output, target, cls_weights)
        loss = myLambda * (torch.logsumexp(loss/myLambda, dim=0) - math.log(loss.numel()))
        return loss

def get_train_loss(loss_type, gamma=None, alpha_t=None, delta=None):
    # Handle non-string types (e.g., float for softplus_approx)
    if not isinstance(loss_type, str):
        return None
    
    if loss_type == 'ce':
        criterion = CBCELoss()
    elif 'SENT' in loss_type:
        criterion = SENT(alpha_t=alpha_t if alpha_t is not None else 2.0)
    elif 'SOX' in loss_type:
        criterion = SOX(gamma=gamma if gamma is not None else 0.95)
    elif 'BSGD' in loss_type:
        criterion = BSGD()
    elif 'U_MAX' in loss_type:
        criterion = U_MAX(delta=delta if delta is not None else 0.0)
    else:
        warnings.warn('Loss type is not listed')
        return None

    return criterion

class CBCELoss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super(CBCELoss, self).__init__()
        self.reduction = reduction
    def forward(self, out, target, weight = None):
        criterion = nn.CrossEntropyLoss(weight=weight, reduction=self.reduction)
        cbloss = criterion(out, target)
        return cbloss


class MSELoss(nn.Module):
    """MSE Loss for regression tasks, compatible with DRO methods."""
    def __init__(self, reduction='none'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, out, target, weight=None):
        # Squeeze to handle shape (batch_size, 1) -> (batch_size,)
        out = out.squeeze()
        target = target.squeeze()
        loss = (out - target) ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
