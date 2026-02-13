import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')
from collections import Counter
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
import torch

from .surrogate import get_surrogate_loss
from ..utils.utils import check_tensor_shape

class ASGD(nn.Module):

    def __init__(self, 
                data_len, 
                lr_dual=1e-3,
                margin=1.0,  
                myLambda=1.0,
                surr_loss='squared_hinge', 
                device=None):
        super(ASGD, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device  
        self.nu = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device)
        self.criterion = get_surrogate_loss(surr_loss)
        self.myLambda = myLambda
        self.margin = margin
        self.lr_dual = lr_dual

    def forward(self, y_pred, y_true, index):
    
        myLambda = self.myLambda
        lr_dual = self.lr_dual

        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))  
        pos_mask = (y_true == 1).squeeze().cpu()
        neg_mask = (y_true == 0).squeeze().cpu() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred): 
            index = index[pos_mask]   # indices for positive samples only         
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1) 
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns), ) 
        surr_loss = self.criterion(self.margin, f_ps - f_ns)   # shape: (len(f_ps), len(f_ns)) 
        expLoss1 = torch.exp((surr_loss) / myLambda)
        expLoss1 = expLoss1.mean(dim=1).detach()

        pos_init = (self.nu[index] == 0.0).squeeze().cpu()
        if pos_init.sum() > 0:
            init_indices = index[pos_init]
            self.nu[init_indices] = torch.log(expLoss1[pos_init].detach().unsqueeze(1))

        log_numerator = torch.log(expLoss1).unsqueeze(1)
        log_denominator = self.nu[index]
        v = log_numerator - log_denominator
        new_loss = torch.exp(v)

        self.nu[index] += myLambda * lr_dual * (new_loss.detach() - 1)

        log_numerator = surr_loss / myLambda
        log_denominator = self.nu[index]
        v = log_numerator - log_denominator
        new_loss = torch.exp(v)

        final_loss = torch.mean(new_loss.detach() * surr_loss)

        return final_loss

class U_MAX(nn.Module):

    def __init__(self, 
                data_len, 
                delta=0.0,
                lr_dual=1e-3,
                margin=1.0,  
                myLambda=1.0,
                surr_loss='squared_hinge', 
                device=None):
        super(U_MAX, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device  
        self.nu = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device)
        self.criterion = get_surrogate_loss(surr_loss)
        self.myLambda = myLambda
        self.margin = margin
        self.delta = delta
        self.lr_dual = lr_dual

    def forward(self, y_pred, y_true, index):
    
        myLambda = self.myLambda
        lr_dual = self.lr_dual

        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))  
        pos_mask = (y_true == 1).squeeze().cpu()
        neg_mask = (y_true == 0).squeeze().cpu() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred): 
            index = index[pos_mask]   # indices for positive samples only         
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1) 
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns), ) 
        surr_loss = self.criterion(self.margin, f_ps - f_ns)   # shape: (len(f_ps), len(f_ns)) 
        expLoss1 = torch.exp((surr_loss) / myLambda)
        expLoss1 = expLoss1.mean(dim=1).detach()

        pos_init = (self.nu[index] == 0.0).squeeze().cpu()
        if pos_init.sum() > 0:
            init_indices = index[pos_init]
            self.nu[init_indices] = torch.log(expLoss1[pos_init].detach().unsqueeze(1))

        log_exp_sum = torch.log(expLoss1.detach().unsqueeze(1))
        threshold = log_exp_sum - self.delta
        condition = self.nu[index] <= threshold
        self.nu[index] = torch.where(condition, log_exp_sum, self.nu[index])

        log_numerator = torch.log(expLoss1).unsqueeze(1)
        log_denominator = self.nu[index]
        v = log_numerator - log_denominator
        new_loss = torch.exp(v)

        self.nu[index] += myLambda * lr_dual * (new_loss.detach() - 1)

        log_numerator = surr_loss / myLambda
        log_denominator = self.nu[index]
        v = log_numerator - log_denominator
        new_loss = torch.exp(v)

        final_loss = torch.mean(new_loss.detach() * surr_loss)

        return final_loss

class SCENT(nn.Module):

    def __init__(self, 
                data_len, 
                alpha_t=2.0,               
                margin=1.0,  
                myLambda=1.0,
                surr_loss='squared_hinge', 
                device=None):
        super(SCENT, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device  
        self.u = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device)
        self.alpha_t = torch.tensor(-alpha_t)
        self.criterion = get_surrogate_loss(surr_loss)
        self.gamma_t = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device)
        self.myLambda = myLambda
        self.margin = margin

    def forward(self, y_pred, y_true, index):
    
        myLambda = self.myLambda

        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))  
        pos_mask = (y_true == 1).squeeze().cpu()
        neg_mask = (y_true == 0).squeeze().cpu() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred): 
            index = index[pos_mask]   # indices for positive samples only         
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1) 
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns), ) 
        surr_loss = self.criterion(self.margin, f_ps - f_ns)   # shape: (len(f_ps), len(f_ns)) 
        expLoss1 = torch.exp((surr_loss) / myLambda)
        expLoss1 = expLoss1.mean(dim=1).detach()

        pos_init = (self.u[index] == 0.0).squeeze().cpu()
        if pos_init.sum() > 0:
            init_indices = index[pos_init]
            self.u[init_indices] = expLoss1[pos_init].detach().unsqueeze(1)

        log_alpha = self.alpha_t 
        x = log_alpha + torch.log(self.u)
        self.gamma_t = torch.sigmoid(x)

        self.u[index] = (1-self.gamma_t[index]) * self.u[index] + self.gamma_t[index] * expLoss1.unsqueeze(1)

        log_numerator = surr_loss / myLambda
        log_denominator = torch.log(self.u[index])
        v = log_numerator - log_denominator
        new_loss = torch.exp(v)

        final_loss = torch.mean(new_loss.detach() * surr_loss)

        return final_loss

class SOX(nn.Module):

    def __init__(self, 
                data_len, 
                gamma=0.5,               
                margin=1.0,  
                myLambda=1.0,
                surr_loss='squared_hinge', 
                device=None):
        super(SOX, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device  
        self.u = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device)
        self.criterion = get_surrogate_loss(surr_loss)
        self.gamma_t = torch.tensor([gamma]*data_len).view(-1, 1).to(self.device)
        self.myLambda = myLambda
        self.margin = margin

    def forward(self, y_pred, y_true, index):
    
        myLambda = self.myLambda

        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))  
        pos_mask = (y_true == 1).squeeze().cpu()
        neg_mask = (y_true == 0).squeeze().cpu() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred): 
            index = index[pos_mask]   # indices for positive samples only         
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1) 
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns), ) 
        surr_loss = self.criterion(self.margin, f_ps - f_ns)   # shape: (len(f_ps), len(f_ns)) 
        expLoss1 = torch.exp((surr_loss) / myLambda)
        expLoss1 = expLoss1.mean(dim=1).detach()

        pos_init = (self.u[index] == 0.0).squeeze().cpu()
        if pos_init.sum() > 0:
            init_indices = index[pos_init]
            self.u[init_indices] = expLoss1[pos_init].detach().unsqueeze(1)

        self.u[index] = (1-self.gamma_t[index]) * self.u[index] + self.gamma_t[index] * expLoss1.unsqueeze(1)

        log_numerator = surr_loss / myLambda
        log_denominator = torch.log(self.u[index])
        v = log_numerator - log_denominator
        new_loss = torch.exp(v)

        final_loss = torch.mean(new_loss.detach() * surr_loss)

        return final_loss


class softplus(nn.Module):

    def __init__(self, 
                data_len, 
                rho=1e-3,
                lr_dual=1e-3,
                margin=1.0,  
                myLambda=1.0,
                surr_loss='squared_hinge', 
                device=None):
        super(softplus, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device  
        self.criterion = get_surrogate_loss(surr_loss)
        self.myLambda = myLambda
        self.margin = margin
        self.alpha = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device)
        self.rho = rho
        self.lr_dual = lr_dual

    def forward(self, y_pred, y_true, index):

        myLambda = self.myLambda

        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))  
        pos_mask = (y_true == 1).squeeze().cpu()
        neg_mask = (y_true == 0).squeeze().cpu() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred): 
            index = index[pos_mask]   # indices for positive samples only         
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1) 
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns), ) 
        surr_loss = self.criterion(self.margin, f_ps - f_ns)   # shape: (len(f_ps), len(f_ns)) 

        exponent = (surr_loss - self.alpha[index]) / myLambda + math.log(self.rho)
        exponent_ = (surr_loss - self.alpha[index]) / myLambda
        w = torch.sigmoid(exponent_ + math.log(self.rho))

        loss = (myLambda / self.rho) * F.softplus(exponent).mean() + self.alpha[index].mean()

        self.alpha[index] = self.alpha[index] - self.lr_dual * (1 - w.detach().mean(dim=1, keepdim=True))    
        self.alpha[index] = self.alpha[index].detach()
        return loss

    
class BSGD(nn.Module):

    def __init__(self, 
                data_len, 
                margin=1.0,  
                myLambda=1.0,
                surr_loss='squared_hinge', 
                device=None):
        super(BSGD, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device  
        self.criterion = get_surrogate_loss(surr_loss)
        self.myLambda = myLambda
        self.margin = margin

    def forward(self, y_pred, y_true, index):
    
        myLambda = self.myLambda

        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))  
        pos_mask = (y_true == 1).squeeze().cpu()
        neg_mask = (y_true == 0).squeeze().cpu()   # indices for positive samples only         
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1) 
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns), ) 
        surr_loss = self.criterion(self.margin, f_ps - f_ns)   # shape: (len(f_ps), len(f_ns)) 
        expLoss = torch.exp((surr_loss) / myLambda)
        expLoss = expLoss.mean(dim=1)

        lsp = torch.log(expLoss)
        final_loss = myLambda * lsp.mean()

        return final_loss


class DRO_Loss(nn.Module):

    def __init__(self, 
                data_len, 
                margin=1.0,  
                myLambda=1.0,
                surr_loss='squared_hinge', 
                device=None):
        super(DRO_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device  
        self.criterion = get_surrogate_loss(surr_loss)
        self.myLambda = myLambda
        self.margin = margin

    def forward(self, y_pred, y_true, index):
    
        myLambda = self.myLambda

        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))  
        pos_mask = (y_true == 1).squeeze().cpu()
        neg_mask = (y_true == 0).squeeze().cpu() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred): 
            index = index[pos_mask]   # indices for positive samples only         
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1) 
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns), ) 
        surr_loss = self.criterion(self.margin, f_ps - f_ns)   # shape: (len(f_ps), len(f_ns)) 
        expLoss = torch.exp((surr_loss) / myLambda)
        expLoss = expLoss.mean(dim=1).detach()

        lsp = torch.log(expLoss)
        final_loss = myLambda * lsp.mean()

        return final_loss