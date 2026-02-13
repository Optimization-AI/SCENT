from libauc.losses.auc import pAUC_DRO_Loss
from libauc.losses.lsp import SCENT, ASGD, SOX, softplus, BSGD, DRO_Loss, U_MAX
from libauc.optimizers import SGD
from libauc.models import resnet18, resnet20, densenet121, densenet169
from libauc.datasets import CIFAR10, CIFAR100, STL10, CAT_VS_DOG
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler # data resampling (for binary class)
from libauc.metrics import pauc_roc_score
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import numpy as np
import torch
from PIL import Image
import argparse
import wandb
import os


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(model_name, num_classes=1, last_activation=None):
    """Get model by name"""
    model_map = {
        'resnet18': resnet18,
        'resnet20': resnet20,
        'densenet121': densenet121,
        'densenet169': densenet169,
    }
    if model_name.lower() not in model_map:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(model_map.keys())}")
    return model_map[model_name.lower()](pretrained=False, num_classes=num_classes, last_activation=last_activation)

def get_linear_layer_name(model_name):
    """Get the name of the final linear layer based on model architecture"""
    if model_name.lower() in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        return 'fc'
    elif model_name.lower() in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
        return 'linear'
    elif model_name.lower() in ['densenet121', 'densenet169', 'densenet201', 'densenet161']:
        return 'classifier'
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_dataset(dataset_name, root='./data'):
    """Get dataset by name"""
    dataset_map = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
        'stl10': STL10,
        'cat_vs_dog': CAT_VS_DOG,
    }
    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(dataset_map.keys())}")
    DatasetClass = dataset_map[dataset_name.lower()]
    train_data, train_targets = DatasetClass(root=root, train=True).as_array()
    test_data, test_targets = DatasetClass(root=root, train=False).as_array()
    return train_data, train_targets, test_data, test_targets

class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
       self.images = images.astype(np.uint8)
       self.targets = targets
       self.mode = mode
       self.transform_train = transforms.Compose([                                                
                              transforms.ToTensor(),
                              transforms.RandomCrop((crop_size, crop_size), padding=None),
                              transforms.RandomHorizontalFlip(),
                              transforms.Resize((image_size, image_size)),
                              ])
       self.transform_test = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize((image_size, image_size)),
                              ])
       
       # for loss function
       self.pos_indices = np.flatnonzero(targets==1)
       self.pos_index_map = {}
       for i, idx in enumerate(self.pos_indices):
           self.pos_index_map[idx] = i

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            idx = self.pos_index_map[idx] if idx in self.pos_indices else -1
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target, idx

def parse_args():
    parser = argparse.ArgumentParser(description='Training with pAUC Loss')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet20', 'densenet121', 'densenet169'],
                        help='Model architecture (default: resnet18)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10', 'cat_vs_dog', 'melanoma'],
                        help='Dataset name (default: cifar10)')

    # Loss parameters
    parser.add_argument('--loss_fn', type=str, default='SCENT',
                        choices=['SCENT', 'ASGD', 'SOX', 'softplus', 'BSGD', 'CE', 'U_MAX'],
                        help='Loss function')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--total_epochs', type=int, default=60,
                        help='Total number of epochs (default: 60)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum (default: 0.9)')
    parser.add_argument('--alpha_t', type=float, default=2.0,
                        help='Alpha_t parameter for SCENT(default: 2.0)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Alpha_t parameter for SOX(default: 0.5)')
    parser.add_argument('--lr_dual', type=float, default=1e-3,
                        help='Learning rate for dual variable(default: 1e-3)')
    parser.add_argument('--rho', type=float, default=1e-3,
                        help='Rho parameter for softplus(default: 1e-3)')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Delta parameter for U_MAX(default: 0.0)')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin parameter for loss (default: 1.0)')
    parser.add_argument('--Lambda', type=float, default=1.0,
                        help='Lambda parameter for loss (default: 1.0)')
    parser.add_argument('--sampling_rate', type=float, default=0.5,
                        help='Sampling rate (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 79)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets (default: ./data)')
    parser.add_argument('--imratio', type=float, default=0.2,
                        help='Imbalance ratio (default: 0.2)')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine'],
                        help='Learning rate scheduler (default: step)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint (default: None)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone and only train linear layer')
    parser.add_argument('--freeze_w', action='store_true',
                        help='Freeze weights and only train nu')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity (optional). Can also set WANDB_ENTITY env var.')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    wandb_entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    wandb.init(
        config=args,
        project="log_sum_exp_pauc",
        name=args.loss_fn+'_'+args.dataset+'_'+str(args.Lambda),
        entity=wandb_entity
    )
    
    # Parameters
    SEED = args.seed
    batch_size = args.batch_size
    total_epochs = args.total_epochs
    lr = args.lr
    eta = 1e1 # learning rate for control negative samples weights
    decay_epochs = [20, 40]
    decay_factor = 10

    gamma = args.gamma
    margin = args.margin
    Lambda = args.Lambda

    sampling_rate = args.sampling_rate
    num_pos = round(sampling_rate*batch_size) 
    num_neg = batch_size - num_pos

    train_data, train_targets, test_data, test_targets = get_dataset(args.dataset, root=args.data_root)
    imratio = args.imratio
    generator = ImbalancedDataGenerator(shuffle=True, verbose=True, random_seed=0)
    (train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
    (test_images, test_labels) = generator.transform(test_data, test_targets, imratio=0.5) 
    trainDataset = ImageDataset(train_images, train_labels)
    testDataset = ImageDataset(test_images, test_labels, mode='test')

    sampler = DualSampler(trainDataset, batch_size, sampling_rate=sampling_rate)
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size, sampler=sampler, shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=1)

    set_all_seeds(SEED)
    model = get_model(args.model, num_classes=1, last_activation=None) 
    model = model.cuda()

    trainable_params = model.parameters()
    
    if args.pretrained:
        PATH = args.pretrained
        checkpoint = torch.load(PATH)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        filtered = {k:v for k,v in state_dict.items() if 'fc' not in k}
        msg = model.load_state_dict(filtered, False)
        print(msg)
        model.fc.reset_parameters()
        
        if args.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            trainable_params = model.fc.parameters()
            print(f"Frozen backbone, only training fc layer")
            print(f"Trainable parameters: {sum(p.numel() for p in model.fc.parameters())}")

        if args.freeze_w:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = False
            trainable_params = model.fc.parameters()
            print(f"Frozen weights, only training nu")

    if args.loss_fn == 'SCENT':
        loss_fn = SCENT(data_len=sampler.pos_len, alpha_t=args.alpha_t, margin=margin, myLambda=Lambda)
    elif args.loss_fn == 'ASGD':
        loss_fn = ASGD(data_len=sampler.pos_len, lr_dual=args.lr_dual, margin=margin, myLambda=Lambda)
    elif args.loss_fn == 'SOX':
        loss_fn = SOX(data_len=sampler.pos_len, gamma=args.gamma, margin=margin, myLambda=Lambda)
    elif args.loss_fn == 'softplus':
        loss_fn = softplus(data_len=sampler.pos_len, rho=args.rho, lr_dual=args.lr_dual, margin=margin, myLambda=Lambda)
    elif args.loss_fn == 'BSGD':
        loss_fn = BSGD(data_len=sampler.pos_len, margin=margin, myLambda=Lambda)
    elif args.loss_fn == 'U_MAX':
        loss_fn = U_MAX(data_len=sampler.pos_len, delta=args.delta, lr_dual=args.lr_dual, margin=margin, myLambda=Lambda)
    elif args.loss_fn == 'CE':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = pAUC_DRO_Loss(data_len=sampler.pos_len, margin=margin, Lambda=Lambda)
    optimizer = SGD(trainable_params, loss_fn=loss_fn, mode='sgd', lr=lr, momentum=args.momentum)   
    dro_loss_fn = DRO_Loss(data_len=sampler.pos_len, margin=margin, myLambda=Lambda)

    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)
    else:
        scheduler = None    

    print('Start Training')
    print('-'*30)
    print(f'Model: {args.model}, Dataset: {args.dataset}, Batch Size: {batch_size}, LR: {lr}, Epochs: {total_epochs}')
    print(f'Loss params: gamma={gamma}, margin={margin}, Lambda={Lambda}')
    print(f'Pretrained: {args.pretrained}, Freeze backbone: {args.freeze_backbone}')
    print('-'*30)
    
    test_best = 0
    train_list, test_list = [], []
    loss_list = []
    for epoch in range(total_epochs):
        
        if scheduler is None and epoch in decay_epochs:
             # decrease learning rate by 10x (step decay)
            optimizer.update_lr(decay_factor=10)
                
        train_pred, train_true = [], []
        dro_loss_list = []
        model.train() 
        for idx, (data, targets, index) in enumerate(trainloader):
            data, targets  = data.cuda(), targets.cuda()
            y_pred = model(data)
            y_prob = torch.sigmoid(y_pred)
            if args.loss_fn == 'CE':
                loss = loss_fn(y_pred.squeeze(), targets.float().squeeze())
            else:
                loss = loss_fn(y_prob, targets, index=index) # postive index is selected inside loss function 
            optimizer.zero_grad()

            if args.freeze_w is False:
                loss.backward()
                optimizer.step()

            train_pred.append(y_prob.cpu().detach().numpy())
            train_true.append(targets.cpu().detach().numpy())

            dro_loss = dro_loss_fn(y_prob, targets, index=index)
            dro_loss_list.append(dro_loss.item())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_pauc = pauc_roc_score(train_true, train_pred, max_fpr=0.3)
        train_list.append(train_pauc)
        
        # evaluation
        model.eval()
        test_pred, test_true = [], [] 
        for j, data in enumerate(testloader):
            test_data, test_targets, index = data
            test_data = test_data.cuda()
            y_pred = model(test_data)
            y_prob = torch.sigmoid(y_pred)
            test_pred.append(y_prob.cpu().detach().numpy())
            test_true.append(test_targets.numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        val_pauc = pauc_roc_score(test_true, test_pred, max_fpr=0.3)
        test_list.append(val_pauc)
        
        if test_best < val_pauc:
           test_best = val_pauc
        
        model.train()
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Update lr_dual with cosine annealing (independent of optimizer lr)
        if (args.loss_fn == 'ASGD' or args.loss_fn == 'softplus' or args.loss_fn == 'U_MAX') and args.scheduler == 'cosine':
            lr_dual_current = args.lr_dual * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / total_epochs))
            loss_fn.lr_dual = lr_dual_current
        
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.lr
        print("epoch: %s, lr: %.4f, train_pauc: %.4f, test_pauc: %.4f, test_best: %.4f"%(epoch, current_lr, train_pauc, val_pauc, test_best))

        if 'SCENT' in args.loss_fn:
            gamma_t_mean = loss_fn.gamma_t.mean()
            gamma_t_max = loss_fn.gamma_t.max()
            gamma_t_min = loss_fn.gamma_t.min()
            alpha_t = loss_fn.alpha_t
            nu_val_mean = torch.log(loss_fn.u).mean()
            nu_val_max = torch.log(loss_fn.u).max()
            nu_val_min = torch.log(loss_fn.u).min()
            nu_val = torch.log(loss_fn.u)
            gamma_t = loss_fn.gamma_t
            wandb.log({"gamma_t_mean": gamma_t_mean}, step=epoch)
            wandb.log({"gamma_t_max": gamma_t_max}, step=epoch)
            wandb.log({"gamma_t_min": gamma_t_min}, step=epoch)
            wandb.log({"alpha_t": alpha_t}, step=epoch)
            wandb.log({"nu_val_mean": nu_val_mean}, step=epoch)
            wandb.log({"nu_val_max": nu_val_max}, step=epoch)
            wandb.log({"nu_val_min": nu_val_min}, step=epoch)


        if 'softplus' in args.loss_fn:
            rho = loss_fn.rho
            wandb.log({"rho": rho}, step=epoch)
        
        if 'SOX' in args.loss_fn:
            gamma_t_mean = loss_fn.gamma_t.mean()
            gamma_t_max = loss_fn.gamma_t.max()
            gamma_t_min = loss_fn.gamma_t.min()
            nu_val_mean = torch.log(loss_fn.u).mean()
            nu_val_max = torch.log(loss_fn.u).max()
            nu_val_min = torch.log(loss_fn.u).min()
            nu_val = torch.log(loss_fn.u)
            gamma_t = loss_fn.gamma_t
            wandb.log({"gamma_t_mean": gamma_t_mean}, step=epoch)
            wandb.log({"gamma_t_max": gamma_t_max}, step=epoch)
            wandb.log({"gamma_t_min": gamma_t_min}, step=epoch)
            wandb.log({"nu_val_mean": nu_val_mean}, step=epoch)
            wandb.log({"nu_val_max": nu_val_max}, step=epoch)
            wandb.log({"nu_val_min": nu_val_min}, step=epoch)
            

        if 'ASGD' in args.loss_fn:
            lr_dual = loss_fn.lr_dual
            wandb.log({"lr_dual": lr_dual}, step=epoch)

        if 'U_MAX' in args.loss_fn:
            delta = loss_fn.delta
            wandb.log({"delta": delta}, step=epoch)
            lr_dual = loss_fn.lr_dual
            wandb.log({"lr_dual": lr_dual}, step=epoch)

        if 'softplus' in args.loss_fn:
            rho = loss_fn.rho
            wandb.log({"rho": rho}, step=epoch)
            lr_dual = loss_fn.lr_dual
            wandb.log({"lr_dual": lr_dual}, step=epoch)

        dro_loss = np.mean(dro_loss_list)
        wandb.log({"dro_loss": dro_loss}, step=epoch)
        loss_list.append(dro_loss)
        wandb.log({"train_pauc": train_pauc}, step=epoch)
        wandb.log({"test_pauc": val_pauc}, step=epoch)
        wandb.log({"test_best": test_best}, step=epoch)
        print("epoch: %s, dro_loss: %.4f"%(epoch, dro_loss))