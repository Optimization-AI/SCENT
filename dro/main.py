# Acknowledge: This code is adapted from the framework provided in paper https://arxiv.org/abs/2509.24894.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pickle
import multiprocessing as mp
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from mylosses import get_train_loss
from utils import load_and_preprocess_data, load_abalone_data, LinearRegressionModel, train_linreg, compute_final_objective_stats


def logsumexp_over_dataset(model: nn.Module, X, y, lam=1.):
    with torch.no_grad():
        preds = model(X)
        errors = (preds - y) ** 2
        L = errors.squeeze(1)
    lse = torch.logsumexp(L/lam, dim=0).item() - math.log(len(y))
    return lam * lse

def softplus_approx(preds: torch.Tensor, targets: torch.Tensor, model: LinearRegressionModel, rho: float, lam=1.):
    errors = (preds - targets) ** 2
    L = errors.squeeze(1)
    exponent = (L - model.alpha) / lam + math.log(rho)
    loss = (lam / rho) * F.softplus(exponent).mean() + model.alpha
    return loss


def train(dataset, batch_sz, lr, method, seed, lam, weights_file='linreg_weights.pt', gamma=None, alpha_t=None):
    """
    Train model with specified method.
    
    Args:
        method: 'BSGD', 'softplus', 'SOX', 'ASGD', or 'SENT'
        gamma: Hyperparameter for SOX method (only used when method='SOX')
        alpha_t: Hyperparameter for SENT method (only used when method='SENT')
    """
    if method == 'SOX' and gamma is not None:
        print(f"Running with method={method}, lam={lam}, batch_sz={batch_sz}, lr={lr}, gamma={gamma}, seed={seed}")
    elif method == 'SENT' and alpha_t is not None:
        print(f"Running with method={method}, lam={lam}, batch_sz={batch_sz}, lr={lr}, alpha_t={alpha_t}, seed={seed}")
    else:
        print(f"Running with method={method}, lam={lam}, batch_sz={batch_sz}, lr={lr}, seed={seed}")
    torch.manual_seed(seed)
    train_loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)
    X, y = dataset.tensors

    # Determine if model needs alpha parameter (for softplus method)
    model = LinearRegressionModel(input_dim=X.shape[1], with_alpha=(method == 'softplus') is not None, weights_file=weights_file)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    criterion = get_train_loss(loss_type=method, gamma=gamma, alpha_t=alpha_t) if method in ['SOX', 'ASGD', 'SENT', 'BSGD', 'U_MAX'] else None

    n_epochs = 300
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    logsumexp_vals = [logsumexp_over_dataset(model, X, y, lam=lam)]
    epochs_passed = [0]

    # Tracking variables per epoch
    tracking_data = {
        'batch_sumexp_vals': [],
        'dataset_sumexp_vals': [],
        'dataset_dro_vals': [],
        'loss_max': [],
        'loss_min': [],
        'loss': [],
    }

    iter_count = 0

    for epoch in range(1, n_epochs + 1):
        
        for X_batch, y_batch in train_loader:

            obj_val = logsumexp_over_dataset(model, X_batch, y_batch, lam=lam)

            iter_count += 1
            optimizer.zero_grad()
            preds = model(X_batch)
            # Compute batch logsumexp metric

            if iter_count % 100 == 0:

                with torch.no_grad():
                    errors = (preds - y_batch) ** 2 
                    L = errors.squeeze(1)
                    batch_se = torch.exp(L/lam).mean().item()
                    tracking_data['batch_sumexp_vals'].append(batch_se)

                # Compute dataset-level logsumexp
                with torch.no_grad():
                    all_preds = model(X)
                    all_errors = (all_preds - y) ** 2
                    all_L = all_errors.squeeze(1)
                    loss_max = L.max().detach()
                    loss_min = L.min().detach()
                    loss = L.mean().detach()
                    dataset_se = torch.exp(all_L/lam).mean().item()
                    dataset_dro = lam * (torch.logsumexp(all_L/lam, dim=0) - math.log(all_L.numel()))
                    tracking_data['dataset_sumexp_vals'].append(dataset_se)
                    tracking_data['dataset_dro_vals'].append(dataset_dro)
                    tracking_data['loss_max'].append(loss_max)
                    tracking_data['loss_min'].append(loss_min)
                    tracking_data['loss'].append(loss)


            # Compute loss based on method
            if method == 'BSGD':
                loss = criterion(preds, y_batch, None, myLambda=lam)
            elif method == 'softplus':
                loss = softplus_approx(preds, y_batch, model, rho=1e-3, lam=lam)
            elif method == "ASGD":
                loss = criterion(preds, y_batch, None, myLambda=lam)
            elif method == "SENT":
                loss = criterion(preds, y_batch, None, myLambda=lam)
            elif method == "SOX":
                loss = criterion(preds, y_batch, None, myLambda=lam)
            elif method == "U_MAX":
                loss = criterion(preds, y_batch, None, myLambda=lam)
            else:
                raise ValueError(f"Unknown method: {method}")
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            obj_val = logsumexp_over_dataset(model, X, y, lam=lam)

            if torch.isfinite(torch.tensor(obj_val)):
                epochs_passed.append(epoch)
                logsumexp_vals.append(obj_val)
            else:
                print(f'inf or nan, batch{batch_sz}_lr{lr}_method{method}_seed{seed}')
                return 0
        
        scheduler.step()

    print("iter_count", iter_count)

    fname = method.lower()
    
    if method == 'SOX' and gamma is not None:
        fname += f'_lam{lam}_batch{batch_sz}_lr{lr}_gamma{gamma}_method{method}_seed{seed}.pickle'
    elif method == 'SENT' and alpha_t is not None:
        fname += f'_lam{lam}_batch{batch_sz}_lr{lr}_alpha_t{alpha_t}_method{method}_seed{seed}.pickle'
    else:
        fname += f'_lam{lam}_batch{batch_sz}_lr{lr}_method{method}_seed{seed}.pickle'
    with open("trajectories/" + fname, "wb") as f:
        pickle.dump((epochs_passed, logsumexp_vals), f)
    
    # Save tracking data to separate pickle file
    if method == 'SOX' and gamma is not None:
        tracking_fname = f'tracking_lam{lam}_batch{batch_sz}_lr{lr}_gamma{gamma}_method{method}_seed{seed}.pickle'
    elif method == 'SENT' and alpha_t is not None:
        tracking_fname = f'tracking_lam{lam}_batch{batch_sz}_lr{lr}_alpha_t{alpha_t}_method{method}_seed{seed}.pickle'
    else:
        tracking_fname = f'tracking_lam{lam}_batch{batch_sz}_lr{lr}_method{method}_seed{seed}.pickle'
    with open("trajectories/" + tracking_fname, "wb") as f:
        pickle.dump(tracking_data, f)
    if method == 'SOX' and gamma is not None:
        print(f"Finished run with method={method}, lam={lam}, batch_sz={batch_sz}, lr={lr}, gamma={gamma}, seed={seed}")
    elif method == 'SENT' and alpha_t is not None:
        print(f"Finished run with method={method}, lam={lam}, batch_sz={batch_sz}, lr={lr}, alpha_t={alpha_t}, seed={seed}")
    else:
        print(f"Finished run with method={method}, lam={lam}, batch_sz={batch_sz}, lr={lr}, seed={seed}")


def collect_and_save_json_results(params, seeds, dataset_name='california_housing', 
                                  trajectories_dir='trajectories', json_dir='results_json'):
    """
    Collect all seed results for each experiment and save to JSON files.
    
    Args:
        params: List of (method, lam, batch_sz, lr) or (method, lam, batch_sz, lr, hyperparam) tuples
               For SOX method, hyperparam is gamma; for SENT method, hyperparam is alpha_t
        seeds: List of seed values used
        dataset_name: Name of the dataset
        trajectories_dir: Directory containing pickle files
        json_dir: Directory to save JSON files
    """
    # Create JSON directory if it doesn't exist
    os.makedirs(json_dir, exist_ok=True)
    
    for param_tuple in params:
        # Handle both old format (method, lam, batch_sz, lr) and new format with hyperparameter
        if len(param_tuple) == 5:
            method, lam, batch_sz, lr, hyperparam = param_tuple
            if method == 'SOX':
                gamma = hyperparam
                alpha_t = None
            elif method == 'SENT':
                gamma = None
                alpha_t = hyperparam
            else:
                gamma = None
                alpha_t = None
        else:
            method, lam, batch_sz, lr = param_tuple
            gamma = None
            alpha_t = None

        prefix = method.lower()
        
        experiment_config = {
            'method': method,
            'lam': lam,
            'batch_sz': batch_sz,
            'lr': lr,
            'dataset': dataset_name,
            'n_seeds': len(seeds)
        }
        if gamma is not None:
            experiment_config['gamma'] = gamma
        if alpha_t is not None:
            experiment_config['alpha_t'] = alpha_t
        
        experiment_results = {
            'experiment_config': experiment_config,
            'seeds': {}
        }
        
        for seed in seeds:
            # Load main trajectory file
            if method == 'SOX' and gamma is not None:
                fname = f'{prefix}_lam{lam}_batch{batch_sz}_lr{lr}_gamma{gamma}_method{method}_seed{seed}.pickle'
            elif method == 'SENT' and alpha_t is not None:
                fname = f'{prefix}_lam{lam}_batch{batch_sz}_lr{lr}_alpha_t{alpha_t}_method{method}_seed{seed}.pickle'
            else:
                fname = f'{prefix}_lam{lam}_batch{batch_sz}_lr{lr}_method{method}_seed{seed}.pickle'
            filepath = os.path.join(trajectories_dir, fname)
            
            # Load tracking file
            if method == 'SOX' and gamma is not None:
                tracking_fname = f'tracking_lam{lam}_batch{batch_sz}_lr{lr}_gamma{gamma}_method{method}_seed{seed}.pickle'
            elif method == 'SENT' and alpha_t is not None:
                tracking_fname = f'tracking_lam{lam}_batch{batch_sz}_lr{lr}_alpha_t{alpha_t}_method{method}_seed{seed}.pickle'
            else:
                tracking_fname = f'tracking_lam{lam}_batch{batch_sz}_lr{lr}_method{method}_seed{seed}.pickle'
            tracking_filepath = os.path.join(trajectories_dir, tracking_fname)
            
            seed_data = {
                'epochs_passed': None,
                'logsumexp_vals': None,
                'tracking_data': None,
                'file_exists': False
            }
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        epochs_passed, logsumexp_vals = pickle.load(f)
                    # Convert to Python native types for JSON
                    seed_data['epochs_passed'] = [int(ep) for ep in epochs_passed]
                    seed_data['logsumexp_vals'] = [float(val) for val in logsumexp_vals]
                    seed_data['file_exists'] = True
                except Exception as e:
                    print(f"Warning: Could not load {fname}: {e}")
            
            if os.path.exists(tracking_filepath):
                try:
                    with open(tracking_filepath, 'rb') as f:
                        tracking_data = pickle.load(f)
                    # Convert numpy arrays and torch tensors to lists for JSON serialization
                    tracking_data_json = {}
                    for key, value in tracking_data.items():
                        if isinstance(value, list):
                            # Convert any numpy/torch types in list to Python native types
                            tracking_data_json[key] = [
                                float(v.item()) if isinstance(v, torch.Tensor) 
                                else float(v) if isinstance(v, (np.number, np.ndarray)) 
                                else v 
                                for v in value
                            ]
                        elif isinstance(value, torch.Tensor):
                            tracking_data_json[key] = float(value.item())
                        elif isinstance(value, (np.number, np.ndarray)):
                            tracking_data_json[key] = float(value)
                        else:
                            tracking_data_json[key] = value
                    seed_data['tracking_data'] = tracking_data_json
                except Exception as e:
                    print(f"Warning: Could not load {tracking_fname}: {e}")
            
            experiment_results['seeds'][str(seed)] = seed_data
        
        if method == 'SOX' and gamma is not None:
            json_filename = f'experiment_dataset{dataset_name}_method{method}_lam{lam}_batch{batch_sz}_lr{lr}_gamma{gamma}.json'
        elif method == 'SENT' and alpha_t is not None:
            json_filename = f'experiment_dataset{dataset_name}_method{method}_lam{lam}_batch{batch_sz}_lr{lr}_alpha_t{alpha_t}.json'
        else:
            json_filename = f'experiment_dataset{dataset_name}_method{method}_lam{lam}_batch{batch_sz}_lr{lr}.json'
        json_filepath = os.path.join(json_dir, json_filename)
        
        # Save to JSON (overwrite if exists)
        # Remove existing file first to ensure clean overwrite
        if os.path.exists(json_filepath):
            os.remove(json_filepath)
        with open(json_filepath, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        # Count successful seeds
        successful_seeds = sum(1 for seed_data in experiment_results['seeds'].values() 
                              if seed_data['file_exists'])
        if method == 'SOX' and gamma is not None:
            print(f"Saved JSON for method={method}, lam={lam}, batch_sz={batch_sz}, lr={lr}, gamma={gamma}: "
                  f"{successful_seeds}/{len(seeds)} seeds -> {json_filepath}")
        elif method == 'SENT' and alpha_t is not None:
            print(f"Saved JSON for method={method}, lam={lam}, batch_sz={batch_sz}, lr={lr}, alpha_t={alpha_t}: "
                  f"{successful_seeds}/{len(seeds)} seeds -> {json_filepath}")
        else:
            print(f"Saved JSON for method={method}, lam={lam}, batch_sz={batch_sz}, lr={lr}: "
                  f"{successful_seeds}/{len(seeds)} seeds -> {json_filepath}")


def main(dataset_name='california_housing'):
    """
    Main training function.
    
    Args:
        dataset_name: 'california_housing' or 'abalone'
    """
    if dataset_name == 'abalone':
        abalone_path = 'abalone.txt' if os.path.exists('abalone.txt') else 'kl-dro/abalone.txt'
        dataset, X_np, y_np = load_abalone_data(abalone_path)
        weights_filename = 'linreg_weights_abalone.pt'
    else:
        dataset, X_np, y_np = load_and_preprocess_data()
        weights_filename = 'linreg_weights.pt'
    
    # Print dataset target range before training
    print(f"Dataset: {dataset_name}")
    print(f"Target range: min={y_np.min():.4f}, max={y_np.max():.4f}")
    print(f"Target mean: {y_np.mean():.4f}, std: {y_np.std():.4f}")
    print(f"Dataset size: {len(y_np)}")
    
    train_linreg(X_np, y_np, filename=weights_filename)  # Store least squares solution for initialization

    params = [  # (method, lam, batch_sz, lr) or (method, lam, batch_sz, lr, hyperparam) 
                # For SOX: hyperparam is gamma; for SENT: hyperparam is alpha_t

        # ('BSGD', 0.2, 100, 1e-5),
        # ('BSGD', 1., 100, 5e-6),
        # ('BSGD', 5., 100, 5e-6),

        # ('softplus', 0.2, 100, 1e-6),
        # ('softplus', 1.0, 100, 1e-6),
        # ('softplus', 5.0, 100, 1e-5),

        # ('U_MAX', 0.2, 100, 1e-5),
        # ('U_MAX', 1., 100, 5e-6), 
        # ('U_MAX', 5., 100, 1e-4),
       
        # ('SOX', 0.2, 100, 5e-6, 0.5),      
        # ('SOX', 1., 100, 5e-6, 0.4), 
        # ('SOX', 5., 100, 1e-5, 0.8),            

        ('SENT', 0.2, 100, 1e-5, 22.0),
        ('SENT', 1., 100, 5e-6, 4.0), 
        ('SENT', 5., 100, 1e-5, 1.1),

    ]

    seeds = list(range(10))
    tasks = []
    for param_tuple in params:
        if len(param_tuple) == 5:
            method, lam, batch_sz, lr, hyperparam = param_tuple
            if method == 'SOX':
                gamma = hyperparam
                alpha_t = None
            elif method == 'SENT':
                gamma = None
                alpha_t = hyperparam
            else:
                gamma = None
                alpha_t = None
            tasks.extend([(dataset, batch_sz, lr, method, seed, lam, weights_filename, gamma, alpha_t)
                         for seed in seeds])
        else:
            method, lam, batch_sz, lr = param_tuple
            tasks.extend([(dataset, batch_sz, lr, method, seed, lam, weights_filename, None, None)
                         for seed in seeds])

    with mp.Pool(processes=32) as pool:
        pool.starmap(train, tasks) 

    compute_final_objective_stats(params)
    
    collect_and_save_json_results(params, seeds, dataset_name=dataset_name)


if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'california_housing'
    main(dataset_name=dataset_name)
