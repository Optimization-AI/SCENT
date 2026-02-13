import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os


def load_and_preprocess_data():
    raw = fetch_california_housing()
    X = raw.data
    y = raw.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tX = torch.from_numpy(X_scaled.astype(np.float32))
    ty = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    dataset = TensorDataset(tX, ty)

    return dataset, X_scaled, y


def load_abalone_data(filepath=None, n_features=12):
    """
    Load and preprocess abalone dataset from libsvm format.
    
    Args:
        filepath: Path to the abalone.txt file (default: 'abalone.txt' in current dir or 'kl-dro/abalone.txt')
        n_features: Number of features (default: 12)
    
    Returns:
        dataset: TensorDataset containing (X, y)
        X_scaled: Scaled feature matrix (numpy array)
        y: Target values (numpy array)
    """
    if filepath is None:
        # Try to find the file in common locations
        if os.path.exists('abalone.txt'):
            filepath = 'abalone.txt'
        else:
            raise FileNotFoundError("Could not find abalone.txt. Please specify the filepath.")
    
    X = []
    y = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            target = float(parts[0])
            y.append(target)
            
            features = np.zeros(n_features)
            
            for part in parts[1:]:
                if ':' in part:
                    idx_str, val_str = part.split(':')
                    idx = int(idx_str) - 1
                    val = float(val_str)
                    if 0 <= idx < n_features:
                        features[idx] = val
            
            X.append(features)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Loaded abalone dataset: {len(y)} samples, {n_features} features")
    print(f"Original target range: [{y.min():.2f}, {y.max():.2f}]")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_min, y_max = y.min(), y.max()
    y_normalized = (y - y_min) / (y_max - y_min) * 10.0
    print(f"Normalized target range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")
    
    tX = torch.from_numpy(X_scaled.astype(np.float32))
    ty = torch.from_numpy(y_normalized.astype(np.float32)).unsqueeze(1)
    dataset = TensorDataset(tX, ty)
    
    return dataset, X_scaled, y_normalized


def train_linreg(X, y, filename='linreg_weights.pt'):
    model = LinearRegression(fit_intercept=True).fit(X, y)
    state_dict = {
        'weights': torch.tensor(model.coef_, dtype=torch.float32),
        'bias': torch.tensor(model.intercept_, dtype=torch.float32)
    }
    torch.save(state_dict, filename)
    return state_dict


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim: int, with_alpha: bool = False, weights_file: str = None):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

        if with_alpha:
            self.alpha = nn.Parameter(torch.zeros(()))

        # Load pretrained weights if weights_file is provided
        if weights_file is not None and os.path.exists(weights_file):
            state_dict = torch.load(weights_file, weights_only=True)
            self.linear.weight.data = state_dict['weights'].unsqueeze(0)
            self.linear.bias.data = state_dict['bias'].unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def compute_final_objective_stats(data_tuples_lr, n_seeds=10, data_dir='trajectories'):
    """
    Compute final objective statistics for experiments.
    
    Args:
        data_tuples_lr: List of (method, lam, batch_sz, lr) or (method, lam, batch_sz, lr, hyperparam) tuples
                       For SOX method, hyperparam is gamma; for SENT method, hyperparam is alpha_t
        n_seeds: Number of seeds
        data_dir: Directory containing pickle files
    """
    data_tuples_obj = []

    for param_tuple in data_tuples_lr:
        # Handle both old format (method, lam, batch_sz, lr) and new format with hyperparameter
        if len(param_tuple) == 5:
            method, lam, batch_sz, lr, hyperparam = param_tuple
            # Determine if it's gamma (for SOX) or alpha_t (for SENT)
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
        
        final_objectives = []

        prefix = method.lower()  # For SOX, ASGD, SENT

        for seed in range(n_seeds):
            if method == 'SOX' and gamma is not None:
                fname = f'{prefix}_lam{lam}_batch{batch_sz}_lr{lr}_gamma{gamma}_method{method}_seed{seed}.pickle'
            elif method == 'SENT' and alpha_t is not None:
                fname = f'{prefix}_lam{lam}_batch{batch_sz}_lr{lr}_alpha_t{alpha_t}_method{method}_seed{seed}.pickle'
            else:
                fname = f'{prefix}_lam{lam}_batch{batch_sz}_lr{lr}_method{method}_seed{seed}.pickle'
            filepath = os.path.join(data_dir, fname)

            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        epochs_passed, logsumexp_vals = pickle.load(f)

                    if len(logsumexp_vals) > 0:
                        final_obj = logsumexp_vals[-1]
                        final_objectives.append(final_obj)
                    else:
                        print(f"Warning: Empty trajectory in {fname}")

                except Exception as e:
                    print(f"Warning: Could not load {fname}: {e}")
            else:
                print(f"Warning: File {fname} not found")

        if len(final_objectives) >= 1:
            mean_final_obj = np.mean(final_objectives)
            std_final_obj = np.std(final_objectives)
            data_tuples_obj.append((method, lam, batch_sz, mean_final_obj, std_final_obj))
            if method == 'SOX' and gamma is not None:
                print(f"Computed: method={method}, lam={lam}, B={batch_sz}, lr={lr}, gamma={gamma}: "
                      f"{len(final_objectives)}/{n_seeds} seeds, "
                      f"mean={mean_final_obj:.4f}, std={std_final_obj:.4f}")
            elif method == 'SENT' and alpha_t is not None:
                print(f"Computed: method={method}, lam={lam}, B={batch_sz}, lr={lr}, alpha_t={alpha_t}: "
                      f"{len(final_objectives)}/{n_seeds} seeds, "
                      f"mean={mean_final_obj:.4f}, std={std_final_obj:.4f}")
            else:
                print(f"Computed: method={method}, lam={lam}, B={batch_sz}, lr={lr}: "
                      f"{len(final_objectives)}/{n_seeds} seeds, "
                      f"mean={mean_final_obj:.4f}, std={std_final_obj:.4f}")
        else:
            if method == 'SOX' and gamma is not None:
                print(f"Error: Insufficient data for method={method}, lam={lam}, B={batch_sz}, lr={lr}, gamma={gamma}: "
                      f"only {len(final_objectives)} seeds available")
            elif method == 'SENT' and alpha_t is not None:
                print(f"Error: Insufficient data for method={method}, lam={lam}, B={batch_sz}, lr={lr}, alpha_t={alpha_t}: "
                      f"only {len(final_objectives)} seeds available")
            else:
                print(f"Error: Insufficient data for method={method}, lam={lam}, B={batch_sz}, lr={lr}: "
                      f"only {len(final_objectives)} seeds available")

    return data_tuples_obj
