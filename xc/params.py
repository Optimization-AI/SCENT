import jsonargparse


def parse_args(args):
    p = jsonargparse.ArgumentParser()
    p.add_argument("--config-file", action=jsonargparse.ActionConfigFile)
    p.add_argument("--name", type=str, help="Name of the experiment")
    p.add_argument("--data-dir", type=str, help="Directory with data (train, val and test; features and labels)")
    p.add_argument("--feature-dim", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=360232)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data-size", type=int, default=17091657)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--algorithm", type=str, default="sox", choices=["sox", "scent", "asgd", "softplus", "bsgd"])
    p.add_argument("--warmup-epochs", type=int, default=0, help="Number of epochs for learning rate warmup")
    p.add_argument("--save-frequency", type=int, default=1, help="Frequency (in epochs) to save model checkpoints")
    p.add_argument("--softplus-rho", type=float, default=1.0)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args(args)
