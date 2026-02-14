import os
import logging
import pathlib
import json
import sys
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from model.lin_model import LinearClassifier
from data.feat_data import FeaturesDataset
from loss import SOXLoss, SoftPlusLoss, PrimalLoss
from params import parse_args

def setup_logging(out_log_file=None):
    logging.root.setLevel(level=logging.INFO)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    stream_handler = logging.StreamHandler()
    logging.root.addHandler(stream_handler)
    if out_log_file is not None:
        file_handler = logging.FileHandler(out_log_file)
        logging.root.addHandler(file_handler)
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)


def build_dataloaders(root_data_dir, batch_size):
    dataloader_list = []
    for split, shuffle in zip(["train", "val", "test"], [True, False, False]):
        data_dir = os.path.join(root_data_dir, split)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
        features = os.path.join(data_dir, "features.pt")
        labels = os.path.join(data_dir, "labels.pt")
        ds = FeaturesDataset(features, labels)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        dataloader_list.append(dataloader)
    train_loader, val_loader, test_loader = dataloader_list
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    for i, batch in enumerate(loader):
        feats, labels, indices = batch
        feats = feats.to(device)
        logits = model(feats, labels)
        loss_dict = criterion(logits, indices)
        loss = loss_dict["loss"]
        with torch.no_grad():
            model.eval()
            labels = labels.to(device, dtype=torch.long)
            train_loss = F.cross_entropy(model.fc(feats), labels)
            loss_dict["train_loss"] = train_loss
            model.train()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += train_loss.item() * feats.size(0)
        total += feats.size(0)

        if i % 100 == 0:
            log_str = f"  Batch {i} / {len(loader)}:"
            for key, value in loss_dict.items():
                log_str += f" {key}={value.item():.6f}"
            logging.info(log_str)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(loader):
        feats, labels, _ = batch
        feats = feats.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits = model.fc(feats)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * feats.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += feats.size(0)

        if i % 100 == 0:
            logging.info(f"  Batch {i} / {len(loader)}: loss={loss.item():.6f}")

    return total_loss / total, correct / total


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, criterion, device):
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
    if "criterion_nu" in checkpoint and hasattr(criterion, "nu"):
        criterion.nu = checkpoint["criterion_nu"].to(device)
        logging.info("Restored criterion nu from checkpoint")

    if hasattr(lr_scheduler, "step"):
        for _ in range(start_epoch):
            lr_scheduler.step()

    logging.info(f"Resumed from epoch {start_epoch}")
    return start_epoch


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {seed}")


def get_device(device_str):
    if not device_str:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_str
    logging.info(f"Using device: {device}")
    return device


def main():
    args = parse_args(sys.argv[1:])
    device = get_device(args.device)

    out_dir = pathlib.Path(args.out_dir) / args.name
    os.makedirs(out_dir / "checkpoints", exist_ok=True)
    out_log_file = out_dir / "out.log"
    setup_logging(out_log_file=out_log_file)

    set_seed(args.seed)

    train_loader, val_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size)

    model = LinearClassifier(args.feature_dim, args.num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.algorithm == "scent":
        criterion = SOXLoss(data_size=args.data_size, gamma=args.gamma, is_scent=True)
    elif args.algorithm == "sox":
        if args.gamma == 0.0:
            args.gamma = 4.177 * args.epochs ** (-0.7265)
            logging.info(f"Setting gamma to {args.gamma:.6f} based on epochs={args.epochs}")
        criterion = SOXLoss(data_size=args.data_size, gamma=args.gamma)
    elif args.algorithm == "asgd":
        criterion = SOXLoss(data_size=args.data_size, gamma=args.gamma, is_sox=False)
    elif args.algorithm == "softplus":
        criterion = SoftPlusLoss(data_size=args.data_size, gamma=args.gamma, rho=args.softplus_rho)
    elif args.algorithm == "bsgd":
        criterion = PrimalLoss()
    elif args.algorithm == "umax":
        criterion = SOXLoss(data_size=args.data_size, gamma=args.gamma, is_sox=False, umax_delta=args.umax_delta)
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not implemented.")

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_test_loss = float("inf")
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, lr_scheduler, criterion, device)

        # Try to restore best metrics from eval.jsonl if it exists
        eval_file = out_dir / f"eval_{args.name}.jsonl"
        if eval_file.exists():
            logging.info(f"Restoring best metrics from eval_{args.name}.jsonl")
            with open(eval_file, "r") as f:
                for line in f:
                    result = json.loads(line)
                    if result["epoch"] <= start_epoch:
                        best_val_acc = max(best_val_acc, result["best_val_acc"])
                        best_test_acc = max(best_test_acc, result["best_test_acc"])
                        if result["best_test_loss"] < best_test_loss:
                            best_test_loss = result["best_test_loss"]
            logging.info(f"Restored best metrics: val_acc={best_val_acc:.6f} test_acc={best_test_acc:.6f} test_loss={best_test_loss:.6f}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        logging.info(f"Epoch {epoch}/{args.epochs}, learning_rate={lr_scheduler.get_last_lr()[0]:.6f}")
        if hasattr(criterion, "adjust_gamma"):
            criterion.adjust_gamma(epoch, args.epochs)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        lr_scheduler.step()

        logging.info("Evaluating on validation set...")
        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info("New best model found, evaluating on test set...")
            best_test_loss, best_test_acc = evaluate(model, test_loader, device)
        logging.info(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f}")
        logging.info(f"  Best val_acc={best_val_acc:.6f} best_test_acc={best_test_acc:.6f} best_test_loss={best_test_loss:.6f}")
        eval_results = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "best_test_loss": best_test_loss,
        }
        with open(out_dir / f"eval_{args.name}.jsonl", "a") as f:
            f.write(json.dumps(eval_results) + "\n")

        if epoch % args.save_frequency == 0 or epoch == args.epochs:
            save_dict = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
            }
            if hasattr(criterion, "nu"):
                save_dict["criterion_nu"] = criterion.nu.cpu()
            torch.save(save_dict, out_dir / "checkpoints" / f"epoch_{epoch}.pt")
            logging.info(f"Saved checkpoint for epoch {epoch}.")


if __name__ == "__main__":
    main()
