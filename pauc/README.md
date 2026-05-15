# Partial AUC Maximization

PyTorch implementation of **SCENT** and several baseline methods for **partial AUC (pAUC)** maximization on imbalanced binary-classification
benchmarks.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Scripts](#training-scripts)


## Installation

```bash
pip install libauc==1.2.0
```
---

## Quick Start

```bash
# Train SCENT on CIFAR-100 with a pretrained ResNet-18 backbone
python -u main.py \
  --model resnet18 --dataset cifar100 --loss_fn SCENT \
  --alpha_t 6 --Lambda 0.1 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 5e-3 --momentum 0.0 \
  --pretrained ./checkpoints/resnet18_cifar100.pth --freeze_backbone
```

---

## Training Scripts

All commands below train on **CIFAR-100** from a pretrained ResNet-18 checkpoint

<details open>
<summary><b>SCENT (ours)</b></summary>

```bash
python -u main.py \
  --model resnet18 --dataset cifar100 --loss_fn SCENT \
  --alpha_t 6 --Lambda 0.1 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 5e-3 --momentum 0.0 \
  --pretrained ./checkpoints/resnet18_cifar100.pth --freeze_backbone
```

</details>

<details>
<summary><b>SOX</b></summary>

```bash
python -u main.py \
  --model resnet18 --dataset cifar100 --loss_fn SOX \
  --gamma 0.9 --Lambda 0.1 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 5e-3 --momentum 0.0 \
  --pretrained ./checkpoints/resnet18_cifar100.pth --freeze_backbone
```

</details>

<details>
<summary><b>ASGD</b></summary>

```bash
python -u main.py \
  --model resnet18 --dataset cifar100 --loss_fn ASGD \
  --lr_dual 1e-0 --Lambda 0.1 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 5e-3 --momentum 0.0 \
  --pretrained ./checkpoints/resnet18_cifar100.pth --freeze_backbone
```

</details>

<details>
<summary><b>ASGD (Softplus)</b></summary>

```bash
python -u main.py \
  --model resnet18 --dataset cifar100 --loss_fn softplus \
  --lr_dual 1e-3 --rho 1e-3 --Lambda 0.1 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 1e-3 --momentum 0.0 \
  --pretrained ./checkpoints/resnet18_cifar100.pth --freeze_backbone
```

</details>

<details>
<summary><b>U-MAX</b></summary>

```bash
python -u main.py \
  --model resnet18 --dataset cifar100 --loss_fn U_MAX \
  --lr_dual 1e-1 --delta 1e0 --Lambda 0.1 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 5e-3 --momentum 0.0 \
  --pretrained ./checkpoints/resnet18_cifar100.pth --freeze_backbone
```

</details>

<details>
<summary><b>BSGD</b></summary>

```bash
python -u main.py \
  --model resnet18 --dataset cifar100 --loss_fn BSGD \
  --Lambda 0.1 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 1e-2 --momentum 0.0 \
  --pretrained ./checkpoints/resnet18_cifar100.pth --freeze_backbone
```

</details>

