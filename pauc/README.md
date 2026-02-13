# Partial AUC Maximization

## Getting Started

### Environment Setup

To set up the environment for training, please install the required dependencies by running the following command:
```
pip install libauc==1.2.0
```

### Training

<details open>
    <summary>Sample script to run <b>SCENT</b> on CIFAR-100</summary>

```bash

python -u main.py \
  --model resnet18 --dataset cifar100 --Lambda 0.1 \
  --loss_fn SCENT --alpha_t 4 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 1e-3 \
  --momentum 0.0 --pretrained ./checkpoints/resnet18_cifar100.pth \
  --freeze_backbone
```

</details>


<details open>
  <summary>Sample script to run <b>SOX</b> on CIFAR-100</summary>

```bash

python -u main.py \
  --model resnet18 --dataset cifar100 --Lambda 0.1 \
  --loss_fn SOX --gamma 0.9 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 1e-3 \
  --momentum 0.0 --pretrained ./checkpoints/resnet18_cifar100.pth \
  --freeze_backbone
```

</details>

<details open>
  <summary>Sample script to run <b>BSGD</b> on CIFAR-100</summary>

```bash

python -u main.py \
  --model resnet18 --dataset cifar100 --Lambda 0.1 \
  --loss_fn BSGD --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 1e-3 \
  --momentum 0.0 --pretrained ./checkpoints/resnet18_cifar100.pth \
  --freeze_backbone
```

</details>

<details open>
  <summary>Sample script to run <b>ASGD</b> on CIFAR-100</summary>

```bash

python -u main.py \
  --model resnet18 --dataset cifar100 --Lambda 0.1 \
  --loss_fn ASGD --lr_dual 1e-1 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 1e-4 \
  --momentum 0.0 --pretrained ./checkpoints/resnet18_cifar100.pth \
  --freeze_backbone
```

</details>

<details open>
  <summary>Sample script to run <b>ASGD(Softplus)</b> on CIFAR-100</summary>

```bash

python -u main.py \
  --model resnet18 --dataset cifar100 --Lambda 0.1 \
  --loss_fn softplus --lr_dual 1e-4 --rho 1e-7 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 1e-4 \
  --momentum 0.0 --pretrained ./checkpoints/resnet18_cifar100.pth \
  --freeze_backbone
```

</details>

<details open>
  <summary>Sample script to run <b>U-MAX</b> on CIFAR-100</summary>

```bash

python -u main.py \
  --model resnet18 --dataset cifar100 --Lambda 0.1 \
  --loss_fn U_MAX --lr_dual 1e0 --delta 1e0 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 1e-3 \
  --momentum 0.0 --pretrained ./checkpoints/resnet18_cifar100.pth \
  --freeze_backbone
```

</details>

