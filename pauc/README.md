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
#!/bin/bash

python -u main.py \
  --model resnet18 --dataset cifar100 --Lambda 0.1 \
  --loss_fn SCENT --alpha_t 4 --scheduler cosine \
  --batch_size 64 --total_epochs 60 --lr 1e-3 \
  --momentum 0.0 --pretrained dir/to/pretrained \
  --freeze_backbone
```

</details>
