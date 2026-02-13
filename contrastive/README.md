# Contrastive Learning

## Getting Started

### Environment Setup

We will use the code of [DRRho-CLIP](https://github.com/Optimization-AI/DRRho-CLIP), please follow the instructions in their README to set up the environment.

### Training

To apply the implementation of different algorithms, we need to apply the provided `scent.patch`. Run the following command in the DRRho-CLIP codebase:
```bash
git apply path/to/scent.patch
```

We present sample slurm scripts to run SCENT and FastCLIP on DFN-14M.

<details open>
    <summary>Sample script to run <b>SCENT</b> on DFN-14M using 4 GPUs</summary>

```bash
#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=scent
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate fastclip

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805

export CUDA_VISIBLE_DEVICES='0,1,2,3'
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/dfn2b/medium/shards/0000{0000..1926}.tar' \
    --train-num-samples 13710637 --data_size 19270000 \
    --warmup 500 \
    --batch-size 1024 \
    --epochs 24 \
    --workers 6 \
    --model ViT-B-32 \
    --name scent \
    --seed 2026 \
    --wd 0.2 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_learnable --temperature 0.07 \
    --lr 5e-4 --lr_tau 6.25e-5 --lr_tau_scheduler step_thresh --rho 11.0 --fastclip_eps 1e-6 \
    --gamma 10.0 --gamma_schedule constant --gamma_decay_epochs 24 \
    --nu_update scent
```

</details>

<details>
    <summary>Sample script to run <b>FastCLIP</b> on DFN-14M using 4 GPUs</summary>

```bash
#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=fastclip
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate fastclip

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805

export CUDA_VISIBLE_DEVICES='0,1,2,3'
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/dfn2b/medium/shards/0000{0000..1926}.tar' \
    --train-num-samples 13710637 --data_size 19270000 \
    --warmup 500 \
    --batch-size 1024 \
    --epochs 24 \
    --workers 6 \
    --model ViT-B-32 \
    --name fastclip \
    --seed 2026 \
    --wd 0.2 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_learnable --temperature 0.07 \
    --lr 5e-4 --lr_tau 6.25e-5 --lr_tau_scheduler step_thresh --rho 11.0 --fastclip_eps 1e-6 \
    --gamma 10.0 --gamma_schedule constant --gamma_decay_epochs 24 \
    --nu_update sox
```

</details>
