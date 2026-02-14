# Extreme Classification

## Getting Started

### Environment Setup

To set up the environment for training, please install the required dependencies by running the following command:
  ```bash
  pip install -r requirements.txt
  ```

### Data Preparation

We select two image datasets and use the features extracted by pretrained models for training. To obtain the features, please follow the following instructions:
- Glint360K:
  1. Download the webdataset version of the Glint360K dataset from [here](https://huggingface.co/datasets/gaunernst/glint360k-wds-gz)
  2. Download the pretrained ResNet-50 model from [here](https://drive.google.com/drive/folders/16hjOGRJpwsJCRjIBbO13z3SrSgvPTaMV)
  3. Extract the features using the pretrained ResNet-50 model
    ```bash
    python extract_feat.py \
      --data 'glint360k' \
      --data-url 'path/to/glint360k-0{000..416}.tar.gz' \
      --batch-size 512 \
      --num-samples 18000000 \
      --data-size 17091657 \
      --out-dir 'dir/to/output' \
      --pretrained-path 'path/to/pretrained/16backbone_r50.pth'
    ```
- TreeOfLife-10M:
  1. Follow the instructions in 'Reproduce TreeOfLife-10M' section on [this page](https://github.com/Imageomics/bioclip/blob/main/docs/imageomics/treeoflife10m.md) to download the webdataset version of the TreeOfLife-10M dataset
  2. In the same environment as the previous step, extract the features using the pretrained CLIP model
    ```bash
    python extract_feat.py \
      --data 'treeoflife10m' \
      --data-url 'path/to/treeoflife10m-000{000..152}.tar' \
      --batch-size 512 \
      --num-samples 10000000 \
      --data-size 9533174 \
      --out-dir 'dir/to/output' \
      --metadata-path 'path/to/metadata/catalog.csv'
    ```
- After these steps, we obtain two tensor files under `dir/to/output`: `features.pt` and `labels.pt`. The `features.pt` file contains the extracted features, the `labels.pt` file contains the corresponding labels.
- Then we need to split the features and labels into training, validation and testing sets, which can be done by taking different rows from the two tensor files.
- Finally, store the splitted features and labels into `dir/to/features/{split}/features.pt` and `dir/to/features/{split}/labels.pt`, where `{split}` can be `train`, `val`, or `test`.

### Training

<details open>
    <summary>Sample script to run <b>SCENT</b> on Glint360K</summary>

In the command, gamma denotes the learning rate for the dual variable $\nu$, corresponding to $\alpha$ in the paper.
```bash
python -u train.py \
  --algorithm scent \
  --data-dir 'dir/to/features/' \
  --data-size 17091657 \
  --epochs 50 \
  --gamma 12.0 \
  --lr 5.0 \
  --name glint360k_scent \
  --num-classes 360232
```

</details>

<details>
    <summary>Sample script to run <b>BSGD</b> on Glint360K</summary>

```bash
python -u train.py \
  --algorithm bsgd \
  --data-dir 'dir/to/features/' \
  --data-size 17091657 \
  --epochs 50 \
  --lr 1.0 \
  --name glint360k_bsgd \
  --num-classes 360232
```

</details>

<details>
    <summary>Sample script to run <b>ASGD</b> on Glint360K</summary>

In the command, gamma denotes the initial learning rate for the dual variable $\nu$, corresponding to $\alpha$ in the paper.
```bash
python -u train.py \
  --algorithm asgd \
  --data-dir 'dir/to/features/' \
  --data-size 17091657 \
  --epochs 50 \
  --gamma 1.0 \
  --lr 0.5 \
  --name glint360k_asgd \
  --num-classes 360232
```

</details>

<details>
    <summary>Sample script to run <b>ASGD (Softplus)</b> on Glint360K</summary>

In the command, gamma denotes the initial learning rate for the dual variable $\nu$, corresponding to $\alpha$ in the paper.
```bash
python -u train.py \
  --algorithm softplus \
  --data-dir 'dir/to/features/' \
  --data-size 17091657 \
  --epochs 50 \
  --gamma 1.0 \
  --lr 0.5 \
  --name glint360k_softplus \
  --num-classes 360232 \
  --softplus-rho 1e-3
```

</details>

<details>
    <summary>Sample script to run <b>U-max</b> on Glint360K</summary>

In the command, gamma denotes the initial learning rate for the dual variable $\nu$, corresponding to $\alpha$ in the paper.
```bash
python -u train.py \
  --algorithm umax \
  --data-dir 'dir/to/features/' \
  --data-size 17091657 \
  --epochs 50 \
  --gamma 1.0 \
  --lr 0.5 \
  --name glint360k_umax \
  --num-classes 360232 \
  --umax-delta 1.0
```

</details>

<details>
    <summary>Sample script to run <b>SOX</b> on Glint360K</summary>

```bash
python -u train.py \
  --algorithm sox \
  --data-dir 'dir/to/features/' \
  --data-size 17091657 \
  --epochs 50 \
  --gamma 1e-5 \
  --lr 5.0 \
  --name glint360k_sox \
  --num-classes 360232
```

</details>
