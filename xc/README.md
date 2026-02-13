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

### Training

<details open>
    <summary>Sample script to run <b>SCENT</b> on Glint360K</summary>

```bash
#!/bin/bash

python -u train.py \
  --algorithm scent \
  --data-dir 'features/glint360k/' \
  --data-size 17091657 \
  --epochs 50 \
  --gamma 12.0 \
  --lr 5.0 \
  --name glint360k_scent \
  --num-classes 360232 \
```

</details>
