import argparse
import math
import glob
import os
import csv

from tqdm import tqdm
import braceexpand
import torch
import webdataset as wds
from torchvision import transforms

from data.webdata import WebDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, choices=["glint360k", "treeoflife10m"], help="Dataset name.")
    p.add_argument("--data-url", type=str, required=True, help="URL to tar files containing images and labels.")
    p.add_argument("--pretrained-path", type=str, help="Path to pretrained model checkpoint.")
    p.add_argument("--metadata-path", type=str, help="Path to metadata CSV file (required for treeoflife10m).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="features")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=10000, help="Number of samples in the dataset.")
    p.add_argument("--feature-dim", type=int, default=512)
    p.add_argument("--data-size", type=int, default=18000000, help="Maximum number of samples in the dataset.")
    return p.parse_args()


def expand_urls(urls, weights=None):
    if weights is None:
        wds_expanded_urls = wds.shardlists.expand_urls(urls)
        # wds.shardlists.expand_urls leverages braceexpand, which does not support wildcards
        expanded_urls = []
        for url in wds_expanded_urls:
            if '*' in url:
                expanded_urls.extend(glob.glob(url))
            else:
                expanded_urls.append(url)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def extract_data_index(data_url: str):
    index = -1
    try:
        index = int(data_url.split('/')[-1].split('.')[0])
    except:
        pass
    return index


def build_dataloaders(args):
    if args.data == "glint360k":
        # https://github.com/deepinsight/insightface/blob/fa9afd095badf5ed1201b59384509e8b145843e5/recognition/partial_fc/pytorch/dataset.py#L75
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.data == "treeoflife10m":
        # https://github.com/Imageomics/bioclip/blob/4ea32471c550a1c42f21ff1c2d08e2064f8ac048/src/training/params.py#L232
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unsupported dataset: {args.data}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    shards_list = expand_urls(args.data_url)[0]
    shards_list.sort(key=extract_data_index)
    dataset = WebDataset(args.data, shards_list, is_train=False, batch_size=args.batch_size,
                         preprocess_img=transform, seed=args.seed)
    num_batches = math.ceil(args.num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = args.num_samples

    return dataloader


def extract_feat_glint360k(args, dataloader, device, features_all, labels_all, is_cached_all):
    from model.iresnet import iresnet50
    model = iresnet50()
    model = model.to(device)
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=dataloader.num_batches)
        for images, labels, keys in pbar:
            images = images.to(device)
            features = model(images)
            features_all[keys] = features.to("cpu", dtype=torch.float)
            labels_all[keys] = labels.to("cpu", dtype=torch.int)
            is_cached_all[keys] = True
            del images, features, labels, keys

    return features_all, labels_all, is_cached_all


def extract_feat_treeoflife10m(args, dataloader, device, features_all, labels_all, is_cached_all):
    import open_clip

    def _get_key_idx_label_map(metadata_path: str):
        key_idx_map = {}
        key_label_map = {}
        key_list = []
        species_list = []
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key_list.append(row["treeoflife_id"])
                species_list.append(row["species"])
        for idx, key in enumerate(key_list):
            key_idx_map[key] = idx
        unique_species = sorted(list(set(species_list)))
        species_to_int = {species: i for i, species in enumerate(unique_species)}
        for key, species in zip(key_list, species_list):
            key_label_map[key] = species_to_int[species]
        return key_idx_map, key_label_map

    model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    model = model.to(device)
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    key_idx_map, key_label_map = _get_key_idx_label_map(args.metadata_path)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=dataloader.num_batches)
        for images, keys in pbar:
            images = images.to(device)
            features = model.encode_image(images, normalize=True)
            indices = torch.tensor([key_idx_map[key] for key in keys], dtype=torch.long)
            labels = torch.tensor([key_label_map[key] for key in keys], dtype=torch.int)
            features_all[indices] = features.to("cpu", dtype=torch.float)
            labels_all[indices] = labels
            is_cached_all[indices] = True
            del images, features, labels, indices

    return features_all, labels_all, is_cached_all


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    dataloader = build_dataloaders(args)

    features_all = torch.zeros(args.data_size, args.feature_dim)
    labels_all = torch.zeros(args.data_size, dtype=torch.int)
    is_cached_all = torch.zeros(args.data_size, dtype=torch.bool)
    if os.path.exists(os.path.join(args.out_dir, "features.pt")):
        features_all = torch.load(os.path.join(args.out_dir, "features.pt"))
    if os.path.exists(os.path.join(args.out_dir, "labels.pt")):
        labels_all = torch.load(os.path.join(args.out_dir, "labels.pt"))
    if os.path.exists(os.path.join(args.out_dir, "is_cached.pt")):
        is_cached_all = torch.load(os.path.join(args.out_dir, "is_cached.pt"))

    if args.data == "glint360k":
        features_all, labels_all, is_cached_all = extract_feat_glint360k(
            args, dataloader, device, features_all, labels_all, is_cached_all)
    elif args.data == "treeoflife10m":
        features_all, labels_all, is_cached_all = extract_feat_treeoflife10m(
            args, dataloader, device, features_all, labels_all, is_cached_all)
    else:
        raise ValueError(f"Unsupported dataset: {args.data}")

    features_path = os.path.join(args.out_dir, "features.pt")
    labels_path = os.path.join(args.out_dir, "labels.pt")
    is_cached_path = os.path.join(args.out_dir, "is_cached.pt")
    torch.save(features_all, features_path)
    torch.save(labels_all, labels_path)
    torch.save(is_cached_all, is_cached_path)


if __name__ == "__main__":
    main()
