import copy
import os
import random
from os import listdir
from os.path import isfile, join

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from ..functions.input_dataset import InputDataset


def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]


def load(path, pickle_file, ratio=1):
    full_path = os.path.join(path, pickle_file)
    dataset = pd.read_pickle(full_path)
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)
    return dataset


def loads(data_sets_dir, ratio=1):
    data_sets_files = sorted([f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f)) and f.endswith(".pkl")])
    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)
    if not data_sets_files:
        raise FileNotFoundError(f"No .pkl files were found under {data_sets_dir}.")

    dataset = load(data_sets_dir + os.sep, data_sets_files[0])
    for ds_file in data_sets_files[1:]:
        dataset = pd.concat([dataset, load(data_sets_dir + os.sep, ds_file)], ignore_index=True)
    return dataset.reset_index(drop=True)


def train_val_test_split(data_frame: pd.DataFrame, shuffle=True, no_balance=False):
    print("=== Splitting dataset ===")
    print(f"=== Total samples: {len(data_frame)} ===")

    negative = data_frame[data_frame.target == 0]
    positive = data_frame[data_frame.target == 1]

    ratio = len(negative) / max(1, len(positive))

    if not no_balance and ratio > 3.0 and len(positive) > 0:
        multiplier = min(int(ratio / 3), 5)
        oversampled_positive = positive.copy()
        for _ in range(multiplier - 1):
            sample_idx = torch.randint(low=0, high=len(positive), size=(min(len(positive), max(1, len(negative) // multiplier)),))
            sample = positive.iloc[sample_idx.tolist()].copy()
            oversampled_positive = pd.concat([oversampled_positive, sample], ignore_index=True)
        positive = oversampled_positive

    train_neg, test_neg = train_test_split(negative, test_size=0.2, shuffle=shuffle, random_state=42)
    test_neg, val_neg = train_test_split(test_neg, test_size=0.5, shuffle=shuffle, random_state=42)
    train_pos, test_pos = train_test_split(positive, test_size=0.2, shuffle=shuffle, random_state=42)
    test_pos, val_pos = train_test_split(test_pos, test_size=0.5, shuffle=shuffle, random_state=42)

    train = pd.concat([train_neg, train_pos], ignore_index=True)
    val = pd.concat([val_neg, val_pos], ignore_index=True)
    test = pd.concat([test_neg, test_pos], ignore_index=True)

    print(f"=== Train size: {len(train)} ===")
    print(f"=== Validation size: {len(val)} ===")
    print(f"=== Test size: {len(test)} ===")

    return InputDataset(train), InputDataset(test), InputDataset(val)


def balance_dataset(input_dataset, multiplier=3):
    try:
        if multiplier is None or multiplier <= 0:
            return input_dataset
    except Exception:
        return input_dataset

    is_pyg_dataset = False
    if hasattr(input_dataset, "__getitem__") and hasattr(input_dataset, "__len__"):
        try:
            sample = input_dataset[0]
            if isinstance(sample, Data) or (hasattr(sample, "y") and hasattr(sample, "edge_index")):
                is_pyg_dataset = True
        except (IndexError, TypeError):
            pass

    if not is_pyg_dataset:
        return input_dataset

    positive_samples = []
    negative_samples = []
    for data in input_dataset:
        label = data.y.item() if hasattr(data.y, "item") else data.y
        if int(label) == 1:
            positive_samples.append(data)
        else:
            negative_samples.append(data)

    ratio = len(negative_samples) / max(1, len(positive_samples))
    if ratio <= 3.0 or len(positive_samples) <= 5:
        return input_dataset

    actual_multiplier = max(2, min(min(multiplier, int(ratio / 2)), 4))
    additional_samples = list(random.sample(positive_samples, min(len(positive_samples), len(positive_samples))))

    for _ in range(actual_multiplier - 1):
        samples_to_enhance = random.choices(
            positive_samples,
            k=min(len(positive_samples), max(1, len(negative_samples) // 3)),
        )
        for sample in samples_to_enhance:
            duplicate = copy.deepcopy(sample)
            if hasattr(duplicate, "x") and duplicate.x is not None and duplicate.x.dim() > 0:
                mask = torch.bernoulli(torch.ones_like(duplicate.x) * 0.2)
                noise = torch.randn_like(duplicate.x) * 0.02 * mask
                duplicate.x = duplicate.x + noise
                if hasattr(duplicate, "edge_index") and duplicate.edge_index is not None and duplicate.edge_index.size(1) > 10:
                    keep_prob = 0.95
                    edge_mask = torch.bernoulli(torch.ones(duplicate.edge_index.size(1)) * keep_prob).bool()
                    duplicate.edge_index = duplicate.edge_index[:, edge_mask]
            additional_samples.append(duplicate)

    all_positive = positive_samples + additional_samples
    target_positive_count = min(len(all_positive), int(len(negative_samples) * 0.6))
    if len(all_positive) > target_positive_count:
        all_positive = random.sample(all_positive, target_positive_count)

    balanced_dataset = negative_samples + all_positive
    random.shuffle(balanced_dataset)
    return balanced_dataset
