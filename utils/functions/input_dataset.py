import numpy as np
from torch.utils.data import Dataset as TorchDataset, Sampler
from torch_geometric.loader import DataLoader


class InputDataset(TorchDataset):
    """Wrap a pandas DataFrame so each row returns a PyG graph sample."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = np.array(dataset["target"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index].input
        data.func = self.dataset.iloc[index].func
        return data

    def get_loader(self, batch_size, shuffle=True):
        if shuffle:
            sampler = BalancedBatchSampler(self.labels, batch_size)
            return DataLoader(dataset=self, batch_sampler=sampler)
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=False)


class BalancedBatchSampler(Sampler):
    """Create mixed batches that always include positive samples when possible."""

    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.positive_indices = np.where(labels == 1)[0]
        self.negative_indices = np.where(labels == 0)[0]
        self.num_batches = max(1, len(self.labels) // batch_size)

    def __iter__(self):
        positive_per_batch = max(1, len(self.positive_indices) // self.num_batches)
        negative_per_batch = max(1, self.batch_size - positive_per_batch)

        positives = np.random.permutation(self.positive_indices)
        negatives = np.random.permutation(self.negative_indices)

        batches = []
        for i in range(self.num_batches):
            pos_batch = positives[i * positive_per_batch:(i + 1) * positive_per_batch]
            neg_batch = negatives[i * negative_per_batch:(i + 1) * negative_per_batch]
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        return self.num_batches
