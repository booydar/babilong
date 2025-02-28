import math
from typing import List, Union, Optional

import torch
import numpy as np

from lm_experiments_tools.utils import get_distributed_rank


class MixtureDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: List[torch.utils.data.Dataset],
                 weights: Optional[List[Union[float, int]]] = None) -> None:
        """MixtureDataset takes each dataset from datasets list with its weight.

        datasets = [d1, d2, d3]
        weights = [1.0, 2.0, 0.5]
        -> len(MixtureDataset) = 1.0 * len(d1) + 2.0 * len(d2) + 0.5 * len(d3)
           MixtureDataset = d1.sample(1.0 * len(d1)) + d2.sample(2.0 * len(d2)) + d3.sample(0.5 * len(d3)),
           where d.sample(n) takes n samples from dataset d

        MixtureDataset is similar to megatron.data.blendable_dataset.BlendableDataset, but has different
        blending/mixturing logic:
            len(BlendableDataset) = len(d1) + len(d2) + len(d3) = len(d)
            weights /= sum(weights)
            BlendableDataset =  d1.sample(w1 * len(d)) + d2.sample(w2 * len(d)) + d3.sample(w3 * len(d)),
            resulting in under-/up-sampling from d1, d2, d3.

        Args:
            datasets (List[torch.utils.data.Dataset]): list of torch Datasets
            weights (Optional[List[Union[float, int]]], optional): weights of datasets, if weights is None
                than we just merge all datasets into one. Defaults to None.
        """
        if weights is None:
            weights = [1.0] * len(datasets)

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = 0
        self.num_samples = []
        for w, dataset in zip(weights, self.datasets):
            self.num_samples += [math.ceil(w * len(dataset))]
        self.size = np.sum(self.num_samples)

        weights = np.array(self.num_samples, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indecies.
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        from megatron.data import helpers
        helpers.build_blending_indices(self.dataset_index,
                                       self.dataset_sample_index,
                                       weights, num_datasets, self.size,
                                       get_distributed_rank() == 0)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return self.datasets[dataset_idx][sample_idx]
