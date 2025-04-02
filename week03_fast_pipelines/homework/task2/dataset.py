import os
import random
from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset as HFDataset, load_from_disk

from numpy import int32
from torch.utils.data import IterableDataset, Sampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, BatchEncoding
from transformers.pipelines import token_classification

TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH = 640
PAD_TOKEN_ID = 0


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def get_dataloader(
    data_path: str,
    data_mode: DataMode,
    batch_size: int,
    ubb_bin_width: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    dataset = BrainDataset(data_path, MAX_LENGTH)
    if data_mode in (DataMode.BRAIN, DataMode.BIG_BRAIN):
        effective_max_length = MAX_LENGTH if data_mode == DataMode.BRAIN else None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, effective_max_length),
        )
    if data_mode == DataMode.ULTRA_BIG_BRAIN:
        assert ubb_bin_width is not None
        batch_sampler = UltraBigBrainBatchSampler(
            token_seqs=dataset.token_seqs,
            batch_size=batch_size,
            bin_width=ubb_bin_width,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )
    raise NotImplementedError(f"Data mode {data_mode} is not implemented yet")


class BaseWikiTokenizedDataset(Dataset):
    def __init__(self, data_path: str, max_length: Optional[int] = None):
        tokenizer_output = read_and_tokenize_data(data_path, max_length)
        self.token_seqs: List[List[int]] = tokenizer_output["input_ids"]
        self.padding_masks: List[List[int]] = tokenizer_output["attention_mask"]


class BrainDataset(BaseWikiTokenizedDataset):
    def __init__(self, data_path: str, max_length: Optional[int] = None):
        super().__init__(data_path, max_length)

    def __getitem__(self, idx: int):
        return torch.tensor(self.token_seqs[idx], dtype=torch.int32), torch.tensor(
            self.padding_masks[idx], dtype=torch.bool
        )

    def __len__(self) -> int:
        return len(self.token_seqs)


# class BigBrainDataset(Dataset):
#     def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
#         pass

#     def __getitem__(self, idx: int):
#         pass


# class UltraBigBrainDataset(Dataset):
#     def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
#         pass

#     def __getitem__(self, idx: int):
#         pass


# class UltraDuperBigBrainDataset(IterableDataset):
#     def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
#         pass

#     def __iter__(self):
#         pass


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    max_length: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    sequences_list, mask_list = zip(*batch)
    batch_size = len(sequences_list)
    max_length = max_length or max(len(seq) for seq in sequences_list)

    output_tensors = []
    for i, lst in enumerate((sequences_list, mask_list)):
        pad_value = PAD_TOKEN_ID if i == 0 else False
        tensor = torch.full(
            size=(batch_size, max_length), fill_value=pad_value, dtype=lst[0].dtype
        )
        for idx, row_tensor in enumerate(lst):
            tensor[idx, : len(row_tensor)] = row_tensor
        output_tensors.append(tensor)
    return torch.transpose(output_tensors[0], 0, 1), output_tensors[1]


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, token_seqs: List[List[int]], batch_size: int, bin_width: int):
        idxs = list(range(len(token_seqs)))

        seq_lengths = [len(seq) for seq in token_seqs]
        min_seq_length = min(seq_lengths)
        bins = list(range(min_seq_length + bin_width, MAX_LENGTH, bin_width))
        bin_idxs = np.digitize(seq_lengths, bins)

        idxs_by_bin = defaultdict(list)
        for i in idxs:
            idxs_by_bin[bin_idxs[i]].append(i)

        self.batches = []
        for _, v in idxs_by_bin.items():
            random.shuffle(v)
            batch_start_idxs = list(range(0, len(v), batch_size))
            for start_idx_inclusive, end_idx_exclusive in zip(
                batch_start_idxs, batch_start_idxs[1:] + [len(v)]
            ):
                if end_idx_exclusive <= start_idx_inclusive:
                    print(start_idx_inclusive, end_idx_exclusive)
                self.batches.append(v[start_idx_inclusive:end_idx_exclusive])
        random.shuffle(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)


def read_and_tokenize_data(data_dir: str, max_length: Optional[int] = None):
    class Paths:
        TRAIN_PATHS = ("train-00000-of-00002.txt", "train-00001-of-00002.txt")
        TOKENIZED_CACHED_DATASET_PATH = "train_tokenized"

    tokenized_dataset_path = os.path.join(data_dir, Paths.TOKENIZED_CACHED_DATASET_PATH)
    try:
        dataset = (
            load_from_disk(tokenized_dataset_path).to_pandas().to_dict(orient="list")
        )
        return dataset

    except Exception:
        print("No cached tokenized dataset found. Tokenizing data...")
        raw_lines = []
        train0_path = os.path.join(data_dir, "train-00000-of-00002.txt")
        train1_path = os.path.join(data_dir, "train-00001-of-00002.txt")
        for path in (train0_path, train1_path):
            with open(path, "r") as f:
                raw_lines.extend(f.readlines())

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        tokenized_data = tokenizer(
            raw_lines,
            truncation=True,
            padding="do_not_pad",
            max_length=max_length or MAX_LENGTH,
        )

        print("Tokenization done. Saving for future use")
        os.makedirs(tokenized_dataset_path, exist_ok=True)
        HFDataset.from_dict(tokenized_data).save_to_disk(tokenized_dataset_path)
        print("Saved dataset")

        return tokenized_data
