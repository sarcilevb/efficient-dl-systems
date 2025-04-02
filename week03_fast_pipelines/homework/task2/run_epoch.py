from enum import Enum

import hydra

import torch
from dataset import DataMode, get_dataloader, TOKENIZER_NAME
from omegaconf import DictConfig
from tqdm import tqdm
from transformer import TransformerModel
from transformers import AutoTokenizer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_epoch(config: DictConfig) -> None:
    # for vocab size
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # create model from config
    model = get_gpt2_model(config.model, tokenizer.vocab_size)
    model.to(config.model.device)

    # create dataloader from input params
    data_config = config.data
    dataloader = get_dataloader(
        data_path=data_config.path,
        data_mode=DataMode[config.loader.mode],
        batch_size=data_config.batch_size,
        ubb_bin_width=(
            config.loader.bin_width
            if DataMode[config.loader.mode] == DataMode.ULTRA_BIG_BRAIN
            else None
        ),
    )

    # run epoch
    model.train()
    for i, (sequence_batch, attention_mask_batch) in enumerate(tqdm(dataloader)):
        sequence_batch = sequence_batch.to(config.model.device)
        attention_mask_batch = attention_mask_batch.to(config.model.device)
        model(sequence_batch, src_key_padding_mask=attention_mask_batch)


def get_gpt2_model(model_config: DictConfig, vocab_size: int) -> torch.nn.Module:
    return TransformerModel(
        ntoken=vocab_size,
        d_model=model_config.embedding_dim,
        nhead=model_config.num_heads,
        d_hid=model_config.embedding_dim,
        nlayers=model_config.num_encoder_layers,
    )


if __name__ == "__main__":
    run_epoch()
