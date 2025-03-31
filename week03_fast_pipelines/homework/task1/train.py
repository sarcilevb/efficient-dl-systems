from contextlib import nullcontext
from math import log
from typing import Dict, Iterator, Optional

import hydra
import torch
import torch.amp
import torch.utils.data
import wandb
from amp.scaler import DynamicScaler, GradientScaler, StaticScaler

from dataset import get_train_data
from matplotlib.artist import get
from matplotlib.pyplot import sca
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

from unet import Unet


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(config: DictConfig):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if config.wandb.enable:
        assert config.wandb.run_name is not None and config.wandb.run_name != ""
        wandb.init(
            project="efficient_dl_week3_hw_task1",
            name=config.wandb.run_name,
            config={str(k): v for k, v in OmegaConf.to_container(config).items()},
        )

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            config,
        )


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: DictConfig,
) -> None:
    model.train()

    scaler = get_scaler(optimizer, config)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        amp_context_maybe = (
            torch.amp.autocast(device.type) if config.amp.enable else nullcontext()
        )
        with amp_context_maybe:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaling_factor = None
        if scaler is not None:
            scaler.backward(loss)
            scaling_factor = scaler.scaling_factor
            scaler.step()
        else:
            loss.backward()
            optimizer.step()

        accuracy = (((outputs > 0.5) == labels).float().mean() * 100).item()
        pbar.set_description(
            f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy, 4)}"
        )
        if config.wandb.enable and i % 10 == 0:
            log_metrics_to_wandb(
                model, config.wandb, accuracy, loss.item(), scaling_factor
            )


def get_scaler(
    optimizer: torch.optim.Optimizer, config: DictConfig
) -> Optional[GradientScaler]:
    if not (config.amp.enable and config.amp.loss_scaling):
        return None

    if config.amp.scaler_type == "static":
        return StaticScaler(optimizer, config.amp.scaling_factor)

    if config.amp.scaler_type == "dynamic":
        return DynamicScaler(
            optimizer=optimizer,
            factor_upper_bound=config.amp.factor_upper_bound,
            factor_lower_bound=config.amp.factor_lower_bound,
            growth_factor=config.amp.growth_factor,
            backoff_factor=config.amp.backoff_factor,
            growth_interval=config.amp.growth_interval,
        )

    raise ValueError(f"Unknown scaler type: {config.amp.scaler_type}")


def log_metrics_to_wandb(
    model: torch.nn.Module,
    config: DictConfig,
    accuracy: float,
    loss: float,
    scaling_factor: Optional[float],
) -> None:
    wand_metrics = {}
    if config.log_gradient_stats:
        wand_metrics.update(calc_gradient_stats(model))
        if scaling_factor is not None:
            wand_metrics["gradient_stats_global/scaling_factor"] = scaling_factor
    if config.log_metrics:
        wand_metrics.update(
            {
                "train/loss": loss,
                "train/accuracy": accuracy,
            }
        )
    if len(wand_metrics) > 0:
        wandb.log(wand_metrics)


def calc_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    log_dict = {}
    global_min = 1e30
    global_max = -1e30
    global_zero_count = 0
    global_param_count = 0
    global_is_overflow = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            tensor_min = param.grad.data.abs().min().item()
            tensor_max = param.grad.data.abs().max().item()
            tensor_zero_count = (param.grad.data == 0).sum().item()
            tensor_param_count = param.grad.data.numel()
            overflow = (
                param.grad.data.isinf().any() or param.grad.data.isnan().any()
            ).item()

            log_dict[f"gradient_stats_per_param/{name}/grad_min"] = tensor_min
            log_dict[f"gradient_stats_per_param/{name}/grad_max"] = tensor_max
            log_dict[f"gradient_stats_per_param/{name}/grad_zero_rate"] = (
                tensor_zero_count / tensor_param_count
            )
            log_dict[f"gradient_stats_per_param/{name}/grad_overflow_present"] = int(
                overflow
            )

            global_min = min(global_min, tensor_min)
            global_max = max(global_max, tensor_max)
            global_zero_count += tensor_zero_count
            global_param_count += tensor_param_count
            global_is_overflow = global_is_overflow or overflow

    log_dict["gradient_stats_global/grad_min"] = global_min
    log_dict["gradient_stats_global/grad_max"] = global_max
    log_dict["gradient_stats_global/grad_zero_rate"] = (
        global_zero_count / global_param_count
    )
    log_dict["gradient_stats_global/grad_overflow_present"] = int(global_is_overflow)

    return log_dict


if __name__ == "__main__":
    train()
