from contextlib import nullcontext

import torch
import torch.amp
import torch.utils.data

from dataset import get_train_data
from torch import nn
from tqdm.auto import tqdm

from unet import Unet


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = False,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        amp_context_maybe = torch.amp.autocast(device.type) if amp else nullcontext()
        with amp_context_maybe:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # TODO: your code for loss scaling here

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(
            f"Loss: {round(loss.item(), 4)} "
            f"Accuracy: {round(accuracy.item() * 100, 4)}"
        )


def train():
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, amp=False)


if __name__ == "__main__":
    train()
