"""contrastive learning for getting better representation"""


from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def contrastive_learning_loop(
    device: torch.device,
    model: nn.Module,
    loss_fn: _Loss,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    enable_amp=False,
) -> Tuple[nn.Module, float]:
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    model.train()
    train_loss_sum = 0
    train_correct_pred = 0
    num_train_data = len(train_dataloader.dataset)  # type: ignore  # map-style Dataset has __len__()
    num_batches_train = len(train_dataloader)
    for data, target in train_dataloader:
        # data transport
        if device != torch.device("cpu"):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

        for param in model.parameters():  # fast zero_grad
            param.grad = None

        with torch.cuda.amp.autocast(enabled=enable_amp):
            prediction = model(data)
            loss = loss_fn(prediction, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss_sum += loss.item()
        train_correct_pred += (prediction.argmax(1) == target).type(torch.float).sum().item()
    train_loss_avg = train_loss_sum / num_batches_train
    train_accuracy = train_correct_pred / num_train_data

    loss: float = 0

    return model, loss
