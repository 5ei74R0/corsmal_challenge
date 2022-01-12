"""
supervised contrastive learning for getting better representation
https://arxiv.org/abs/2004.11362

contrastive pretraining of CLIP
https://openai.com/blog/clip/
"""
# path resolving
import sys
from pathlib import Path

current_dir: Path = Path().cwd().resolve()
project_root: Path = current_dir.parent
data_dir: Path = project_root / "data" / "train"

sys.path.append(str(project_root))


# main
from typing import Dict, List, Tuple  # noqa (E402)

import matplotlib.pyplot as plt
import torch  # noqa (E402)
from torch import nn  # noqa (E402)
from torch.nn.modules.loss import _Loss  # noqa (E402)
from torch.optim import Optimizer  # noqa (E402)
from torch.utils.data import DataLoader  # noqa (E402)

from corsmal_challenge.data.data_loader import (  # noqa (E402)
    ReproducibleDataLoader as DataLoader,
)
from corsmal_challenge.data.dataset import AudioDataset  # noqa (E402)
from corsmal_challenge.models.audio import LogMelEncoder  # noqa (E402)
from corsmal_challenge.models.task1_2 import T1Head, T2Head  # noqa (E402)
from corsmal_challenge.utils import fix_random_seeds  # noqa (E402)


def validation_loop(
    device: torch.device,
    model: nn.Module,
    loss_fn: _Loss,
    validation_dataloader: DataLoader,
) -> Tuple[nn.Module, Dict]:
    model.eval()
    validation_loss_sum = 0
    validation_correct_pred = 0
    num_validation_data = len(validation_dataloader.dataset)  # type: ignore  # map-style Dataset has __len__()
    num_batches = len(validation_dataloader)
    with torch.no_grad():
        for data, target in validation_dataloader:
            # data transport
            if device != torch.device("cpu"):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                prediction = model(data)
                loss = loss_fn(prediction, target)

            validation_loss_sum += loss.item()
            validation_correct_pred += (prediction.argmax(1) == target).type(torch.float).sum().item()
    val_loss_avg = validation_loss_sum / num_batches
    val_accuracy = validation_correct_pred / num_validation_data

    return model, {
        "val loss": val_loss_avg,
        "val accuracy": val_accuracy,
    }


class TaskChallenger2(nn.Module):
    def __init__(self, task_id: int = 1):
        super(TaskChallenger2, self).__init__()
        self.task_id = task_id
        self.encoder = LogMelEncoder(num_encoder_blocks=4, num_heads=4)
        self.classify_head1 = T1Head()
        self.classify_head2 = T2Head()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.encoder(inputs)
        if self.task_id == 1:
            return self.classify_head1(x[:, 0, :].squeeze(1))  # extract embedding of class token
        elif self.task_id == 2:
            # return self.classify_head2(x[:, 0, :].squeeze(1))  # extract embedding of class token
            return self.classify_head2(x[:, -1, :].squeeze(1))  # extract embedding of class token
        return x  # error


RAND_SEED = 0
EPOCH = 150

if __name__ == "__main__":
    fix_random_seeds(RAND_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskChallenger2()
    # model.load_state_dict(torch.load(current_dir / "20220111-result?.pt"))
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    train_dataset = AudioDataset(
        data_dir,
        data_dir / "ccm_train_annotation.json",
        seed=RAND_SEED,
        train=True,
    )
    val_dataset = AudioDataset(
        data_dir,
        data_dir / "ccm_train_annotation.json",
        seed=RAND_SEED,
        train=False,
    )
    train_dataloader = DataLoader(train_dataset, specified_seed=RAND_SEED, shuffle=True)
    val_dataloader = DataLoader(val_dataset, specified_seed=RAND_SEED, shuffle=True)

    train_loss_t1 = []
    val_loss_t1 = []
    train_loss_t2 = []
    val_loss_t2 = []

    for step in range(EPOCH):
        train_dataset.query = "type"
        val_dataset.query = "type"
        train_dataset.random_crop = step % 2 == 0
        train_dataset.strong_crop = True
        model.task_id = 2

        tup = validation_loop(  # type: ignore
            device,
            model,
            loss_fn,
            val_dataloader,
        )
        metrics = tup[1]
        train_loss_t1.append(metrics["train loss"])
        val_loss_t1.append(metrics["val loss"])
        print(metrics)

        train_dataset.query = "level"
        val_dataset.query = "level"
        train_dataset.random_crop = False
        train_dataset.strong_crop = False
        model.task_id = 1

        tup = validation_loop(  # type: ignore
            device,
            model,
            loss_fn,
            val_dataloader,
        )
        metrics = tup[1]
        train_loss_t2.append(metrics["train loss"])
        val_loss_t2.append(metrics["val loss"])
        print(metrics)

    plt.plot(train_loss_t1, label="train loss: t1")
    plt.plot(val_loss_t2, label="val loss: t1")
    plt.plot(train_loss_t2, label="train loss: t2")
    plt.plot(val_loss_t2, label="val loss: t2")
    plt.legend()
    plt.savefig(str(current_dir / "20220111-result2.png"))
    torch.save(model.state_dict(), current_dir / "20220111-result2.pt")

# def contrastive_learning_loop(
#     device: torch.device,
#     model: nn.Module,
#     loss_fn: _Loss,
#     optimizer: Optimizer,
#     dataloaders: List[DataLoader],
#     enable_amp=False,
# ) -> Tuple[nn.Module, float]:
#     scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
#     model.train()
#     train_loss_sum = 0
#     train_correct_pred = 0
#     for data_tuple in zip(dl for dl in dataloaders):
#         # data transport
#         for i in range(len(data_tuple)):
#             if device != torch.device("cpu"):
#                 data_tuple[i] = data_tuple[i].to(device, non_blocking=True)

#         for param in model.parameters():  # fast zero_grad
#             param.grad = None

#         with torch.cuda.amp.autocast(enabled=enable_amp):
#             prediction = model(data)
#             loss = loss_fn(prediction, target)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         train_loss_sum += loss.item()
#         train_correct_pred += (prediction.argmax(1) == target).type(torch.float).sum().item()
#         # train_loss_avg = train_loss_sum / num_batches_train
#         # train_accuracy = train_correct_pred / num_train_data

#     loss: float = 0

#     return model, loss
