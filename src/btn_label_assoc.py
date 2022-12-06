import os
from glob import glob
import pickle

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from termcolor import cprint


class BTNLabelPairs(Dataset):
    def __init__(self, buildings) -> None:
        self.pkl_files = []
        for b in buildings:
            self.pkl_files.extend(glob(os.path.join(b, "**/*.pkl")))

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, index):
        path = self.pkl_files[index]
        with open(path, "rb") as f:
            data = pickle.load(f)

        input_vector = torch.tensor(data["input"], dtype=torch.float32)
        gt = torch.tensor(data["gt"], dtype=torch.int64)
        return input_vector, gt


class BTNLabelAssoc(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.LogSoftmax(dim=1),
        )

        self.lr = lr

    def training_step(self, batch, batch_idx):
        vector, gt = batch
        outputs = self.model(vector)
        loss = F.nll_loss(outputs, gt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vector, gt = batch
        outputs = self.model(vector)
        loss = F.nll_loss(outputs, gt)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer


def main():
    torch.autograd.set_detect_anomaly(True)
    buildings = glob("/home/abhinavchadaga/cs/fri_II/final_project/datasets/pairs/*")
    train_buildings = [b for b in buildings if any(["mixed" in b, "inspire" in b])]
    val_buildings = [b for b in buildings if any(["ahg" in b, "bergstrom" in b, "ruckus" in b])]
    cprint("train buildings:", "white", attrs=["bold"])
    for b in train_buildings:
        print(b)

    train_set = BTNLabelPairs(buildings=train_buildings)
    val_set = BTNLabelPairs(buildings=val_buildings)

    trainloader = DataLoader(
        dataset=train_set,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )

    model = BTNLabelAssoc()
    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=40,
        accelerator="gpu",
        log_every_n_steps=2,
        val_check_interval=2,
    )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
