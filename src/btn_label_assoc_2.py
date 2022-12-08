import os
from glob import glob
import pickle

import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from termcolor import cprint
from PIL import Image


class BTNLabelAssocSet(Dataset):
    def __init__(self, buildings) -> None:
        self.data = []
        for b in buildings:
            self.data.extend(glob(os.path.join(b, "**/*.pkl")))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        d = pickle.load(open(path, "rb"))
        img = d["img"][:, :, ::-1]
        img = Image.fromarray(img)
        t = transforms.Compose([Resize((224, 224)), ToTensor()])
        img = t(img)
        gt = torch.tensor(d["gt"]).type(torch.float32)
        return img, gt


class BTNLabelAssoc2(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = resnet18(progress=True)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

        self.lr = lr

    def training_step(self, batch, batch_idx):
        img, gt = batch
        outputs = self.model(img)
        loss = F.mse_loss(outputs, gt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch
        outputs = self.model(img)
        loss = F.mse_loss(outputs, gt)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer


def main():
    torch.autograd.set_detect_anomaly(True)
    buildings = glob("/home/abhinavchadaga/cs/fri_II/final_project/datasets/association_set/*")
    train_b = ["mixed", "inspire_on_22nd", "ruckus_on_rio", "signature", "ahg"]
    test_b = ["mixed_val", "gdc", "bergstrom", "burdine"]
    train_buildings = []
    test_buildings = []
    for b in buildings:
        for x in test_b:
            if x in b:
                test_buildings.append(b)
    for b in buildings:
        for x in train_b:
            if x in b:
                train_buildings.append(b)

    cprint("train buildings:", "white", attrs=["bold"])
    for b in train_buildings:
        print(b)

    cprint("test buildings:", "white", attrs=["bold"])
    for b in test_buildings:
        print(b)

    train_set = BTNLabelAssocSet(buildings=train_buildings)
    val_set = BTNLabelAssocSet(buildings=test_buildings)

    trainloader = DataLoader(
        dataset=train_set,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )

    imgs, gt = next(iter(trainloader))
    print(imgs.shape)

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.001, patience=10, verbose=True, mode="min"
    )

    model = BTNLabelAssoc2()
    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=10,
        accelerator="gpu",
        log_every_n_steps=5,
        val_check_interval=10,
        callbacks=[early_stop_callback],
    )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
