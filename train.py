BATCH_SIZE = 512

import pytorch_lightning as pl
import torch
from torch import nn
import torchvision 
from torchvision import transforms
from efficientnet import EfficientNetB0
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint
import os

transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_ds = torchvision.datasets.CIFAR10("./datasets/", train=True, transform=inference_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)

val_ds = torchvision.datasets.CIFAR10("./datasets/", train=False, transform=inference_transform, download=True)
val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=False)


class Lightning(pl.LightningModule):
    def __init__(self, hparams, beta):
        super().__init__()
        self.model = EfficientNetB0(jacobian_dropout=hparams.jacobian_dropout, dropout_rate=hparams.dropout_rate, beta=beta)

    def forward(self, x):
        return self.model(x)
    
    def loss_fn(self, y_pred, y_true):
        return torch.nn.functional.cross_entropy(y_pred, y_true)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        tensorboard_logs = {'loss': loss, 'train_acc': torch.tensor(accuracy_score(y.cpu(), out.detach().argmax(dim=1).cpu()))}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        return {'val_loss': loss, 'val_acc': torch.tensor(accuracy_score(y.cpu(), out.argmax(dim=1).cpu()))}
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-2)
    

def get_next_version(dir, search_string):
    folders = os.listdir(dir)
    versions = [int(item.split("_")[-1]) for item in folders if item.find(search_string) != -1]
    if len(versions) == 0:
        return 0
    latest_version = max(versions)
    return latest_version + 1

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--jacobian_dropout', type=int, default=0)


    args = parser.parse_args()
    #print(args.jacobian_dropout, type(args.jacobian_dropout))

    beta = 0.2#BATCH_SIZE / len(train_ds)
    
    lightning = Lightning(args, beta)
        
    checkpoint_callback = ModelCheckpoint(
                                    monitor='val_acc',
    )

    dropout = f"jacobian_beta_{beta}" if args.jacobian_dropout else "vanilla"
    version = get_next_version("./logs", f"{dropout}_{args.dropout_rate}")

    trainer = pl.Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=100, 
        #fast_dev_run=True, 
        logger=pl.loggers.TensorBoardLogger(
                    save_dir="./", 
                    name="logs", 
                    version=f"{dropout}_{args.dropout_rate}_{version}"
                    )
        )
        
    trainer.add_argparse_args(parser)

    trainer.fit(lightning, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    