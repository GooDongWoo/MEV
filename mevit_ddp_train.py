import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from mevit_model import MultiExitViT
from Dloaders import Dloaders

IMG_SIZE = 224
dataset_name = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100, 'imagenet': datasets.ImageNet}
dataset_outdim = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000}
batch_size = 1024
data_choice = 'imagenet'
unfreeze_ees_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
max_epochs = 200
start_lr = 1e-4
weight_decay = 1e-4
lr_decrease_factor = 0.6
lr_decrease_patience = 2
early_stop_patience = 10

# Distributed setup
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Trainer class remains largely unchanged but adapts to DDP
class Trainer:
    def __init__(self, model, params, rank):
        self.model = model
        self.num_epochs = params['num_epochs']
        self.loss_func = params["loss_func"]
        self.opt = params["optimizer"]
        self.train_dl = params["train_dl"]
        self.val_dl = params["val_dl"]
        self.lr_scheduler = params["lr_scheduler"]
        self.device = rank
        self.model = DDP(self.model, device_ids=[rank])

    # Loss and accuracy calculation remains the same
    ...

    def train(self):
        """Train the model."""
        for epoch in range(self.num_epochs):
            # Train phase
            self.model.train()
            for xb, yb in tqdm(self.train_dl, desc=f"Train Epoch {epoch}"):
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.opt.zero_grad()
                output = self.model(xb)
                loss = self.loss_func(output, yb)
                loss.backward()
                self.opt.step()

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for xb, yb in tqdm(self.val_dl, desc=f"Validation Epoch {epoch}"):
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    output = self.model(xb)
                    val_loss += self.loss_func(output, yb).item()
            
            print(f"Epoch {epoch}: Validation Loss {val_loss / len(self.val_dl)}")

def train_ddp(rank, world_size):
    setup(rank, world_size)

    # Prepare dataset with DistributedSampler
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = dataset_name[data_choice](root='./data', train=True, download=True, transform=transform)
    val_dataset = dataset_name[data_choice](root='./data', train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # Model setup
    pretrained_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    pretrained_vit.heads.head = nn.Linear(pretrained_vit.heads.head.in_features, dataset_outdim[data_choice])

    model = MultiExitViT(pretrained_vit, num_classes=dataset_outdim[data_choice])
    model = model.to(rank)

    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decrease_factor, patience=lr_decrease_patience)

    # Trainer instance
    params = {'num_epochs': max_epochs, 'loss_func': criterion, 'optimizer': optimizer, 'train_dl': train_loader,
              'val_dl': val_loader, 'lr_scheduler': lr_scheduler}
    trainer = Trainer(model, params, rank)
    trainer.train()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
