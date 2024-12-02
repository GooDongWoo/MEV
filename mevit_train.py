# utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,models
from tqdm import tqdm  # Importing tqdm for progress bar
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
#from torchsummary import summary as summary_
import os
from torch.utils.tensorboard import SummaryWriter
from mevit_model import MultiExitViT
from Dloaders import Dloaders
IMG_SIZE = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}
##############################################################
################ 0. Hyperparameters ##########################
unfreeze_ees_list=[8]
##############################################################
batch_size = 1024
data_choice='imagenet'
mevit_isload=False
mevit_pretrained_path=f'models/{data_choice}/{unfreeze_ees_list[0]}/best_model.pth'
max_epochs = 200  # Set your max epochs

backbone_path=f'models/{data_choice}/vit_{data_choice}_backbone.pth'
start_lr=1e-4
weight_decay=1e-4

ee_list=[0,1,2,3,4,5,6,7,8,9]#exit list ex) [0,1,2,3,4,5,6,7,8,9]
exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1]#exit마다 가중치

classifier_wise=True
unfreeze_ees=[] #unfreeze exit list ex) [0,1,2,3,4,5,6,7,8,9]

# Early stopping parameters
early_stop_patience = 8

lr_decrease_factor = 0.6
lr_decrease_patience = 2
##############################################################
# # 3. Training part
class Trainer:
    def __init__(self, model, params):
        self.model = model
        self.num_epochs = params['num_epochs'];self.loss_func = params["loss_func"]
        self.opt = params["optimizer"];self.train_dl = params["train_dl"];self.data_choice = params["data_choice"]
        self.val_dl = params["val_dl"];self.lr_scheduler = params["lr_scheduler"]
        self.isload = params["isload"];self.path_chckpnt = params["path_chckpnt"]
        self.classifier_wise = params["classifier_wise"];self.unfreeze_ees = params["unfreeze_ees"]
        self.best_loss = float('inf');self.early_stop_patience = params["early_stop_patience"]
        self.old_epoch = 0
        self.device = next(model.parameters()).device

        # Initialize directory for model saving
        self.current_time = time.strftime('%m%d_%H%M%S', time.localtime())
        self.path = f'./models/{self.data_choice}/{self.unfreeze_ees[0]}'
        os.makedirs(self.path, exist_ok=True)

        # Setup TensorBoard writer
        self.writer = SummaryWriter(f'./runs/{self.data_choice}/{self.unfreeze_ees[0]}')

        # Load model checkpoint if required
        if self.isload:
            self._load_checkpoint()

        # Optionally freeze layers
        if self.classifier_wise:
            self._freeze_layers()

        # Save model specifications
        self._save_specifications()

    def _load_checkpoint(self):
        """Load model checkpoint and optimizer state."""
        checkpoint = torch.load(self.path_chckpnt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.old_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']

    def _freeze_layers(self):
        """Freeze layers based on classifier_wise and unfreeze_ees settings."""
        for param in self.model.parameters():
            param.requires_grad = False
        for idx in self.unfreeze_ees:
            for param in self.model.classifiers[idx].parameters():
                param.requires_grad = True
            for param in self.model.ees[idx].parameters():
                param.requires_grad = True

    def _save_specifications(self):
        """Save model training specifications to a text file."""
        spec_txt = (
            f'opt: {self.opt.__class__.__name__}\n'
            f'lr: {self.opt.param_groups[0]["lr"]}\n'
            f'batch: {self.train_dl.batch_size}\n'
            f'epoch: {self.num_epochs}\n'
            f'isload: {self.isload}\n'
            f'path_chckpnt: {self.path_chckpnt}\n'
            f'exits_loss_weights: {self.model.getELW()}\n'
        )
        with open(f"{self.path}/spec.txt", "w") as file:
            file.write(spec_txt)

    @staticmethod
    def get_lr(opt):
        """Retrieve current learning rate from optimizer."""
        for param_group in opt.param_groups:
            return param_group['lr']

    @staticmethod
    def metric_batch(output, label):
        """Calculate accuracy for a batch."""
        pred = output.argmax(1, keepdim=True)
        corrects = pred.eq(label.view_as(pred)).sum().item()
        return corrects

    def loss_batch(self, output_list, label, elws,mode):
        """Calculate loss and accuracy for a batch."""
        losses = [self.loss_func(output, label) * elw for output, elw in zip(output_list, elws)]
        accs = [self.metric_batch(output, label) for output in output_list]
        if mode=="train":
            self.opt.zero_grad()
            cnter=1
            tot_train=len(self.unfreeze_ees)
            for idx,loss in enumerate(losses):
                if idx in self.unfreeze_ees:
                    if (cnter<tot_train):
                        loss.backward(retain_graph=True)
                        cnter+=1
                    else:loss.backward()
            self.opt.step()
        return [loss.item() for loss in losses], accs

    def loss_epoch(self, data_loader, epoch, mode="train"):
        """Calculate loss and accuracy for an epoch."""
        running_loss = [0.0] * self.model.exit_num
        running_metric = [0.0] * self.model.exit_num
        len_data = len(data_loader.dataset)
        elws = self.model.getELW()

        with tqdm(data_loader, desc=f"{mode}: {epoch}th Epoch", unit="batch", leave=False) as t:
            for xb, yb in t:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output_list = self.model(xb)
                losses, accs = self.loss_batch(output_list, yb, elws,mode)

                running_loss = [sum(x) for x in zip(running_loss, losses)]
                running_metric = [sum(x) for x in zip(running_metric, accs)]
                t.set_postfix(accuracy=[round(100 * acc / len(xb),3) for acc in accs])

        running_loss = [loss / len_data for loss in running_loss]
        running_acc = [100 * metric / len_data for metric in running_metric]

        # TensorBoard logging
        loss_dict = {f'exit{idx}': loss for idx, loss in enumerate(running_loss)}
        acc_dict = {f'exit{idx}': acc for idx, acc in enumerate(running_acc)}
        self.writer.add_scalars(f'{mode}/loss', loss_dict, epoch)
        self.writer.add_scalars(f'{mode}/acc', acc_dict, epoch)
        self.writer.add_scalar(f'{mode}/loss_total_sum', sum(running_loss), epoch)

        return sum(running_loss), running_acc

    def train(self):
        """Train the model."""
        start_time = time.time()
        early_stop_cnter=0

        for epoch in range(self.old_epoch, self.old_epoch + self.num_epochs):
            print(f'Epoch {epoch}/{self.old_epoch + self.num_epochs - 1}, lr={self.get_lr(self.opt)}')

            # Train phase
            self.model.train()
            train_loss, train_accs = self.loss_epoch(self.train_dl, epoch, mode="train")

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_loss, val_accs = self.loss_epoch(self.val_dl, epoch, mode="val")

            # Save best model
            if val_loss < self.best_loss:
                early_stop_cnter=0
                self.best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'loss': val_loss,
                }, f'{self.path}/best_model.pth')
                print("Saved best model weights!")
            else:
                early_stop_cnter+=1
                print(f"No improvement in validation accuracy! cnter: {early_stop_cnter}")
                if early_stop_cnter>=self.early_stop_patience:
                    print("Early stopping!");print('@' * 50);break

            self.lr_scheduler.step(val_loss)

            # Logging
            elapsed_time = (time.time() - start_time) / 60
            hours, minutes = divmod(elapsed_time, 60)
            print(f'train_loss: {train_loss:.6f}, train_acc: {train_accs}')
            print(f'val_loss: {val_loss:.6f}, val_acc: {val_accs}, time: {int(hours)}h {int(minutes)}m')
            print('-' * 50)

        # Save final checkpoint
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss': val_loss,
        }, f'{self.path}/final_model.pth')

        self.writer.close()

        # Save final training summary
        with open(f"{self.path}/spec.txt", "a") as file:
            file.write(f"final_val_acc: {val_accs}\nfinal_train_acc: {train_accs}\n")
##############################################################
if __name__ == '__main__':
    dloaders=Dloaders(data_choice=data_choice,batch_size=batch_size,IMG_SIZE=IMG_SIZE)
    train_loader,test_loader = dloaders.get_loaders()

    # Load the pretrained ViT model from the saved file
    pretrained_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    
    if data_choice != 'imagenet':
        pretrained_vit.heads.head = nn.Linear(pretrained_vit.heads.head.in_features, dataset_outdim[data_choice])  # Ensure output matches the number of classes

        # Load model weights
        pretrained_vit.load_state_dict(torch.load(backbone_path))
        pretrained_vit = pretrained_vit.to(device)
    #from torchinfo import summary
    #summary(pretrained_vit,input_size= (64, 3, IMG_SIZE, IMG_SIZE))

    for i in unfreeze_ees_list:
        model = MultiExitViT(pretrained_vit,num_classes=dataset_outdim[data_choice],ee_list=ee_list,exit_loss_weights=exit_loss_weights).to(device)
        optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        lr_scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=lr_decrease_factor, patience=lr_decrease_patience, verbose=True)
        params={'num_epochs':max_epochs, 'loss_func':criterion, 'optimizer':optimizer, 'data_choice':data_choice,
            'train_dl':train_loader, 'val_dl':test_loader, 'lr_scheduler':lr_scheduler, 
            'isload':mevit_isload, 'path_chckpnt':mevit_pretrained_path,'classifier_wise':classifier_wise,
            'unfreeze_ees':unfreeze_ees,'early_stop_patience':early_stop_patience}
        params["unfreeze_ees"]=[i]
        t1=Trainer(model=model, params=params)
        print(f'unfreeze_ees: {params["unfreeze_ees"]}')
        t1.train()