# utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
from tqdm import tqdm  # Importing tqdm for progress bar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import time
#from torchsummary import summary as summary_
import os
from torch.utils.tensorboard import SummaryWriter

IMG_SIZE = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_name=dict()
dataset_name['cifar10']=datasets.CIFAR10
dataset_name['cifar100']=datasets.CIFAR100

dataset_outdim=dict()
dataset_outdim['cifar10']=10
dataset_outdim['cifar100']=100

##############################################################
################ 0. Hyperparameters ##########################
unfreeze_ees_list=[0,1,2,3,4,5,6,7,8,9]
##############################################################
batch_size = 56
data_choice='cifar100'
mevit_isload=False
mevit_pretrained_path=f'models/1108_103451/best_model.pth'
max_epochs = 100  # Set your max epochs

backbone_path=f'vit_{data_choice}_backbone.pth'
start_lr=1e-4
weight_decay=1e-4

ee_list=[0,1,2,3,4,5,6,7,8,9]#exit list ex) [0,1,2,3,4,5,6,7,8,9]
exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1]#exit마다 가중치

classifier_wise=True
unfreeze_ees=[0] #unfreeze exit list ex) [0,1,2,3,4,5,6,7,8,9]

# Early stopping parameters
early_stop_patience = 5
early_stop_counter = 0
best_val_accuracy = 0.0
##############################################################
# # 1. Data Preparation and Pretrained ViT model
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = dataset_name[data_choice](root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataset_name[data_choice](root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the pretrained ViT model from the saved file
pretrained_vit = models.vit_b_16(weights=None)
pretrained_vit.heads.head = nn.Linear(pretrained_vit.heads.head.in_features, dataset_outdim[data_choice])  # Ensure output matches the number of classes

# Load model weights
pretrained_vit.load_state_dict(torch.load(backbone_path))
pretrained_vit = pretrained_vit.to(device)
#from torchinfo import summary
#summary(pretrained_vit,input_size= (64, 3, IMG_SIZE, IMG_SIZE))
#freezing model
for param in pretrained_vit.parameters():
    param.requires_grad = False

##############################################################
# # 2. Define Multi-Exit ViT
class MultiExitViT(nn.Module):
    def __init__(self, base_model,dim=768, ee_list=[0,1,2,3,4,5,6,7,8,9],exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1],num_classes=10,image_size=IMG_SIZE,patch_size=16):
        super(MultiExitViT, self).__init__()
        assert len(ee_list)+1==len(exit_loss_weights), 'len(ee_list)+1==len(exit_loss_weights) should be True'
        self.base_model = base_model

        self.patch_size=patch_size
        self.hidden_dim=dim
        self.image_size=image_size
        
        # base model load
        self.conv_proj = base_model.conv_proj
        self.class_token = base_model.class_token
        self.pos_embedding = base_model.encoder.pos_embedding
        self.dropdout=base_model.encoder.dropout
        self.encoder_blocks = nn.ModuleList([encoderblock for encoderblock in [*base_model.encoder.layers]])
        self.ln= base_model.encoder.ln
        self.heads = base_model.heads
        
        # Multiple Exit Blocks 추가
        self.exit_loss_weights = [elw/sum(exit_loss_weights) for elw in exit_loss_weights]
        self.ee_list = ee_list
        self.exit_num=len(ee_list)+1
        self.ees = nn.ModuleList([self.create_exit_Tblock(dim) for _ in range(len(ee_list))])
        self.classifiers = nn.ModuleList([nn.Linear(dim, num_classes) for _ in range(len(ee_list))])
        
        

    def create_exit_Tblock(self, dim):
        return nn.Sequential(
            models.vision_transformer.EncoderBlock(num_heads=12, hidden_dim=dim, mlp_dim= 3072, dropout=0.0, attention_dropout=0.0),
            nn.LayerNorm(dim)
        )

    def getELW(self):
        if(self.exit_loss_weights is None):
            self.exit_loss_weights = [1]*self.exit_num
        return self.exit_loss_weights

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
    
    def forward(self, x):
        ee_cnter=0
        outputs = []
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropdout(x)
        
        #x = self.encoder(x)
        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            if idx in self.ee_list:
                y = self.ees[ee_cnter](x)
                y = y[:, 0]
                y = self.classifiers[ee_cnter](y)
                outputs.append(y)
                ee_cnter+=1
        # Classifier "token" as used by standard language architectures
        # Append the final output from the original head
        x = self.ln(x)
        x = x[:, 0]

        x = self.heads(x)
        outputs.append(x)
        return outputs
##############################################################
# # 3. Training part
class Trainer:
    def __init__(self, model, params):
        self.model = model
        self.num_epochs = params['num_epochs'];self.loss_func = params["loss_func"]
        self.opt = params["optimizer"];self.train_dl = params["train_dl"]
        self.val_dl = params["val_dl"];self.lr_scheduler = params["lr_scheduler"]
        self.isload = params["isload"];self.path_chckpnt = params["path_chckpnt"]
        self.classifier_wise = params["classifier_wise"];self.unfreeze_ees = params["unfreeze_ees"]
        self.best_loss = float('inf')
        self.old_epoch = 0
        self.device = next(model.parameters()).device

        # Initialize directory for model saving
        self.current_time = time.strftime('%m%d_%H%M%S', time.localtime())
        self.path = f'./models/{self.current_time}'
        os.makedirs(self.path, exist_ok=True)

        # Setup TensorBoard writer
        self.writer = SummaryWriter(f'./runs/{self.current_time}')

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
                self.best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'loss': val_loss,
                }, f'{self.path}/best_model.pth')
                print("Saved best model weights!")

            self.lr_scheduler.step(val_loss)

            # Logging
            elapsed_time = (time.time() - start_time) / 60
            hours, minutes = divmod(elapsed_time, 60)
            print(f'train_loss: {train_loss:.6f}, train_acc: {train_accs}')
            print(f'val_loss: {val_loss:.6f}, val_acc: {val_accs}, time: {int(hours)}h {int(minutes)}m')
            print('-' * 10)

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

model = MultiExitViT(pretrained_vit,num_classes=dataset_outdim[data_choice],ee_list=ee_list,exit_loss_weights=exit_loss_weights).to(device)
optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
lr_scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

params={'num_epochs':max_epochs, 'loss_func':criterion, 'optimizer':optimizer, 
        'train_dl':train_loader, 'val_dl':test_loader, 'lr_scheduler':lr_scheduler, 
        'isload':mevit_isload, 'path_chckpnt':mevit_pretrained_path,'classifier_wise':classifier_wise,
        'unfreeze_ees':unfreeze_ees}
t1=Trainer(model=model, params=params)
t1.train()
for i in range(unfreeze_ees_list):
    unfreeze_ees=[i]
    t1.train()