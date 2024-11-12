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
dataset_name=dict()
dataset_name['cifar10']=datasets.CIFAR10
dataset_name['cifar100']=datasets.CIFAR100

dataset_outdim=dict()
dataset_outdim['cifar10']=10
dataset_outdim['cifar100']=100

##############################################################
################ 0. Hyperparameters ##########################
unfreeze_ees_list=[0,1,2,3,4,5,6,7,8,9]

# Path to the saved model
ee0_path=''
ee1_path=''
ee2_path=''
ee3_path=''
ee4_path=''
ee5_path=''
ee6_path=''
ee7_path=''
ee8_path=''
ee9_path=''
##############################################################
data_choice='cifar100'
# Load the pretrained ViT model from the saved file
pretrained_vit = models.vit_b_16(weights=None)
pretrained_vit.heads.head = nn.Linear(pretrained_vit.heads.head.in_features, dataset_outdim[data_choice])  # Ensure output matches the number of classes

# Load model weights
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
base_model = MultiExitViT(base_model=pretrained_vit,num_classes=dataset_outdim[data_choice])

##############################################################
paths=[ee0_path,ee1_path,ee2_path,ee3_path,ee4_path,ee5_path,ee6_path,ee7_path,ee8_path,ee9_path]


##############################################################
checkpoint = torch.load(paths[0])['model_state_dict']
ees_name=[[] for _ in range(10)]
for i in range(10):
    for j in list(checkpoint.keys()):
        if (f'ees.{i}' in j) or (f'classifiers.{i}' in j):
            ees_name[i].append(j)
##############################################################
for i in range(10):
    checkpoint = torch.load(paths[i])
    for j in ees_name[i]:
        base_model.load_state_dict(checkpoint[j])

##############################################################
# Save the model
torch.save(base_model.state_dict(), 'integrated_ee.pth')