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
batch_size = 32
data_choice='cifar100'
mevit_isload=False
mevit_pretrained_path=f'models/1108_103451/best_model.pth'
max_epochs = 100  # Set your max epochs

backbone_path=f'vit_{data_choice}_backbone.pth'
start_lr=1e-4
weight_decay=1e-4
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

for name, param in pretrained_vit.named_parameters():
    print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")
    
##############################################################
# # 2. Define Multi-Exit ViT
class MultiExitViT(nn.Module):
    def __init__(self, base_model,dim=768, ee_list=[0,1,2,3,4,5,6,7,8,9],exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1],num_classes=10,image_size=IMG_SIZE,patch_size=16):
        super(MultiExitViT, self).__init__()
        assert len(ee_list)+1==len(exit_loss_weights), 'len(ee_list)+1==len(exit_loss_weights) should be True'
        self.base_model = base_model

        self.class_token = base_model.class_token
        self.patch_size=patch_size
        self.hidden_dim=dim
        self.image_size=image_size
        
        self.exit_loss_weights = [elw/sum(exit_loss_weights) for elw in exit_loss_weights]
        
        # Multiple Exit Blocks 추가
        self.ee_list = ee_list
        self.exit_num=len(ee_list)+1
        self.ees = nn.ModuleList([self.create_exit_Tblock(dim) for _ in range(len(ee_list))])
        self.classifiers = nn.ModuleList([nn.Linear(dim, num_classes) for _ in range(len(ee_list))])
        
        # base model load
        self.conv_proj = base_model.conv_proj
        self.encoder_blocks = nn.ModuleList([encoderblock for encoderblock in [*base_model.encoder.layers]])
        
        # Final head
        self.ln= base_model.encoder.ln
        self.heads = base_model.heads

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

        #x = self.encoder(x)
        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            if idx in self.ee_list:
                y = self.ees[ee_cnter](x)#TODO: check if (idx) this is correct
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
# function to get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# function to calculate metric per mini-batch
def metric_batch(output, label):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(label.view_as(pred)).sum().item()
    return corrects

# function to calculate loss per mini-batch
def loss_batch(loss_func, output_list, label, elws, opt=None):
    losses = [loss_func(output,label)*elw for output,elw in zip(output_list,elws)]   # raw losses -> 굳이 각각 exit의 길이로 나눠줘야하나? 로스 크기만 달라지지 않나;;
    acc_s = [metric_batch(output, label) for output in output_list]
    
    if opt is not None:
        opt.zero_grad()
        #backprop
        for loss in losses[:-1]:
            #ee losses need to keep graph
            loss.backward(retain_graph=True)
        #final loss, graph not required
        losses[-1].backward()
        opt.step()
    
    losses = [loss.item() for loss in losses] #for out of cuda memory error
    
    return losses, acc_s

# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset_dl, writer, epoch, opt=None):
    device = next(model.parameters()).device
    running_loss = [0.0] * model.exit_num
    running_metric = [0.0] * model.exit_num
    len_data = len(dataset_dl.dataset)
    TorV='train' if opt is not None else 'val'
    
    with tqdm(dataset_dl, desc=f"{TorV}: {epoch}th Epoch", unit="batch",leave=False) as t:
        for xb, yb in t:
            xb = xb.to(device)
            yb = yb.to(device)
            output_list = model(xb)
            elws=model.getELW()

            losses, acc_s = loss_batch(loss_func, output_list, yb, elws, opt)

            running_loss = [sum(i) for i in zip(running_loss,losses)]
            running_metric = [sum(i) for i in zip(running_metric,acc_s)]
            
            batch_acc=[100*i/len(xb) for i in acc_s]
            t.set_postfix(accuracy=batch_acc)
    
    running_loss=[i/len_data for i in running_loss]
    running_acc=[100*i/len_data for i in running_metric]
    
    # Tensorboard
    tmp_loss_dict = dict();tmp_acc_dict = dict()
    for idx in range(model.exit_num):
        tmp_loss_dict[f'exit{idx}'] = running_loss[idx];tmp_acc_dict[f'exit{idx}'] = running_acc[idx]
    writer.add_scalars(f'{TorV}/loss', tmp_loss_dict, epoch)
    writer.add_scalars(f'{TorV}/acc', tmp_acc_dict, epoch)
    
    losses_sum = sum(running_loss) # float
    writer.add_scalar(f'{TorV}/loss_total_sum', losses_sum, epoch)
    accs = running_acc # float list[exit_num]

    return losses_sum, accs

# function to start training
def train_val(model, params):   
    num_epochs=params['num_epochs'];loss_func=params["loss_func"]
    opt=params["optimizer"];train_dl=params["train_dl"]
    val_dl=params["val_dl"];lr_scheduler=params["lr_scheduler"]
    isload=params["isload"];path_chckpnt=params["path_chckpnt"]
    classifier_wise=params["classifier_wise"]
    unfreeze_num=params["unfreeze_num"];freeze_last=params["freeze_last"]
    
    start_time = time.time()
    
    # path to save the model weights
    current_time = time.strftime('%m%d_%H%M%S', time.localtime())
    path=f'./models/{current_time}'
    os.makedirs(path, exist_ok=True)
    
    spec_txt=f'opt: {opt.__class__.__name__}\nlr: {opt.param_groups[0]["lr"]}\nbatch: {train_dl.batch_size}\nepoch: {num_epochs}\nisload: {isload}\npath_chckpnt: {path_chckpnt}\nexits_loss_weights: {model.getELW()}\n'
    with open(f"{path}/spec.txt", "w") as file:
        file.write(spec_txt)
    
    best_loss = float('inf')
    old_epoch=0
    if(isload):
        chckpnt = torch.load(path_chckpnt,weights_only=True)
        model.load_state_dict(chckpnt['model_state_dict'])
        opt.load_state_dict(chckpnt['optimizer_state_dict'])
        old_epoch = chckpnt['epoch']
        best_loss = chckpnt['loss']
    
    if classifier_wise:
        #freeze all except classifier
        pass    
        #unfreeze specific classifier
        pass
    #writer=None
    writer = SummaryWriter('./runs/'+current_time,)
    #writer.add_graph(model, torch.rand(1,3,resize,resize).to(next(model.parameters()).device))
    
    for epoch in range(old_epoch,old_epoch+num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, old_epoch+num_epochs-1, current_lr))

        model.train()
        
        
        train_loss, train_accs = loss_epoch(model, loss_func, train_dl, writer, epoch, opt)

        model.eval()
        with torch.no_grad():
            val_loss, val_accs = loss_epoch(model, loss_func, val_dl, writer, epoch, opt=None)

        if val_loss < best_loss:
            best_loss = val_loss
            #best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': val_loss,
            }, path+'/best_model.pth')
            print('saved best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        total_time=(time.time()-start_time)/60
        hours, minutes = divmod(total_time, 60)
        print(f'train_loss: {train_loss:.6f}, train_acc: {train_accs}')
        print(f'val_loss: {val_loss:.6f}, val_acc: {val_accs}, time: {int(hours)}h {int(minutes)}m')
        print('-'*10)
        writer.flush()

    torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': val_loss,
            }, path+'/chckpoint.pth')
    writer.close()
    linedivisor='#'*10+'\n'
    result_txt=linedivisor+f'last_val_acc: {val_accs}\nlast_train_acc: {train_accs}\nlast_val_loss: {best_loss:.6f}\ntotal_time: {total_time:.2f}m\n'
    with open(f"{path}/spec.txt", "a") as file:
        file.write(result_txt)
    
    return model
model = MultiExitViT(pretrained_vit,num_classes=dataset_outdim[data_choice],ee_list=[0,1,2,3,4,5,6,7,8,9],exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
lr_scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

params={'num_epochs':max_epochs, 'loss_func':criterion, 'optimizer':optimizer, 
        'train_dl':train_loader, 'val_dl':test_loader, 'lr_scheduler':lr_scheduler, 
        'isload':mevit_isload, 'path_chckpnt':mevit_pretrained_path,'classifier_wise':True,
        'unfreeze_num':0,'freeze_last':1}

train_val(model=model, params=params)
#from torchinfo import summary
#summary(pretrained_vit,input_size= (64, 3, IMG_SIZE, IMG_SIZE))