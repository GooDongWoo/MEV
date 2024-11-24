# utils
import torch
import torch.nn as nn
from torchvision import datasets, models
from tqdm import tqdm  # Importing tqdm for progress bar
from mevit_model import MultiExitViT
from Dloaders import Dloaders
IMG_SIZE = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}
##############################################################
################ 0. Hyperparameters ##########################
batch_size = 32
data_choice = 'cifar100'
mevit_pretrained_path=f'models/{data_choice}/integrated_ee.pth'

backbone_path = f'vit_{data_choice}_backbone.pth'
ee_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # exit list ex) [0,1,2,3,4,5,6,7,8,9]
exit_loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # exit마다 가중치

##############################################################
if __name__ == '__main__':
    train_loader,test_loader = Dloaders(data_choice=data_choice,batch_size=batch_size,IMG_SIZE=IMG_SIZE)

    # Load the pretrained ViT model from the saved file
    pretrained_vit = models.vit_b_16(weights=None)
    pretrained_vit.heads.head = nn.Linear(pretrained_vit.heads.head.in_features, dataset_outdim[data_choice])  # Ensure output matches the number of classes

    # Load model weights
    pretrained_vit.load_state_dict(torch.load(backbone_path))
    pretrained_vit = pretrained_vit.to(device)
    
    model = MultiExitViT(pretrained_vit, num_classes=dataset_outdim[data_choice], ee_list=ee_list, exit_loss_weights=exit_loss_weights).to(device)
    model.load_state_dict(torch.load(mevit_pretrained_path))    
    
    model.eval()
    running_metric = [0.0] * model.exit_num
    len_data = len(test_loader.dataset)

    with torch.no_grad():
        with tqdm(test_loader, unit="batch", leave=False) as t:
            for xb, yb in t:
                xb, yb = xb.to(device), yb.to(device)
                output_list = model(xb)
                accs = [output.argmax(1).eq(yb).sum().item() for output in output_list]
                running_metric = [sum(x) for x in zip(running_metric, accs)]
                
                t.set_postfix(accuracy=[round(100 * acc / len(xb),3) for acc in accs])

    running_acc = [100 * metric / len_data for metric in running_metric]
    print(f'total Test Accuracy: {running_acc}')