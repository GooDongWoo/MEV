
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models,datasets, transforms
from mevit_model import MultiExitViT
from tqdm import tqdm


# Define the Temperature Scaling class
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # Initialize temperature as 1.0

    def forward(self, logits):
        return logits / self.temperature

def optimize_temperature(model, scalers, test_loader,exit_num=11,lr=0.01, max_iter=50):
    model.eval()
    output_list_list = [[] for _ in range(exit_num)]
    labels_list = []
    
    # Collect logits and labels
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Collecting logits",leave=False):
            images, labels = images.cuda(), labels.cuda()
            output_list = model(images)
            for i in range(exit_num):
                output_list_list[i].append(output_list[i])
            labels_list.append(labels)

    output_list=[torch.cat(output_list_list[i]) for i in range(exit_num)]
    labels = torch.cat(labels_list)
    
    # Define NLL loss and optimizer
    nll_criterion = nn.CrossEntropyLoss()
    optimizers = [optim.LBFGS([scalers[i].temperature], lr=lr, max_iter=max_iter) for i in range(exit_num)]

    # Optimization loop
    for i in tqdm(range(exit_num),desc="Optimizing temperature",leave=False):
        def closure():
            optimizers[i].zero_grad()
            loss = nll_criterion(scalers[i](output_list[i]), labels)
            loss.backward()
            return loss
        optimizers[i].step(closure)

    print(f"Optimized Temperature: {[scaler.temperature.item() for scaler in scalers]}")
    return scalers
####################################################################
if __name__=='__main__':
    IMG_SIZE = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_name=dict();dataset_name['cifar10']=datasets.CIFAR10;dataset_name['cifar100']=datasets.CIFAR100;dataset_name['imagenet']=datasets.ImageNet
    dataset_outdim=dict();dataset_outdim['cifar10']=10;dataset_outdim['cifar100']=100;dataset_outdim['imagenet']=1000
    ##############################################################
    ################ 0. Hyperparameters ##########################
    ##############################################################
    batch_size = 1024
    data_choice='cifar10'
    mevit_isload=True
    mevit_pretrained_path=f'models/{data_choice}/integrated_ee.pth'
    max_epochs = 200  # Set your max epochs

    backbone_path=f'models/{data_choice}/vit_{data_choice}_backbone.pth'
    start_lr=1e-4
    weight_decay=1e-4

    ee_list=[0,1,2,3,4,5,6,7,8,9]#exit list ex) [0,1,2,3,4,5,6,7,8,9]
    exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1]#exit마다 가중치
    exit_num=11
    ##############################################################
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
    pretrained_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    if data_choice != 'imagenet':
        pretrained_vit.heads.head = nn.Linear(pretrained_vit.heads.head.in_features, dataset_outdim[data_choice])  # Ensure output matches the number of classes

        # Load model weights
        pretrained_vit.load_state_dict(torch.load(backbone_path))
        pretrained_vit = pretrained_vit.to(device)
    #from torchinfo import summary
    #summary(pretrained_vit,input_size= (64, 3, IMG_SIZE, IMG_SIZE))

    model = MultiExitViT(pretrained_vit,num_classes=dataset_outdim[data_choice],ee_list=ee_list,exit_loss_weights=exit_loss_weights).to(device)
    # Assume a pretrained model (replace with your own model)
    model.load_state_dict(torch.load(mevit_pretrained_path))  # Load your trained weights
    model.eval()

    # Temperature Scaling
    temperature_scalers = [TemperatureScaling().to(device) for _ in range(exit_num)]

    # Define a function to optimize the temperature

    # Optimize temperature
    temperature_scaler = optimize_temperature(model, temperature_scalers, test_loader)

    # save temperature scaling values
    torch.save(temperature_scaler, f'models/{data_choice}/temperature_scaler.pth')