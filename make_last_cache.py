import torch
import torch.nn as nn
from torchvision import models,datasets
from mevit_model import MultiExitViT
from tqdm import tqdm
import numpy as np
from Dloaders import Dloaders

if __name__ == '__main__':
    ####################################################################
    IMG_SIZE = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
    dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}
    ##############################################################
    ################ 0. Hyperparameters ##########################
    ##############################################################
    batch_size = 1024
    data_choice='cifar100'
    mevit_isload=True
    mevit_pretrained_path=f"models/{data_choice}/integrated_ee.pth"

    backbone_path=f'models/{data_choice}/vit_{data_choice}_backbone.pth'
    cache_file_path = f'cache_result_mevit_{data_choice}.pt'
    
    ee_list=[0,1,2,3,4,5,6,7,8,9]#exit list ex) [0,1,2,3,4,5,6,7,8,9]
    exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1]#exit마다 가중치
    exit_num=11
    ##############################################################
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

    model = MultiExitViT(pretrained_vit,num_classes=dataset_outdim[data_choice],ee_list=ee_list,exit_loss_weights=exit_loss_weights).to(device)

    # Assume a pretrained model (replace with your own model)
    if mevit_isload:
        model.load_state_dict(torch.load(mevit_pretrained_path))  # Load your trained weights

    # Initialize lists to store outputs from each exit and labels
    output_list_list = [[] for _ in range(exit_num)]
    labels_list = []

    # Run inference on the test set and collect the logits from each exit
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Collecting logits", leave=False):
            images, labels = images.cuda(), labels.cuda()
            output_list = model(images)  # Get the output from all exits
            
            # Store the output from each exit
            for i in range(exit_num):
                output_list_list[i].append(output_list[i])
            
            # Store the labels
            labels_list.append(labels)
    # Concatenate the collected outputs and labels
    output_tensor = torch.tensor(np.array([torch.cat(output_list_list[i]).to('cpu') for i in range(exit_num)])).to(device)
    labels = torch.cat(labels_list).to(device)
    # 리스트를 바이너리 파일로 저장하기
    torch.save(output_tensor, cache_file_path)