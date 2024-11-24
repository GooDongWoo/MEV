# utils
import torch
import torch.nn as nn
from torchvision import datasets,models
from tqdm import tqdm  # Importing tqdm for progress bar
from Dloaders import Dloaders
# Check if GPU is available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
IMG_SIZE = 224
batch_size = 32
dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}

##############################################################
data_choice='cifar100'
##############################################################

dloaders=Dloaders(data_choice=data_choice,batch_size=batch_size,IMG_SIZE=IMG_SIZE)
train_loader,test_loader = dloaders.get_loaders()
# Load the pretrained ViT model from the saved file
pretrained_vit = models.vit_b_16(weights=None)
pretrained_vit.heads.head = nn.Linear(pretrained_vit.heads.head.in_features, dataset_outdim[data_choice])  # Ensure output matches the number of classes

model_path = f'vit_{data_choice}_backbone.pth'
# Load model weights
pretrained_vit.load_state_dict(torch.load(model_path, map_location=device))
pretrained_vit = pretrained_vit.to(device)

def getTestACC():
    # Set the model to evaluation mode
    pretrained_vit.eval()
    # Test the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_loader, unit="batch",leave=False) as t:
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = pretrained_vit(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                t.set_postfix(accuracy=100*correct/total)

    print(f"Accuracy of the model on the test images: {100 * correct / total}%")

#getTestACC()
from torchinfo import summary
summary(pretrained_vit,input_size= (64, 3, IMG_SIZE, IMG_SIZE))