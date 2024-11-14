# utils
import torch
import torch.nn as nn
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm  # Importing tqdm for progress bar
from torch.utils.data import DataLoader

# Check if GPU is available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
IMG_SIZE = 224

# 데이터셋 전처리 설정: CIFAR-10을 사용하고 이미지 크기를 32로 리사이즈
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset_name=dict()
dataset_name['cifar10']=datasets.CIFAR10
dataset_name['cifar100']=datasets.CIFAR100

dataset_outdim=dict()
dataset_outdim['cifar10']=10
dataset_outdim['cifar100']=100

##############################################################
data_choice='cifar100'
##############################################################

test_dataset = dataset_name[data_choice](root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

    print(f"Accuracy of the model on the {len(test_dataset)} test images: {100 * correct / total}%")

#getTestACC()
from torchinfo import summary
summary(pretrained_vit,input_size= (64, 3, IMG_SIZE, IMG_SIZE))