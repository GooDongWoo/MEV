import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter

IMG_SIZE = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load dataset
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
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
start_lr=5e-5
isload=True
##############################################################

train_dataset = dataset_name[data_choice](root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = dataset_name[data_choice](root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model (assuming you have a similar model as in TensorFlow)
model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)

# Update the input size to match 224x224 for ViT
model.heads.head = nn.Linear(model.heads.head.in_features, dataset_outdim[data_choice])
model = model.to(device)

#load model
if isload:
    model.load_state_dict(torch.load(f'vit_{data_choice}_backbone.pth'))
    print('model loaded')
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=start_lr)

# Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=2, verbose=True, min_lr=1e-7)

# Early stopping parameters
early_stop_patience = 5
early_stop_counter = 0
best_val_accuracy = 0.0

# Training loop
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, max_epochs):
    global best_val_accuracy, early_stop_counter,IMG_SIZE
    current_time = time.strftime('%m%d_%H%M%S', time.localtime())
    writer = SummaryWriter('./runs/'+current_time,)
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epochs}]", unit="batch") as t:
            for images, labels in t:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                t.set_postfix(loss=running_loss/len(train_loader), accuracy=100 * correct / total)

        train_accuracy = 100 * correct / total
        writer.add_scalar(f'train/loss', running_loss/len(train_loader), epoch)
        writer.add_scalar(f'train/acc', train_accuracy, epoch)

        # Validation phase
        model.eval()
        running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        writer.add_scalar(f'val/loss', running_loss/len(test_loader), epoch)
        writer.add_scalar(f'val/acc', val_accuracy, epoch)
        # Check for best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'vit_{data_choice}_backbone.pth')
            print(f"Model improved and saved at epoch {epoch+1}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation accuracy for {early_stop_counter} epochs")

        # Scheduler step
        scheduler.step(val_accuracy)

        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

    print("Training complete")
    writer.close()
    return model, best_val_accuracy

# Training the model
max_epochs = 100  # Set your max epochs

model, test_accuracy = train(model, train_loader, test_loader, criterion, optimizer, scheduler, max_epochs)
print(f"Best Validation Accuracy: {test_accuracy:.2f}%")