import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from tqdm import tqdm

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the model (assuming you have a similar model as in TensorFlow)
model = torchvision.models.vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, 10)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=2, verbose=True, min_lr=1e-7)

# Early stopping parameters
early_stop_patience = 5
early_stop_counter = 0
best_val_accuracy = 0.0

# Training loop
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, max_epochs):
    global best_val_accuracy, early_stop_counter
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
        print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Check for best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'vit_cifar10_v1.pth')
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
    return model, best_val_accuracy

# Training the model
max_epochs = 100  # Set your max epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, test_accuracy = train(model, train_loader, val_loader, criterion, optimizer, scheduler, max_epochs)
print(f"Best Validation Accuracy: {test_accuracy:.2f}%")