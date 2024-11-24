from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import imagenet_load

class Dloaders:
    def __init__(self,data_choice='cifar100',batch_size=1024,IMG_SIZE=224):
        self.dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
        self.dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}
        
        self.data_choice = data_choice
        if data_choice == 'imagenet':
            train_dataset = imagenet_load.IMAGENET_DATASET_TRAIN
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = imagenet_load.IMAGENET_DATASET_TEST
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = self.dataset_name[data_choice](root='./data', train=True, download=True, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = self.dataset_name[data_choice](root='./data', train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader,test_loader