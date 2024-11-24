from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

class Dloaders:
    def __init__(self,data_choice='cifar100',batch_size=1024,IMG_SIZE=224):
        self.dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
        self.dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}
        
        self.data_choice = data_choice
        if data_choice == 'imagenet':
            import imagenet_load
            
            self.train_dataset = imagenet_load.IMAGENET_DATASET_TRAIN
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            self.test_dataset = imagenet_load.IMAGENET_DATASET_TEST
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        else:
            transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.train_dataset = self.dataset_name[data_choice](root='./data', train=True, download=True, transform=transform)
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            self.test_dataset = self.dataset_name[data_choice](root='./data', train=False, download=True, transform=transform)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        return None
    
    def get_loaders(self):
        return self.train_loader,  self.test_loader
    
    def get_datasets(self):
        return self.train_dataset,  self.test_dataset