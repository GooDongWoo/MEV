import huggingface_hub
import datasets
import torchvision.transforms as transforms
import torch

DOWNLOAD = False
IMG_SIZE = 224

if DOWNLOAD:
    huggingface_hub.login()
class ImageNetdataset(torch.utils.data.Dataset):
    def __init__(self, train_mode=True, transforms=None):
        try: imagenet_dataset = datasets.load_dataset("imagenet-1k",cache_dir='./data/imagenet')
        except: raise Exception("Please download the dataset in ./data/imagenet")
        if train_mode:
            self.dataset = imagenet_dataset["train"]
        else:
            self.dataset = imagenet_dataset["validation"]
        self.transforms = None
        if transforms:
            self.transforms = transforms
        self.targets = self.dataset['label']

        
    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        label = self.dataset[index]["label"]
        current = image.convert("RGB")
        if self.transforms:
            current = self.transforms(current)
        return current, label
    def __len__(self):
        return len(self.dataset)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(std=std, mean=mean)
])
IMAGENET_DATASET_TRAIN = ImageNetdataset(train_mode=True, transforms=trans)
IMAGENET_DATASET_TEST = ImageNetdataset(train_mode=False, transforms=trans)