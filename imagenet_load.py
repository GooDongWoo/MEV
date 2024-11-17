import huggingface_hub
import datasets
from secret import *
huggingface_hub.login(token=token)
imagenet_dataset = datasets.load_dataset("imagenet-1k",cache_dir='./data/imagenet')
'''
import torch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train_mode=True, transforms=None):
        if train_mode:
            self.dataset = imagenet_dataset["train"]
        else:
            self.dataset = imagenet_dataset["validation"]
        self.transforms = None
        if transforms:
            self.transforms = transforms
        
    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        label = self.dataset[index]["label"]
    
        current = image.convert("RGB")
        if self.transforms:
            current = self.transforms(current)
    
        return current, label
    
    def __len__(self):
        return len(self.dataset)
    
import torchvision.transforms as transforms
from pytorch_ood.utils import ToRGB

mean = [x for x in [0.485, 0.456, 0.406]]
std = [x for x in [0.229, 0.224, 0.225]]

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    ToRGB(),
    transforms.ToTensor(),
    transforms.Normalize(std=std, mean=mean)
])

from torchvision.datasets import ImageFolder

batch_size = 1

train_dataset = ImageFolder("./imagenet_100_images_per_class", transform=trans)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

test_dataset = CustomDataset(train_mode=False, transforms=trans)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

print("Size of training dataset:", len(train_dataset))
print("Size of test dataset:", len(test_dataset))

from torchcam.methods import LayerCAM

num_classes = 1000
img_size = 224
input_shape = (3, img_size, img_size)

model = resnet50(num_classes=num_classes, pretrained=True).cuda().eval()
target_layer = model.layer2
localize_net = LayerCAM(model, target_layer=target_layer, input_shape=input_shape)
'''