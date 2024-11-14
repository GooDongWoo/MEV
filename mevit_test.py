# utils
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # Importing tqdm for progress bar
IMG_SIZE = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_name = dict()
dataset_name['cifar10'] = datasets.CIFAR10
dataset_name['cifar100'] = datasets.CIFAR100
dataset_outdim = dict()
dataset_outdim['cifar10'] = 10
dataset_outdim['cifar100'] = 100
##############################################################
################ 0. Hyperparameters ##########################
batch_size = 32
data_choice = 'cifar100'
mevit_pretrained_path = f'integrated_ee.pth'

backbone_path = f'vit_{data_choice}_backbone.pth'
ee_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # exit list ex) [0,1,2,3,4,5,6,7,8,9]
exit_loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # exit마다 가중치
##############################################################
# # 2. Define Multi-Exit ViT
class MultiExitViT(nn.Module):
    def __init__(self, base_model, dim=768, ee_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], exit_loss_weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], num_classes=10, image_size=IMG_SIZE, patch_size=16):
        super(MultiExitViT, self).__init__()
        assert len(ee_list) + 1 == len(exit_loss_weights), 'len(ee_list)+1==len(exit_loss_weights) should be True'
        self.base_model = base_model

        self.patch_size = patch_size
        self.hidden_dim = dim
        self.image_size = image_size

        # base model load
        self.conv_proj = base_model.conv_proj
        self.class_token = base_model.class_token
        self.pos_embedding = base_model.encoder.pos_embedding
        self.dropdout = base_model.encoder.dropout
        self.encoder_blocks = nn.ModuleList([encoderblock for encoderblock in [*base_model.encoder.layers]])
        self.ln = base_model.encoder.ln
        self.heads = base_model.heads

        # Multiple Exit Blocks 추가
        self.exit_loss_weights = [elw / sum(exit_loss_weights) for elw in exit_loss_weights]
        self.ee_list = ee_list
        self.exit_num = len(ee_list) + 1
        self.ees = nn.ModuleList([self.create_exit_Tblock(dim) for _ in range(len(ee_list))])
        self.classifiers = nn.ModuleList([nn.Linear(dim, num_classes) for _ in range(len(ee_list))])

    def create_exit_Tblock(self, dim):
        return nn.Sequential(
            models.vision_transformer.EncoderBlock(num_heads=12, hidden_dim=dim, mlp_dim=3072, dropout=0.0, attention_dropout=0.0),
            nn.LayerNorm(dim)
        )

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        ee_cnter = 0
        outputs = []
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropdout(x)

        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            if idx in self.ee_list:
                y = self.ees[ee_cnter](x)
                y = y[:, 0]
                y = self.classifiers[ee_cnter](y)
                outputs.append(y)
                ee_cnter += 1
        
        # Classifier "token" as used by standard language architectures
        x = self.ln(x)
        x = x[:, 0]
        x = self.heads(x)
        outputs.append(x)
        return outputs
##############################################################
if __name__ == '__main__':
    # # 1. Data Preparation and Pretrained ViT model
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = dataset_name[data_choice](root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the pretrained ViT model from the saved file
    pretrained_vit = models.vit_b_16(weights=None)
    pretrained_vit.heads.head = nn.Linear(pretrained_vit.heads.head.in_features, dataset_outdim[data_choice])  # Ensure output matches the number of classes

    # Load model weights
    pretrained_vit.load_state_dict(torch.load(backbone_path))
    pretrained_vit = pretrained_vit.to(device)
    
    model = MultiExitViT(pretrained_vit, num_classes=dataset_outdim[data_choice], ee_list=ee_list, exit_loss_weights=exit_loss_weights).to(device)
    model.load_state_dict(torch.load(mevit_pretrained_path))    
    
    model.eval()
    running_metric = [0.0] * model.exit_num
    len_data = len(test_loader.dataset)

    with torch.no_grad():
        with tqdm(test_loader, unit="batch", leave=False) as t:
            for xb, yb in t:
                xb, yb = xb.to(device), yb.to(device)
                output_list = model(xb)
                accs = [output.argmax(1).eq(yb).sum().item() for output in output_list]
                running_metric = [sum(x) for x in zip(running_metric, accs)]
                
                t.set_postfix(accuracy=[round(100 * acc / len(xb),3) for acc in accs])

    running_acc = [100 * metric / len_data for metric in running_metric]
    print(f'total Test Accuracy: {running_acc}')