# utils
import torch
import torch.nn as nn
from torchvision import models

# # 2. Define Multi-Exit ViT
class MultiExitViT(nn.Module):
    def __init__(self, base_model,dim=768, ee_list=[0,1,2,3,4,5,6,7,8,9],exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1],num_classes=10,image_size=224,patch_size=16):
        super(MultiExitViT, self).__init__()
        assert len(ee_list)+1==len(exit_loss_weights), 'len(ee_list)+1==len(exit_loss_weights) should be True'
        self.base_model = base_model

        self.patch_size=patch_size
        self.hidden_dim=dim
        self.image_size=image_size
        
        # base model load
        self.conv_proj = base_model.conv_proj
        self.class_token = base_model.class_token
        self.pos_embedding = base_model.encoder.pos_embedding
        self.dropdout=base_model.encoder.dropout
        self.encoder_blocks = nn.ModuleList([encoderblock for encoderblock in [*base_model.encoder.layers]])
        self.ln= base_model.encoder.ln
        self.heads = base_model.heads
        
        # Multiple Exit Blocks 추가
        self.exit_loss_weights = [elw/sum(exit_loss_weights) for elw in exit_loss_weights]
        self.ee_list = ee_list
        self.exit_num=len(ee_list)+1
        self.ees = nn.ModuleList([self.create_exit_Tblock(dim) for _ in range(len(ee_list))])
        self.classifiers = nn.ModuleList([nn.Linear(dim, num_classes) for _ in range(len(ee_list))])
        
        self.each_ee_test_mode = False

    def create_exit_Tblock(self, dim):
        return nn.Sequential(
            models.vision_transformer.EncoderBlock(num_heads=12, hidden_dim=dim, mlp_dim= 3072, dropout=0.0, attention_dropout=0.0),
            nn.LayerNorm(dim)
        )

    def getELW(self):
        if(self.exit_loss_weights is None):
            self.exit_loss_weights = [1]*self.exit_num
        return self.exit_loss_weights

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
    def set_each_ee_test_mode(self,num = 10):#last exit
        self.each_ee_test_mode = True
        self.each_ee_test_what_num = num
        
    def forward_each_ee(self, x):
        target_exit = self.each_ee_test_what_num
        
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropdout(x)
        
        if target_exit == (self.exit_num - 1):   #last exit
            for block in (self.encoder_blocks):
                x = block(x)
            x = self.ln(x)
            x = x[:, 0]

            x = self.heads(x)
            return x
        
        else:
            for idx, block in enumerate(self.encoder_blocks):
                x = block(x)
                if idx == target_exit:
                    y = self.ees[target_exit](x)
                    y = y[:, 0]
                    y = self.classifiers[target_exit](y)
                    return y

    def forward(self, x):
        if self.each_ee_test_mode:return self.forward_each_ee(x)
        ee_cnter=0
        outputs = []
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropdout(x)
        
        #x = self.encoder(x)
        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            if idx in self.ee_list:
                y = self.ees[ee_cnter](x)
                y = y[:, 0]
                y = self.classifiers[ee_cnter](y)
                outputs.append(y)
                ee_cnter+=1
        # Classifier "token" as used by standard language architectures
        # Append the final output from the original head
        x = self.ln(x)
        x = x[:, 0]

        x = self.heads(x)
        outputs.append(x)
        return outputs