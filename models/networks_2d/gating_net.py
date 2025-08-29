import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)    

class OutputHead(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(OutputHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv1(x)
 
class GatingModule(nn.Module):
    def __init__(self, in_channel=64*3):
        super().__init__()
        self.dim = in_channel
        self.linear1 = nn.Linear(self.dim, int(self.dim * 0.5))
        
        # Attention mechanism
        self.attention = nn.Linear(int(self.dim * 0.5), int(self.dim * 0.5))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(int(self.dim * 0.5), in_channel // 64)

    def forward(self, x):
    
        x = torch.mean(x, dim=2, keepdim=True).squeeze(dim=2)
        x = torch.mean(x, dim=2, keepdim=True).squeeze(dim=2)
        x = self.linear1(x)
        x = self.relu(x)
        
        attention_scores = self.attention(x)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        x = x * attention_weights
        
        x = self.dropout(x)
        x = self.linear2(x)
        prob = F.softmax(x, dim=-1)

        prob = prob.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # B, num_expert, 1, 1, 1
        
        return prob

class MultiGatingNetwork(nn.Module):
    def __init__(self, in_channels=64*3):
        super().__init__()

        self.gating1 = GatingModule(in_channels)
        self.gating2 = GatingModule(in_channels)
        self.gating3 = GatingModule(in_channels)

        self.mask_head = OutputHead(64, 2)
        self.sdf_head = OutputHead(64, 1)
        self.bnd_head = OutputHead(64, 2)
    
    def forward(self, x):
        B, C, H, W = x.shape

        w1 = self.gating1(x)
        w2 = self.gating2(x)
        w3 = self.gating3(x)
        x = x.view(B, 3, 64, H, W)

        out1_f = (x * w1).sum(dim=1)
        out2_f = (x * w2).sum(dim=1) 
        out3_f = (x * w3).sum(dim=1)

        out1 = self.mask_head(out1_f)
        out2 = self.sdf_head(out2_f)
        out3 = self.bnd_head(out3_f)
        
        return out1, out2, out3


def multi_gating_attention(in_channels, num_classes):
    model = MultiGatingNetwork()
    init_weights(model, 'kaiming')
    return model

