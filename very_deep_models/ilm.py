import torch
from torch import nn
import torch.nn.functional as F

#######
## ILM GN, IN and LN implementations are from the authors who proposed the ILM. 
## Their implementation is used for these three, see https://github.com/Gasoonjia/ILM-Norm/tree/master/lib/nn
##
## FSC BN, is the one technique we created as a combination of BN and idea of gathering statistics during the forward pass.
#######

"""
Instance-level Meta Normalization
Instance Normalization
"""
import torch
from torch import nn
import torch.nn.functional as F


class ilm_IN(nn.Module):
    def __init__(self, channels, key_group_size=5, reduction=5, eps=1e-5):
        super(ilm_IN, self).__init__()
        assert channels % key_group_size == 0
        assert reduction > 0
        
        self.num_groups = channels
        self.feat_per_group = 1

        self.key_groups = channels // key_group_size
        self.key_feat_per_group = key_group_size

        if self.key_groups > reduction:
            self.embed_size = self.key_groups // reduction
        else:
            self.embed_size = 2

        self.fc_embed = nn.Sequential(
            nn.Linear(self.key_groups, self.embed_size, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc_weight = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.key_groups, bias=False),
            nn.Sigmoid()
        )

        self.fc_bias = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.key_groups, bias=False),
            nn.Tanh()
        )
        
        self.weight_bias = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.bias_bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.eps = eps


    def forward(self, x):
        b, c, d, h, w = x.size()
        g = self.key_groups

        x = x.view(b, g, 1, -1)
        key_mean = x.mean(-1, keepdim=True)
        key_var = x.var(-1, keepdim=True)
        
        weight = self.fc_weight(key_var.view(b, g)).view(b, g, 1).repeat(1, 1, self.key_feat_per_group).view(b, c, 1, 1, 1)
        weight = weight + self.weight_bias
        bias = self.fc_bias(key_mean.view(b, g)).view(b, g, 1).repeat(1, 1, self.key_feat_per_group).view(b, c, 1, 1, 1)
        bias = bias + self.bias_bias

        g = self.num_groups
        x = x.view(b, g, 1, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(b, c, d, h, w)

        return x * weight + bias


"""
Instance-level Meta Normalization
Group Normalization
"""    
class ilm_GN(nn.Module):
    def __init__(self, channels, num_groups=5, key_group_size=5, reduction=5, eps=1e-5):
        super(ilm_GN, self).__init__()        
        assert num_groups % reduction == 0
        assert channels % num_groups == 0
        assert channels % key_group_size == 0

        self.num_groups = num_groups
        self.feat_per_group = channels // num_groups

        self.key_groups = channels // key_group_size
        self.key_feat_per_group = key_group_size

        if self.key_groups >= reduction:
            self.embed_size = self.key_groups // reduction
        else:
            self.embed_size = 2
        
        self.fc_embed = nn.Sequential(
            nn.Linear(self.key_groups, self.embed_size, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc_weight = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.key_groups, bias=False),
            nn.Sigmoid()
        )
        self.fc_bias = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.key_groups, bias=False),
            nn.Tanh()
        )
        
        self.weight_bias = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.bias_bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.eps = eps


    def forward(self, x):
        b, c, d, h, w = x.size()
        g = self.key_groups

        x = x.view(b, g, 1, -1)
        key_mean = x.mean(-1, keepdim=True)
        key_var = x.var(-1, keepdim=True)
        
        weight = self.fc_weight(key_var.view(b, g)).view(b, g, 1).repeat(1, 1, self.key_feat_per_group).view(b, c, 1, 1, 1)
        weight = weight + self.weight_bias
        bias = self.fc_bias(key_mean.view(b, g)).view(b, g, 1).repeat(1, 1, self.key_feat_per_group).view(b, c, 1, 1, 1)
        bias = bias + self.bias_bias

        g = self.num_groups
        x = x.view(b, g, 1, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(b, c, d, h, w)

        return x * weight + bias


"""
Instance-level Meta Normalization
Layer Normalization
"""
class ilm_LN(nn.Module):
    def __init__(self, channels, key_group_size=5, reduction=5, eps=1e-5):
        super(ilm_LN, self).__init__()
        assert channels % key_group_size == 0
        assert reduction > 0
        
        self.eps = eps
        
        self.num_groups = 1
        self.feat_per_group = channels

        self.key_groups = channels // key_group_size
        self.key_feat_per_group = key_group_size

        if self.key_groups > reduction:
            self.embed_size = self.key_groups // reduction
        else:
            self.embed_size = 2

        self.fc_embed = nn.Sequential(
            nn.Linear(self.key_groups, self.embed_size, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc_weight = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.key_groups, bias=False),
            nn.Sigmoid()
        )

        self.fc_bias = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.key_groups, bias=False),
            nn.Tanh()
        )
        
        self.weight_bias = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.bias_bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))


    def forward(self, x):
        b, c, d, h, w = x.size()
        g = self.key_groups

        x = x.view(b, g, 1, -1)
        key_mean = x.mean(-1, keepdim=True)
        key_var = x.var(-1, keepdim=True)
        
        weight = self.fc_weight(key_var.view(b, g)).view(b, g, 1).repeat(1, 1, self.key_feat_per_group).view(b, c, 1, 1, 1)
        weight = weight + self.weight_bias
        bias = self.fc_bias(key_mean.view(b, g)).view(b, g, 1).repeat(1, 1, self.key_feat_per_group).view(b, c, 1, 1, 1)
        bias = bias + self.bias_bias

        g = self.num_groups
        x = x.view(b, g, 1, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(b, c, d, h, w)
        
        return x * weight + bias


"""
Using the idea of acquiring attributions during the forward pass with an Auto-Encoder for normalization
such as is done with Instance-level meta normalization but on a batch rather than instances.
Batch Normalization
"""
## forward_stats_catcher_BN (short fsc)
class fsc_BN3d(nn.Module):
    def __init__(self, channels, embed_size, eps=1e-5):
        super(fsc_BN3d, self).__init__()
        assert channels > 0
        assert embed_size < channels
        
        self.eps = eps
        self.channels = channels
        self.embed_size = embed_size

        self.fc_embed = nn.Sequential(
            nn.Linear(self.channels, self.embed_size, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.fc_weight = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.channels, bias=False),
            nn.Sigmoid()
        )

        self.fc_bias = nn.Sequential(
            self.fc_embed,
            nn.Linear(self.embed_size, self.channels, bias=False),
            nn.Tanh()
        )
        
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.zeros(channels))
        
        self.weight_bias = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.bias_bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))


    def forward(self, x):
        b, c, d, h, w = x.size()
        
        mean = x.view(b, c, -1).mean([-1, 0], keepdim=True).squeeze()
        var = x.view(b, c, -1).var([-1, 0], keepdim=True).squeeze()
        
        weight = self.fc_weight(mean)
        weight = weight[None, :, None, None, None] + self.weight_bias
        
        bias = self.fc_bias(var)
        bias = bias[None, :, None, None, None] + self.bias_bias
        
        if self.training:
            ## X_hat_new = (1−momentum) × X_hat + momentum × x_t , where X_hat is the estimated statistic and x_t is the new observed value.
            ## See pytorch documentation on batch norm for how running mean and var are calculated using momentum.
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean ## momentum is 0.1 since that is what was used in batch norm on the BN model lacking statistics from forward pass
            self.running_var = 0.9 * self.running_var + 0.1 * var  ## momentum is 0.1 since that is what was used in batch norm on the BN model lacking statistics from forward pass
            
            x = (x - mean[None, :, None, None, None]) / (var + self.eps).sqrt()[None, :, None, None, None]
        else:
            x = (x - self.running_mean[None, :, None, None, None]) / (self.running_var + self.eps).sqrt()[None, :, None, None, None]
        
        return x * weight + bias