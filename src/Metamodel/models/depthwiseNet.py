import torch
import torch.nn as nn
import torch.nn.functional as F



class depthwise_conv(nn.Module):
    
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, bias=False):
        super(depthwise_conv, self).__init__()
        
        self.depthwise = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, groups=in_dim, bias=bias)
        self.pointwise = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=bias)
        
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        
        return out
    
    
    
class DepthNet(nn.Module):
    def __init__(self, lengths=30, patch_size=1, in_chans=5, embed_dim=256, norm_layer=None, output_dim=2):
        super().__init__()
        #num_patches = num_voxels // patch_size
        #self.patch_shape = patch_size
        #self.num_voxels = num_voxels
        #self.patch_size = patch_size
        #self.num_patches = num_patches
        self.lengths = lengths
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.temporal_embed_1 = nn.Conv1d(in_chans, 8*in_chans, kernel_size=9, stride=2, groups=in_chans, )
        self.temporal_embed_2 = nn.Conv1d(8*in_chans, 64*in_chans, kernel_size=9, stride=2, groups=in_chans,)
        # self.temporal_embed_3 = nn.Conv1d(64*in_chans, 128*in_chans, kernel_size=9, stride=2, groups=in_chans)
        self.proj = nn.Conv1d(64, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.relu = torch.nn.ReLU()
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
            
        else:
            self.norm = None
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2560, 128), # length=30 768  // length=10 256
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
        
            
    def forward(self, x,):
        B, C, L = x.shape # batch, channel, lengths
        x = F.relu(self.temporal_embed_1(x))
        x = F.relu(self.temporal_embed_2(x))
        # x = F.relu(self.temporal_embed_3(x))
        
        x_ = torch.split(x, x.shape[1]//C, dim=1)
        x_ = torch.cat([x_[i] for i in range(C)], dim=-1) # B x 64 x 12*28
        
        x = self.proj(x_)
        x = x.transpose(1, 2).contiguous() # B x 28 x 512
        
        if self.norm is not None:
            x = self.norm(x)
        x = self.classifier(x.view(B, -1))
        return x