import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModule(nn.Module):
    def __init__(self,C):
        super(MyModule, self).__init__()
        self.C = C

    def forward(self, x):
        C = self.C
        x_ = torch.split(x, x.shape[1]//C, dim=1)
        x_ = torch.cat([x_[i] for i in range(C)], dim=-1)
        return x_

class MyModule2(nn.Module):
    def __init__(self):
        super(MyModule2, self).__init__()

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        return x
    
    
class DepthNet(nn.Module):
    def __init__(self, lengths=30, patch_size=1, in_chans=5, embed_dim=256, norm_layer=None, output_dim=3):
        print("DepthNet is used...")
        super().__init__()
        #num_patches = num_voxels // patch_size
        #self.patch_shape = patch_size
        #self.num_voxels = num_voxels
        #self.patch_size = patch_size
        #self.num_patches = num_patches
        self.lengths = lengths
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # self.temporal_embed_1 = nn.Conv1d(in_chans, 8*in_chans, kernel_size=9, stride=2, groups=in_chans, )
        # self.temporal_embed_2 = nn.Conv1d(8*in_chans, 64*in_chans, kernel_size=9, stride=2, groups=in_chans,)
        # self.temporal_embed_3 = nn.Conv1d(64*in_chans, 128*in_chans, kernel_size=9, stride=2, groups=in_chans)
        # self.proj = nn.Conv1d(64, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.relu = torch.nn.ReLU()
        self.C = in_chans
            
        self.features = torch.nn.Sequential(
            nn.Conv1d(in_chans, 16*in_chans, kernel_size=7, stride=1, groups=in_chans,),
            nn.ReLU(),
            nn.Conv1d(16*in_chans, 128*in_chans, kernel_size=7, stride=1, groups=in_chans,),
            nn.ReLU(),
            MyModule(self.C),
            nn.Conv1d(128, embed_dim, kernel_size=patch_size, stride=patch_size),
            MyModule2(),
            nn.LayerNorm(embed_dim)
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 128), # length=30 768  // length=10 256
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
    
    def forward(self, x,):
        x = self.features(x)
        activation = [x]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x, activation

    def reset_parameters(self):
        for component in [self.features, self.classifier]:
            for m in component:
            # Not all modules have parameters to reset
                try:
                    m.reset_parameters()
                except AttributeError:
                    pass
                
                
        # def forward(self, x,):
    #     B, C, L = x.shape # batch, channel, lengths
    #     x = F.relu(self.temporal_embed_1(x))
    #     x = F.relu(self.temporal_embed_2(x))
    #     # x = F.relu(self.temporal_embed_3(x))
        
    #     x_ = torch.split(x, x.shape[1]//C, dim=1)
    #     x_ = torch.cat([x_[i] for i in range(C)], dim=-1) # B x 64 x 12*28
        
    #     x = self.proj(x_)
    #     x = x.transpose(1, 2).contiguous() # B x 28 x 512
        
    #     if self.norm is not None:
    #         x = self.norm(x)
    #     x = self.classifier(x.view(B, -1))
    #     return x