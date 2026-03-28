from  VMamba.kernels.vmamba import Backbone_VSSM
import torch
import torch.nn as nn
import torch.nn.functional as F
from HA import HA

class Vmamba_Backbone(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer="ln",
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=128,
                 pretrained=None,
                 mlp_ratio=0.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[352, 352],
                 patch_size=4,
                 drop_path_rate=0.6,
                 **kwargs):
        super().__init__()

        self.ape = ape

        self.vssm_r = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
            forward_type="v0",
            ssm_d_state=16,
            ssm_ratio=2.0,
            patchembed_version="v1",
        )
        self.ha = HA()
    def forward(self, x):
        """
        rgb: B x C x H x W
        """
        B = x.shape[0]
        outs_fused = []
        outs = self.vssm_r(x) # B x C x H x W
        return outs
    

class vssm_small(Vmamba_Backbone):
     def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )
