from  VMamba.kernels.vmamba import CrossMambaFusion_SS2D_SSM,ChannelAttentionBlock,SaliencyMambaBlock
from basebone_vmamba import vssm_small as backbone
import torch.nn as nn
import torch
from  VMamba.kernels.vmamba import LayerNorm,Mlp
from  Registration import PRFI,SR
from HA import HA
import torch.nn.functional as F
from MambaDecoder import MambaDecoder
from CA_BPM import CABoundaryModule

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder=encoder_Block()
        self.channels = self.channels = [96, 192, 384, 768]
        self.decoder = MambaDecoder(img_size=[448, 448],
                                    in_channels=self.channels, 
                                    num_classes=1, 
                                    depths=[4, 4, 4, 4],
                                    embed_dim=self.channels[0], 
                                    deep_supervision=False)
    def forward(self, rgb, t):
        orisize = rgb.shape
        x, saliency, CR = self.encoder(rgb, t)
        out = self.decoder.forward(x,CR)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        saliency = F.interpolate(saliency, size=orisize[2:], mode='bilinear', align_corners=False)
        return out, saliency
    
    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

dims =96
class CrossMamba_(nn.Module):
    def __init__(self, dim):
        super(CrossMamba_, self).__init__()
        self.cross_mamba = CrossMambaFusion_SS2D_SSM(dim)
        self.dynamic_gate = DynamicGate(dim)
        self.CA = ChannelAttentionBlock(dim) #channel attention
        self.norm= LayerNorm(dim)
        self.mlp= Mlp(in_features=dim,hidden_features=dim*4, out_features=dim)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim), 
            nn.ReLU(inplace=True) 
        )
        self.prfi = PRFI(dim)
        self.sr = SR(CA=True, dim=dim)
        self.reduce = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)


    def forward(self, rgb, t):  
         rgb_in, t_in = rgb, t
         rgb,t=self.prfi(rgb, t)
         rgb = rgb.permute(0, 2, 3, 1).contiguous()
         t = t.permute(0, 2, 3, 1).contiguous()
         rgb_,t_=self.cross_mamba(rgb,t)
         rgb_ = rgb_.permute(0, 3, 1, 2).contiguous()
         t_ = t_.permute(0, 3, 1, 2).contiguous()
         rgb_, t_ = self.sr(rgb_, t_)
         alpha, beta = self.dynamic_gate(rgb_, t_)
         rgb_out = rgb_in + (alpha * rgb_)
         t_out = t_in + (beta * t_)
         cat_feat = torch.cat([rgb_out, t_out], dim=1) # [B, 2C, H, W]
         x = self.reduce(cat_feat)
         x = x.permute(0, 2, 3, 1).contiguous()
         x = x.permute(0, 3, 1, 2).contiguous()# [B, C, H, W]
         return x

class DynamicGate(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #  MLP: Input 2*dim, Output 2*dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim * 2, bias=False),
            nn.Sigmoid() # Output 0~1 
        )

    def forward(self, x_rgb, x_t):
        B, C, H, W = x_rgb.shape
        
        # Squeeze: [B, C]
        rgb_vector = self.avg_pool(x_rgb).view(B, C)
        t_vector = self.avg_pool(x_t).view(B, C)
        
        # Concat : [B, 2C]
        joint_vector = torch.cat([rgb_vector, t_vector], dim=1)
        
        # Excitation: [B, 2C]
        gates = self.gate_mlp(joint_vector)
        
        # Split : [B, C]
        alpha, beta = torch.split(gates, C, dim=1)
        
        # Reshape for broadcast: [B, C, 1, 1]
        return alpha.view(B, C, 1, 1), beta.view(B, C, 1, 1)
         
         

class encoder_Block(nn.Module):
    def __init__(self):
        super(encoder_Block, self).__init__()
        self.vssm_r = backbone()
        self.dim = 96
        self.fusion_stages = nn.ModuleList([
      CrossMamba_(dim=dims * (2 ** i)) for i in range(4)
  ])
        self.pred_saliency = nn.Conv2d(in_channels=dims * 8, out_channels=1, kernel_size=1, bias=False)
        self.ha = HA()
        self.saliency_mamba = nn.ModuleList(
            SaliencyMambaBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
                lip = False
            ) for i in range(4)
        )
        self.cr_modules = nn.ModuleList([CABoundaryModule(dims * (2 ** i), dims * (2 ** i)) for i in range(4)])

    def forward(self, rgb, t):
        outs_fused = []
        outs_rgb = self.vssm_r(rgb) 
        outs_t = self.vssm_r(t)
        CR=self.CR(outs_rgb,outs_t)
        deep_feat_rgb = outs_rgb[3]
        deep_feat_t= outs_t[3]
        saliency = self.pred_saliency(F.interpolate(deep_feat_rgb, scale_factor=8, mode='bilinear', align_corners=False))
        saliency_t = self.pred_saliency(F.interpolate(deep_feat_t, scale_factor=8, mode='bilinear', align_corners=False))
        for i in range(4):
            curr_rgb = outs_rgb[i].permute(0, 2, 3, 1).contiguous()
            curr_t = outs_t[i].permute(0, 2, 3, 1).contiguous()
            B,H,W,C = curr_rgb.shape
            B,H_t,W_t,C_t = curr_t.shape
            resized_gt_rgb = F.interpolate(saliency, size=(H, W), mode='bilinear', align_corners=False)
            resized_gt_t = F.interpolate(saliency_t, size=(H_t,W_t), mode='bilinear', align_corners=False)
            resized_gt_rgb = torch.sigmoid(resized_gt_rgb)
            resized_gt_t = torch.sigmoid(resized_gt_t)
            curr_rgb = self.saliency_mamba[i](curr_rgb, resized_gt_rgb)
            curr_t = self.saliency_mamba[i](curr_t, resized_gt_t)
            curr_rgb = curr_rgb.permute(0, 3, 1, 2).contiguous()  
            curr_t = curr_t.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
            fused_feat = self.fusion_stages[i](curr_rgb, curr_t)
            outs_fused.append(fused_feat)
        guide_saliency = torch.nn.Sigmoid()(saliency)
        guide_saliency = self.ha(guide_saliency)
        return outs_fused, saliency, CR
    def CR(self,rgb,t):
        CR=[]
        for i in range(4):
            rgb = rgb[i]
            t = t[i]
            rgb = rgb.permute(0, 3, 1, 2).contiguous()
            t = t.permute(0, 3, 1, 2).contiguous()
            fuse=rgb+t
            cr=self.cr_modules[i](rgb,t,fuse)
            CR.append(cr)
        return CR


if __name__ == '__main__':
    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = flow = depth = torch.ones([2, 3, 448, 448]).to(device)
    out = model(image, flow)
    print(model)
    print(out.shape)
