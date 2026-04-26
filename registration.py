import torch
from torch import nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN, InPlaceABNSync

class PR(nn.Module):
    def __init__(self, features):
        super(PR, self).__init__()
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            InPlaceABNSync(features),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )
        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            InPlaceABNSync(features),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )
        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()
    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w / s, h / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid,align_corners=True)
        return output
    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]]).type_as(input).to(input.device)  # not [h/s, w/s]
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=True)
        return output
    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)
        return low_stage, high_stage
    


class PRFI(nn.Module):
    def __init__(self, dim):
        super(PRFI, self).__init__()
        self.PR = PR(dim)
        self.mlp_pool = Feature_Pool(dim)
        self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7, padding=3, groups=dim*2)
        self.cse = Channel_Attention(dim*2)
        self.sse_r = Spatial_Attention(dim)
        self.sse_t = Spatial_Attention(dim)
    def forward(self, RGB, T):
        ##########################pixel-level registration#####################################
        R1, T1 = self.PR(RGB, T)
        RGB = R1 + RGB
        T = T1 + T
        b, c, h, w = RGB.size()
        New_RGB_A = RGB
        New_T_A = T
        x_cat = torch.cat((New_RGB_A,New_T_A),dim=1)
        fuse_gate = torch.sigmoid(self.cse(self.dwconv(x_cat)))
        rgb_gate, t_gate = fuse_gate[:, 0:c, :], fuse_gate[:, c:c * 2, :]
        New_RGB = RGB + New_RGB_A * rgb_gate
        New_T = T + New_T_A * t_gate
        ##########################################################################
        New_fuse_RGB = self.sse_r(New_RGB)
        New_fuse_T = self.sse_t(New_T)
        attention_vector = torch.cat([New_fuse_RGB, New_fuse_T], dim=1)
        attention_vector = torch.softmax(attention_vector,dim=1)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        New_RGB = New_RGB * attention_vector_l
        New_T = New_T * attention_vector_r
        ##########################################################################
        return New_RGB, New_T
    
class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1,bias=True)
    def forward(self, x):
        x1 = self.conv1(x)
        return x1
    
class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gap_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)
    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2)
        return max_out
    
class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=2):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2).view(b,c)
        return y

class SR(nn.Module):
    def __init__(self, CA=True, dim=256):
        super(SR, self).__init__()
        self.CA = CA
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        if self.CA:
            self.att1 = CrossAttention(in_channel=dim)
            self.att2 = CrossAttention(in_channel=dim)
        else:
            print("Warning: not use CrossAttention!")
            self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
            self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
    def forward(self, rgb, t):
        if self.CA:
            feat_1 = self.att1(rgb, t)
            feat_2 = self.att2(t, rgb)
        else:
            w1 = self.conv2(rgb)
            w2 = self.conv3(t)
            feat_1 = F.relu(w2*rgb, inplace=True)
            feat_2 = F.relu(w1*t, inplace=True)
        out1 = rgb + feat_1
        out2 = t + feat_2
        return out1, out2

class CrossAttention(nn.Module):
    def __init__(self, in_channel=256, ratio=1):
        super(CrossAttention, self).__init__()
        self.conv_k = nn.Conv2d(in_channel, in_channel, kernel_size=ratio, stride=ratio)
        self.conv_v = nn.Conv2d(in_channel, in_channel, kernel_size=ratio, stride=ratio)
        self.conv_q = nn.Conv2d(in_channel, in_channel, kernel_size=1)
    def forward(self, rgb, depth):
        bz, c, h, w = rgb.shape
        rgb_q = self.conv_q(rgb).view(bz, -1, h*w).permute(0, 2, 1)
        depth_k = self.conv_k(depth).view(bz, c, -1)
        depth_v = self.conv_v(depth).view(bz, c, -1).permute(0, 2, 1)
        mask = torch.bmm(rgb_q, depth_k)
        mask = torch.softmax(mask, dim=-1)
        feat = torch.bmm(mask, depth_v)
        feat = rgb_q + feat
        feat = feat.permute(0, 2, 1).view(bz, -1, h, w)
        return feat
