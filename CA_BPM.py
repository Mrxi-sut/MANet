import torch.nn as nn
import torch


class CABoundaryModule(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        self.edge_extractor_rgb = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.edge_extractor_t = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        
        reduction = max(dim_out // 4, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_out * 2 + dim_in, reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, dim_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.refine_proj = nn.Conv2d(dim_out, dim_in, kernel_size=1)


    def forward(self, F_rgb, F_t, F_guide):
        

        E_rgb = self.edge_extractor_rgb(F_rgb)
        E_t = self.edge_extractor_t(F_t)
        

        combined_context = torch.cat([E_rgb, E_t, F_guide], dim=1) 
        
        

        W = self.channel_gate(combined_context) # W: [B, dim_out, 1, 1]
        E_fused = W * E_rgb + (1 - W) * E_t
        residual = self.refine_proj(E_fused)
        
  
        return F_guide+residual
    

