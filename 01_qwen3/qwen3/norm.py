import torch
import torch.nn as nn

#  学习参数    scale   shift
# 先得到  平均的权重，  再  *  scale

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-6, bias = False, qwen3__compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3__compatible = qwen3__compatible
        # 会随训练进化的参数：nn.Parameter(torch.ones(10))
        # 普通的常数数据：torch.ones(10)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None
    
    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3__compatible:
            x = x.to(torch.float32)
        
        var = x.pow(2).mean(dim=-1, keepdim= True)
        norm_x = x * torch.rsqrt(var + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)
        
