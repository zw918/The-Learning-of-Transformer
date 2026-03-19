import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
    
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        # SiLU 的输出在 0 到 1 之间波动，决定了哪些信息可以通过
            # 逐元素相乘：门控分支对数据分支进行“过滤”，只有重要的特征会被放大，不重要的被抑制
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)
