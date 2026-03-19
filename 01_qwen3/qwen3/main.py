import torch
from config import get_qwen3_config
from model import Qwen3Model

CHOOSE_MODEL = "0.6B"
QWEN3_CONFIG = get_qwen3_config(CHOOSE_MODEL)

torch.manual_seed(123)
model = Qwen3Model(QWEN3_CONFIG)

print(f"成功加载 {CHOOSE_MODEL} 配置，Embedding 维度为: {QWEN3_CONFIG['emb_dim']}")
