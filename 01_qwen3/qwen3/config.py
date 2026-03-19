import torch
def get_qwen3_config(choose_model):
    configs = {
        "0.6B":{
        "vocab_size": 151_936,           # Vocabulary size
        "context_length": 40_960,        # Context length that was used to train the model
        "emb_dim": 1024,                 # Embedding dimension
        "n_heads": 16,                   # Number of attention heads
        "n_layers": 28,                  # Number of layers
        "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
        "head_dim": 128,                 # Size of the heads in GQA
        "qk_norm": True,                 # Whether to normalize queries and keys in GQA
        "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
        "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
        },

        "1.7B":{
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 2048,                 # 2x larger than above
        "n_heads": 16,
        "n_layers": 28,
        "hidden_dim": 6144,              # 2x larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
        },   

        "4B":{
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 2560,                 # 25% larger than above
        "n_heads": 32,                   # 2x larger than above
        "n_layers": 36,                  # 29% larger than above
        "hidden_dim": 9728,              # ~3x larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
        },

        "8B":{
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 4096,                 # 60% larger than above
        "n_heads": 32,
        "n_layers": 36,                  # 26% larger than above
        "hidden_dim": 12288,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
        },

        "14B":{
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,                 # 25% larger than above
        "n_heads": 40,                   # 25% larger than above
        "n_layers": 40,                  # 11% larger than above
        "hidden_dim": 17408,             # 42% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
    },
    "32B":{
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,                
        "n_heads": 64,                   # 60% larger than above
        "n_layers": 64,                  # 60% larger than above
        "hidden_dim": 25600,             # 47% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
        }
    }
    if choose_model not in configs:
        raise ValueError(f"{choose_model} is not supported. Choose from {list(configs.keys())}")
    
    return configs[choose_model]