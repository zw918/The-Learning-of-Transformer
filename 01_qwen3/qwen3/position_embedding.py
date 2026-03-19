import torch
# 本质：RoPE 是把“第几个词”变成了“向量转了多少度”。
# 两两配对：我们把高维向量拆成一对对的小平面，每一对都在自己的平面里转圈。
# 计算技巧：代码里通过 cat(-x2, x1) 构造出一个“镜像向量”，从而用最简单的加减乘法代替了复杂的旋转矩阵计算。
def compute_rope_params(head_dim, theta_base=10000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[:(head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)

    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimesion must be even"
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    # 预计算的 cos/sin 是按照最大长度 context_length（比如 4096）算的，但你当前的句子可能只有 512 个词。所以要取前 seq_len 行，使其与输入 x 的长度对齐
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)
