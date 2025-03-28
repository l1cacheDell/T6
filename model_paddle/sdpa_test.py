import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def precision_cmp_torch(t1: torch.Tensor, t2: torch.Tensor):
    x, xx = t1.to(dtype=torch.float32), t2.to(dtype=torch.float32)
    # 重塑张量并计算余弦相似度
    x_reshaped = torch.reshape(x, [1, -1])
    xx_reshaped = torch.reshape(xx, [1, -1])
    sim = torch.nn.functional.cosine_similarity(x_reshaped, xx_reshaped).item()
    
    # 计算 L1 误差
    l1 = (torch.abs(x - xx).sum() / torch.abs(xx).sum()).item()

    max_diff = torch.max(x - xx)
    # print("Max Diff: ", max_diff.item())
    
    return sim, l1, max_diff

def test_attention_equivalence():
    # 配置参数
    bsz = 2
    seq_len = 16
    num_heads = 32
    head_dim = 128
    dtype = torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化QKV (使用正态分布生成随机值)
    torch.manual_seed(42)
    q = torch.randn(bsz, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(bsz, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(bsz, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    
    # 生成因果掩码
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = torch.hstack(
        [torch.zeros((seq_len, 0), device="cuda"), mask]
    ).type_as(q)
    
    # 方法1: 使用SDPA计算
    q_sdpa = q.transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        o_sdpa = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=mask,
            scale=math.sqrt(head_dim),
        )
    
    # 方法2: 手动计算
    k_transposed = k.transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
    q_transposed = q.transpose(1, 2)
    
    # 计算注意力分数
    scores = torch.matmul(q_transposed, k_transposed.transpose(-2, -1)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask
    scores = F.softmax(scores, dim=-1).type_as(q)
    output = torch.matmul(scores, v.transpose(1, 2))

    print(o_sdpa.shape, output.shape)
    
    # 比较结果
    if torch.allclose(o_sdpa, output, rtol=1e-5, atol=1e-8):
        print("✅ 两种实现结果一致")
    else:
        sim, l1, max_diff = precision_cmp_torch(o_sdpa, output)
        print(f"❌ 结果存在差异 - 相似度: {sim:.6f}, L1误差: {l1:.6f}, 最大差异: {max_diff:.6f}")
    
    # 打印形状信息验证
    print("\n张量形状验证:")
    print(f"Q shape: {q.shape}")
    print(f"SDPA输出形状: {o_sdpa.shape}")
    print(f"手动计算输出形状: {output.shape}")
    
    return o_sdpa, output

if __name__ == "__main__":
    # 运行测试
    o_sdpa, output = test_attention_equivalence()
    