import numpy as np
import paddle
import os

def precision_cmp_paddle(t1: paddle.Tensor, t2: paddle.Tensor):
    
    x, xx = paddle.cast(t1, dtype='float32'), paddle.cast(t2, dtype='float32')
    # 重塑张量并计算余弦相似度
    x_reshaped = paddle.reshape(x, [1, -1])
    xx_reshaped = paddle.reshape(xx, [1, -1])
    sim = paddle.nn.functional.cosine_similarity(x_reshaped, xx_reshaped).item()
    
    # 计算 L1 误差
    l1 = (paddle.abs(x - xx).sum() / paddle.abs(xx).sum()).item()
    max_diff = paddle.max(x - xx)
    
    return sim, l1, max_diff

TORCH_DIR = "torch_results"
PADDLE_DIR = "paddle_results"

q_paddle = np.load(os.path.join(PADDLE_DIR, "q.npy"))
k_paddle = np.load(os.path.join(PADDLE_DIR, "k.npy"))
v_paddle = np.load(os.path.join(PADDLE_DIR, "v.npy"))

q_torch = np.load(os.path.join(TORCH_DIR, "q.npy"))
k_torch = np.load(os.path.join(TORCH_DIR, "k.npy"))
v_torch = np.load(os.path.join(TORCH_DIR, "v.npy"))

q_paddle = paddle.to_tensor(q_paddle, dtype=paddle.bfloat16)
k_paddle = paddle.to_tensor(k_paddle, dtype=paddle.bfloat16)
v_paddle = paddle.to_tensor(v_paddle, dtype=paddle.bfloat16)

q_th_paddle = paddle.to_tensor(q_torch, dtype=paddle.bfloat16)
k_th_paddle = paddle.to_tensor(k_torch, dtype=paddle.bfloat16)
v_th_paddle = paddle.to_tensor(v_torch, dtype=paddle.bfloat16)

sim, _, maxd = precision_cmp_paddle(q_paddle, q_th_paddle)
print(f"sim: {sim}, max diff: {maxd}")
sim, _, maxd = precision_cmp_paddle(k_paddle, k_th_paddle)
print(f"sim: {sim}, max diff: {maxd}")
sim, _, maxd = precision_cmp_paddle(v_paddle, v_th_paddle)
print(f"sim: {sim}, max diff: {maxd}")