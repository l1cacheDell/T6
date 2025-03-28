import math
from dataclasses import dataclass
from typing import Optional, Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn.initializer import XavierUniform

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    head_dim: int = -1
    q_rank: int = 12
    rank: int = 2
    using_groupnorm: bool = False

class T6GroupNorm(nn.Layer):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = self.create_parameter(
                shape=[dim],
                default_initializer=nn.initializer.Constant(1.0)
            )
        else:
            self.weight = None

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.astype('float32')).astype(x.dtype)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (paddle.arange(0, dim, 2)[: (dim // 2)].astype('float32') / dim))
    t = paddle.arange(end, dtype='float32')
    freqs = paddle.outer(t, freqs)
    freqs_cis = paddle.complex(paddle.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: paddle.Tensor, x: paddle.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(shape)

def apply_rotary_emb(
    xq: paddle.Tensor,
    xk: paddle.Tensor,
    freqs_cis: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    xq_ = paddle.as_complex(xq.astype('float32').reshape(*xq.shape[:-1], -1, 2))
    xk_ = paddle.as_complex(xk.astype('float32').reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = paddle.as_real(xq_ * freqs_cis).flatten(3)
    xk_out = paddle.as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)

class TPA(nn.Layer):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim if args.head_dim > 0 else args.dim // args.n_heads
        self.n_head = args.n_heads
        self.q_rank = args.q_rank
        self.rank = args.rank
        self.dim = args.dim
        self.using_groupnorm = args.using_groupnorm
        
        self.W_A_q = nn.Linear(args.dim, self.n_head * self.q_rank, bias_attr=False)
        self.W_A_k = nn.Linear(args.dim, self.n_head * self.rank, bias_attr=False)
        self.W_A_v = nn.Linear(args.dim, self.n_head * self.rank, bias_attr=False)

        self.W_B_q = nn.Linear(args.dim, self.q_rank * self.head_dim, bias_attr=False)
        self.W_B_k = nn.Linear(args.dim, self.rank * self.head_dim, bias_attr=False)
        self.W_B_v = nn.Linear(args.dim, self.rank * self.head_dim, bias_attr=False)
        
        self.cache_kA = paddle.zeros([args.max_batch_size, args.max_seq_len, self.n_heads, self.rank])
        self.cache_vA = paddle.zeros([args.max_batch_size, args.max_seq_len, self.n_heads, self.rank])
        self.cache_kB = paddle.zeros([args.max_batch_size, args.max_seq_len, self.rank, self.head_dim])
        self.cache_vB = paddle.zeros([args.max_batch_size, args.max_seq_len, self.rank, self.head_dim])
        
        self.reset_parameters(args)

        if self.using_groupnorm:
            self.subln = T6GroupNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def reset_parameters(self, args):
        W_A_q_tensor = self.W_A_q.weight.view([self.dim, self.n_head, self.q_rank])
        W_A_k_tensor = self.W_A_k.weight.view([self.dim, self.n_head, self.rank])
        W_A_v_tensor = self.W_A_v.weight.view([self.dim, self.n_head, self.rank])
        XavierUniform()(W_A_q_tensor)
        XavierUniform()(W_A_k_tensor)
        XavierUniform()(W_A_v_tensor)
        self.W_A_q.weight.set_value(W_A_q_tensor.reshape([-1]))
        self.W_A_k.weight.set_value(W_A_k_tensor.reshape([-1]))
        self.W_A_v.weight.set_value(W_A_v_tensor.reshape([-1]))

        W_B_q_tensor = self.W_B_q.weight.reshape([self.dim, self.q_rank, self.head_dim])
        W_B_k_tensor = self.W_B_k.weight.reshape([self.dim, self.rank, self.head_dim])
        W_B_v_tensor = self.W_B_v.weight.reshape([self.dim, self.rank, self.head_dim])
        XavierUniform()(W_B_q_tensor)
        XavierUniform()(W_B_k_tensor)
        XavierUniform()(W_B_v_tensor)
        self.W_B_q.weight.set_value(W_B_q_tensor.reshape([-1]))
        self.W_B_k.weight.set_value(W_B_k_tensor.reshape([-1]))
        self.W_B_v.weight.set_value(W_B_v_tensor.reshape([-1]))
        
    def forward(self, 
        x: paddle.Tensor, 
        start_pos: int, 
        freqs_cis: paddle.Tensor, 
        mask=None):
        bsz, seqlen, _ = x.shape

        A_q = self.W_A_q(x).reshape([bsz, seqlen, self.n_head, self.q_rank])
        A_k = self.W_A_k(x).reshape([bsz, seqlen, self.n_head, self.rank])
        A_v = self.W_A_v(x).reshape([bsz, seqlen, self.n_head, self.rank])

        B_q = self.W_B_q(x).reshape([bsz, seqlen, self.q_rank, self.head_dim])
        B_k = self.W_B_k(x).reshape([bsz, seqlen, self.rank, self.head_dim])
        B_v = self.W_B_v(x).reshape([bsz, seqlen, self.rank, self.head_dim])

        B_q, B_k = apply_rotary_emb(B_q, B_k, freqs_cis=freqs_cis)
        
        # Update caches
        self.cache_kA = self.cache_kA.astype(A_k.dtype)
        self.cache_vA = self.cache_vA.astype(A_v.dtype)
        self.cache_kA[:bsz, start_pos:start_pos + seqlen] = A_k
        self.cache_vA[:bsz, start_pos:start_pos + seqlen] = A_v
        
        A_k = self.cache_kA[:bsz, :start_pos+seqlen]
        A_v = self.cache_vA[:bsz, :start_pos+seqlen]
        
        self.cache_kB = self.cache_kB.astype(B_k.dtype)
        self.cache_vB = self.cache_vB.astype(B_v.dtype)
        self.cache_kB[:bsz, start_pos:start_pos+seqlen] = B_k
        self.cache_vB[:bsz, start_pos:start_pos+seqlen] = B_v
        
        B_k = self.cache_kB[:bsz, :start_pos+seqlen]
        B_v = self.cache_vB[:bsz, :start_pos+seqlen]
        
        # Reshape and compute attention
        A_q = A_q.reshape([bsz * seqlen, self.n_head, self.q_rank])
        A_k = A_k.reshape([bsz * seqlen, self.n_head, self.rank])
        A_v = A_v.reshape([bsz * seqlen, self.n_head, self.rank])

        B_q = B_q.reshape([bsz * seqlen, self.q_rank, self.head_dim])
        B_k = B_k.reshape([bsz * seqlen, self.rank, self.head_dim])
        B_v = B_v.reshape([bsz * seqlen, self.rank, self.head_dim])
        
        q = paddle.bmm(A_q, B_q).div_(self.q_rank).reshape([bsz, seqlen, self.n_head, self.head_dim])
        k = paddle.bmm(A_k, B_k).div_(self.rank).reshape([bsz, seqlen, self.n_head, self.head_dim])
        v = paddle.bmm(A_v, B_v).div_(self.rank).reshape([bsz, seqlen, self.n_head, self.head_dim])
        
        k = k.transpose([0, 2, 1, 3])
        # compute q @ k^T
        scores = paddle.matmul(q.transpose([0, 2, 1, 3]), k.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.astype('float32'), axis=-1).astype(q.dtype)
        output = paddle.matmul(scores, v.transpose([0, 2, 1, 3]))
        output = output.transpose([0, 2, 1, 3]).reshape([bsz, seqlen, -1])
        return output

class T6MLP(nn.Layer):
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = math.floor(8 / 3 * dim)
        self.c_fc1 = nn.Linear(dim, hidden_dim, bias_attr=False)
        self.c_fc2 = nn.Linear(dim, hidden_dim, bias_attr=False)
        self.c_proj = nn.Linear(hidden_dim, dim, bias_attr=False)
        self.c_proj.weight.set_value(paddle.zeros_like(self.c_proj.weight))

    def forward(self, x):
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        x = F.silu(x1) * x2
        return self.c_proj(x)

class T6Block(nn.Layer):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = TPA(args)
        self.feed_forward = T6MLP(dim=args.dim)
        self.layer_id = layer_id

    def forward(self, x, start_pos, freqs_cis, mask=None):
        h = x + self.attention(F.rms_norm(x, [x.shape[-1]]), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(F.rms_norm(h, [h.shape[-1]]))
        return out

class T6Model(nn.Layer):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.LayerList([T6Block(layer_id, params) for layer_id in range(params.n_layers)])
        self.output = nn.Linear(params.dim, params.vocab_size, bias_attr=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @paddle.no_grad()
    def forward(self, tokens, start_pos):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.astype(h.dtype)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]

        mask = None
        if seqlen > 1:
            mask = paddle.full([seqlen, seqlen], float('-inf'), dtype=h.dtype)
            mask = paddle.triu(mask, diagonal=1)
            mask = paddle.concat(
                [paddle.zeros([seqlen, start_pos], dtype=h.dtype), mask],
                axis=1)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = F.rms_norm(h, [h.shape[-1]])
        output = self.output(h).astype('float32')
        return output