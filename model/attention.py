import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from .rope import precompute_freqs_cis, apply_rotary_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm


class LowRankAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        d_latent_q=None,
        d_latent_out=None,
        max_seq_len=8192,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0

        # Комплексный RoPE — один буфер вместо двух
        freqs_cis = precompute_freqs_cis(dim=self.head_dim,end=max_seq_len,)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        

        if d_latent_q is None or d_latent_q >= d_model:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.use_lowrank_q = False
        else:
            self.q_down = nn.Linear(d_model, d_latent_q, bias=False)
            self.q_up = nn.Linear(d_latent_q, d_model, bias=False)
            self.use_lowrank_q = True

        kv_dim = n_kv_heads * self.head_dim
        self.kv_proj = nn.Linear(d_model, 2 * kv_dim, bias=False)

        self.q_norm = LigerRMSNorm(self.head_dim)
        self.k_norm = LigerRMSNorm(self.head_dim)

        if d_latent_out is None or d_latent_out >= d_model:
            self.o_proj = nn.Linear(d_model, d_model, bias=False)
            self.use_lowrank_out = False
        else:
            self.o_down = nn.Linear(d_model, d_latent_out, bias=False)
            self.o_up = nn.Linear(d_latent_out, d_model, bias=False)
            self.use_lowrank_out = True

    def forward(self, x, is_local=False, window_size=128):
        B, T, C = x.shape

        if self.use_lowrank_q:
            q = self.q_up(F.silu(self.q_down(x)))
        else:
            q = self.q_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim)
        q = self.q_norm(q)

        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
        k, v = kv.unbind(2)
        v = v.contiguous() 
        k = self.k_norm(k)

        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])  # [B,T,H,D]

        if q.dtype not in (torch.float16, torch.bfloat16):
            flash_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q = q.to(flash_dtype)
            k = k.to(flash_dtype)
            v = v.to(flash_dtype)

        ws = (window_size - 1, 0) if is_local else (-1, -1)
        attn_out = flash_attn_func(q, k, v, causal=True, window_size=ws)

        attn_out = attn_out.view(B, T, C)

        if self.use_lowrank_out:
            return self.o_up(self.o_down(attn_out))
        return self.o_proj(attn_out)