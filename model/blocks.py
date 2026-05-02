import torch.nn as nn
import torch.nn.functional as F

from .attention import LowRankAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        d_latent_q=None,
        d_latent_out=None,
        max_seq_len=8192,
        ffn_mult=4,
    ):
        super().__init__()

        self.attn = LowRankAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_latent_q=d_latent_q,
            d_latent_out=d_latent_out,
            max_seq_len=max_seq_len,
        )

        self.ffn = FeedForward(d_model, (int(d_model * ffn_mult) + 63) // 64 * 64)

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x, is_local=False, window_size=128):
        x = x + self.attn(
            self.norm1(x),
            is_local=is_local,
            window_size=window_size,
        )

        x = x + self.ffn(self.norm2(x))
        return x