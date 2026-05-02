import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(end), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # [seq, dim//2]

def apply_rotary_emb(xq, xk, freqs_cis):
    dtype = xq.dtype
    freqs_cis = freqs_cis[None, :, None, :]  # [1, seq, 1, dim//2]
    xq_c = torch.view_as_complex(xq.float().contiguous().view(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().contiguous().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(3)
    return xq_out.to(dtype), xk_out.to(dtype)