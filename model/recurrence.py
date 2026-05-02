import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers.rms_norm import LigerRMSNorm

class StableLTIInjection(nn.Module):
    def __init__(
        self,
        d_model,
        init_dt=-3.0,
        init_delta_gate=-4.0,
    ):
        super().__init__()

        self.log_decay = nn.Parameter(torch.zeros(d_model))
        self.log_dt = nn.Parameter(torch.tensor(init_dt))

        self.e_norm = LigerRMSNorm(d_model)
        self.delta_norm = LigerRMSNorm(d_model)

        self.delta_gate = nn.Parameter(
            torch.full((d_model,), init_delta_gate)
        )

    def forward(self, h, e, delta):
        decay_rate = F.softplus(self.log_decay) + 1e-6
        dt = torch.exp(self.log_dt)

        A = torch.exp(-dt * decay_rate)
        B = 1.0 - A

        e = self.e_norm(e)
        delta = self.delta_norm(delta)

        delta_gate = torch.sigmoid(self.delta_gate)

        return A * h + B * e + B * delta_gate * delta


class AdaptiveRecurrentGlobalBlock(nn.Module):
    def __init__(
        self,
        block,
        d_model,
        max_loops=8,
        min_loops=2,
        act_threshold=0.99,
        halt_bias=-2.0,
    ):
        super().__init__()

        assert max_loops >= 1
        assert min_loops >= 1
        assert min_loops <= max_loops

        self.block = block
        self.d_model = d_model

        self.max_loops = max_loops
        self.min_loops = min_loops
        self.act_threshold = act_threshold

        self.loop_embedding = nn.Embedding(max_loops, d_model)
        self.loop_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.normal_(self.loop_embedding.weight, mean=0.0, std=0.02)

        self.injection = StableLTIInjection(d_model)

        self.halt_norm = LigerRMSNorm(d_model)
        self.halt_head = nn.Linear(d_model, 1)

        nn.init.constant_(self.halt_head.bias, halt_bias)

    def forward(self, x, window_size=128, return_stats=False):
        B, T, C = x.shape

        e = x
        h = x

        halting = x.new_zeros(B, T, 1)
        output = torch.zeros_like(x)

        raw_ponder_loss = x.new_tensor(0.0)

        used_loops = x.new_zeros(B, T, 1)

        loop_ids = torch.arange(self.max_loops, device=x.device)
        loop_embeddings = self.loop_embedding(loop_ids)

        for loop_idx in range(self.max_loops):
            still_running = (halting < self.act_threshold).to(x.dtype)

            old_h = h

            loop_emb = loop_embeddings[loop_idx].view(1, 1, C)

            z = h + self.loop_scale * loop_emb

            candidate = self.block(
                z,
                is_local=False,
                window_size=window_size,
            )
            delta = candidate - z

            new_h = self.injection(
                h=h,
                e=e,
                delta=delta,
            )

            h = still_running * new_h + (1.0 - still_running) * old_h

            if loop_idx < self.min_loops - 1:
                p_halt = x.new_zeros(B, T, 1)
            else:
                p_halt = torch.sigmoid(
                    self.halt_head(self.halt_norm(h))
                )

            remaining = (self.act_threshold - halting).clamp_min(0.0)

            if loop_idx == self.max_loops - 1:
                weight = remaining
            else:
                weight = torch.minimum(p_halt, remaining)

            weight = weight * still_running

            output.add_(weight * h)  # in-place, без аллокации
            halting = halting + weight

            used_loops = used_loops + still_running
            if loop_idx >= self.min_loops - 1:
                raw_ponder_loss.add_((still_running * (1.0 - p_halt)).mean())

            if not self.training and loop_idx >= self.min_loops - 1:
                if halting.min() >= self.act_threshold:  # min() быстрее all()
                    break

        output = output / self.act_threshold

        if return_stats:
            stats = {
                "mean_loops": used_loops.mean().detach(),
                "max_loops_used": used_loops.max().detach(),
                "mean_halting": halting.mean().detach(),
            }
            return output, raw_ponder_loss, stats

        return output, raw_ponder_loss