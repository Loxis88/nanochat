import torch.nn as nn

from .blocks import TransformerBlock
from .recurrence import AdaptiveRecurrentGlobalBlock

class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_kv_heads,
        total_layers=15,
        window_size=128,
        max_seq_len=8192,
        max_global_loops=8,
        min_global_loops=2,
        act_threshold=0.99,
        q_rank_ratio=0.5,
        use_lowrank_q=True,
        use_lowrank_out=False,
        ffn_mult=4,
    ):
        super().__init__()

        assert total_layers % 5 == 0, "total_layers должен быть кратен 5"

        self.total_layers = total_layers
        self.window_size = window_size

        self.embedding = nn.Embedding(vocab_size, d_model)

        if use_lowrank_q:
            d_latent_q = int(d_model * q_rank_ratio)
        else:
            d_latent_q = None

        if use_lowrank_out:
            d_latent_out = d_model // 2
        else:
            d_latent_out = None

        layers = []

        for layer_idx in range(total_layers):
            is_global = (layer_idx + 1) % 5 == 0

            block = TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_latent_q=d_latent_q,
                d_latent_out=d_latent_out,
                max_seq_len=max_seq_len,
                ffn_mult=ffn_mult,
            )

            if is_global:
                block = AdaptiveRecurrentGlobalBlock(
                    block=block,
                    d_model=d_model,
                    max_loops=max_global_loops,
                    min_loops=min_global_loops,
                    act_threshold=act_threshold,
                    halt_bias=-2.0,
                )

            layers.append(block)

        self.layers = nn.ModuleList(layers)

        self.final_norm = nn.RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embedding.weight

        for block in self.layers:
          if isinstance(block, AdaptiveRecurrentGlobalBlock):
            nn.init.constant_(block.halt_head.bias, -2.0)

    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              nn.init.zeros_(module.bias)

      elif isinstance(module, nn.Embedding):
          nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, return_aux_loss=False, return_stats=False):
        x = self.embedding(input_ids)

        aux_loss = x.new_tensor(0.0)
        all_stats = []

        for layer_idx, block in enumerate(self.layers):
            is_global = (layer_idx + 1) % 5 == 0

            if is_global:
                if return_stats:
                    x, ponder_loss, stats = block(
                        x,
                        window_size=self.window_size,
                        return_stats=True,
                    )
                    all_stats.append(stats)
                else:
                    x, ponder_loss = block(
                        x,
                        window_size=self.window_size,
                    )

                aux_loss = aux_loss + ponder_loss

            else:
                x = block(
                    x,
                    is_local=True,
                    window_size=self.window_size,
                )

        x = self.final_norm(x)
        # logits = self.lm_head(x)

        if return_aux_loss and return_stats:
            return x, aux_loss, all_stats

        if return_aux_loss:
            return x, aux_loss

        return x