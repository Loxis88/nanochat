import math

def get_lr(step: int, warmup_steps: int, max_steps: int, learning_rate: float, min_lr: float):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + cosine * (learning_rate - min_lr)