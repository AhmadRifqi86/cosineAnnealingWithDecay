# CosineAnnealingWarmRestartsWithDecay Scheduler

This is a custom PyTorch learning rate scheduler that extends the idea of **Cosine Annealing with Warm Restarts (SGDR)** by introducing:

- 🔻 **Decaying maximum learning rate** after each cycle
- 🔄 **Shortening cycle lengths** over time
- ✅ Easy to integrate with any PyTorch optimizer

---

## 📌 Features

- Cosine-shaped learning rate decay within each cycle
- Warm restarts (resets the cosine curve after each cycle)
- Multiplies max learning rate by `decay` after each cycle
- Shrinks cycle length using `freq_mult` (e.g. 0.9 means shorter cycle next round)

---

## 📦 Installation

No installation required. Just copy the following class into your training script or into a `schedulers.py` file.

---

## 🚀 Usage

```python
from schedulers import CosineAnnealingWarmRestartsWithDecay
import torch

# Your optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Scheduler
scheduler = CosineAnnealingWarmRestartsWithDecay(
    optimizer,
    T_0=10,          # Initial cycle length (in epochs or steps)
    T_mult=1,        # Optional: Multiply cycle length (default 1 = constant length)
    eta_min=1e-6,    # Minimum learning rate
    decay=0.9,       # Decay factor for max LR each cycle
    freq_mult=0.9    # Shrinks cycle length (e.g., 0.9 means next cycle is 90% as long)
)

# In training loop
for epoch in range(num_epochs):
    train(...)
    scheduler.step()  # Call once per epoch or per step depending on use
