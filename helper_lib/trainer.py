from typing import Optional, Dict, List
import torch
from torch import nn
from .evaluator import evaluate_model
import torch.nn.functional as F
from .diffusion_utils import train_diffusion as _train_diffusion

def train_model(
    model: nn.Module,
    train_loader,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    epochs: int = 10,
    val_loader=None,
    log_interval: int = 100,
) -> Dict[str, List[float]]:
    """
    Generic training loop that returns a history dict with per-epoch stats.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % log_interval == 0:
                avg = running_loss / log_interval
                print(f"[Epoch {epoch:02d}] Batch {batch_idx:04d} | train loss {avg:.4f}")
                running_loss = 0.0

        # End of epoch: compute averages and validation
        epoch_train_loss = _average_epoch_loss(model, train_loader, criterion, device)
        history["train_loss"].append(epoch_train_loss)

        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"[Epoch {epoch:02d}] train {epoch_train_loss:.4f} | "
                  f"val {val_loss:.4f} | acc {val_acc:.2f}%")
        else:
            print(f"[Epoch {epoch:02d}] train {epoch_train_loss:.4f}")

    return history

@torch.no_grad()
def _average_epoch_loss(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += criterion(model(x), y).item()
        count += 1
    return total / max(count, 1)


# =========================
# VAE-specific training API
# =========================

# =========================
# VAE-specific training API
# =========================
from typing import Optional, Callable
import torch.nn.functional as F

def default_vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    """
    BCE (reconstruction) + beta * KL, averaged per sample.
    Assumes recon,x in [0,1] (decoder uses Sigmoid; inputs not mean/std normalized).
    """
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + beta * kl) / x.size(0)

def train_vae_model(
    model: nn.Module,
    train_loader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    epochs: int = 10,
    log_interval: int = 100,
    # Optional per-PDF style override; if None we use default_vae_loss
    criterion: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    beta: float = 1.0,
):
    """
    VAE training loop. Ignores labels (expects (x, y) from loader but discards y).
    If `criterion` is provided, it should be a callable: (recon, x, mu, logvar) -> loss.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = criterion if criterion is not None else (lambda r, x, mu, lv: default_vae_loss(r, x, mu, lv, beta))

    model.to(device)
    history = {"train_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0

        for batch_idx, (x, _) in enumerate(train_loader, start=1):
            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            loss = loss_fn(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

            if batch_idx % log_interval == 0:
                print(f"[VAE E{epoch:02d}] B{batch_idx:04d} | train loss {loss.item():.4f}")

        epoch_loss = running / max(seen, 1)
        history["train_loss"].append(epoch_loss)
        print(f"[VAE E{epoch:02d}] avg train loss {epoch_loss:.4f}")

    return history


# helper_lib/trainer.py
# import math
# import torch

# ========================= 
@torch.no_grad()
def _noise(batch_size: int, z_dim: int, device: torch.device):
    return torch.randn(batch_size, z_dim, device=device)


def train_gan(
    model,                  # dict with {"gen", "critic", "z_dim"}
    data_loader,
    device="cpu",
    epochs: int = 5,
    n_critic: int = 5,      # critic steps per generator step
    clip_value: float = 0.01,
    lr: float = 2e-4,
    betas: tuple = (0.5, 0.999),
    use_rmsprop: bool = True,  # classic WGAN (RMSprop) vs Adam
):
    """
    WGAN with weight clipping, mirroring the Module 6 practical:
    - noise: torch.randn(B, z_dim, 1, 1)
    - critic trained n_critic times per gen step
    - loss_D = -(E[D(real)] - E[D(fake)])
    - loss_G = -E[D(fake)]
    - weight clipping after each critic update
    """
    device = torch.device(device)
    gen = model["gen"].to(device).train()
    critic = model["critic"].to(device).train()
    z_dim = int(model.get("z_dim", 100))

    if use_rmsprop:
        opt_gen = torch.optim.RMSprop(gen.parameters(), lr=lr)
        opt_critic = torch.optim.RMSprop(critic.parameters(), lr=lr)
    else:
        opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
        opt_critic = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

    for epoch in range(1, epochs + 1):
        for batch_idx, (real, *_) in enumerate(data_loader):
            real = real.to(device)
            batch_size = real.size(0)

            # === Train Critic ===
            for _ in range(n_critic):
                noise = _noise(batch_size, z_dim, device)
                fake = gen(noise).detach()  # stop grad into G
                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                loss_critic = -(critic_real - critic_fake)

                critic.zero_grad(set_to_none=True)
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping
                # AFTER (only if a value is provided)
                if clip_value is not None:
                    for p in critic.parameters():
                        p.data.clamp_(-clip_value, clip_value)


            # === Train Generator ===
            noise = _noise(batch_size, z_dim, device)
            fake = gen(noise)
            loss_gen = -critic(fake).mean()

            gen.zero_grad(set_to_none=True)
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}/{len(data_loader)}] "
                    f"D: {loss_critic.item():.4f} G: {loss_gen.item():.4f}"
                )

    return {"gen": gen, "critic": critic, "z_dim": z_dim}









def train_diffusion(model, data_loader, device="cpu", epochs=1, T=1000, lr=1e-3):
    """Expose diffusion training via helper_lib.trainer to match Module-4 pattern."""
    return _train_diffusion(model, data_loader, device=device, epochs=epochs, T=T, lr=lr)