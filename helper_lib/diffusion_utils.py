# helper_lib/diffusion_utils.py
import torch
import torch.nn.functional as F

# ---------- schedules ----------
def linear_beta_schedule(T=1000, start=1e-4, end=0.02, device="cpu"):
    betas  = torch.linspace(start, end, T, device=device)     # β_t
    alphas = 1.0 - betas                                      # α_t
    abar   = torch.cumprod(alphas, dim=0)                     # ᾱ_t
    return betas, alphas, abar

@torch.no_grad()
def to_minus1_1(x):
    # if [0,1] -> [-1,1]; else assume already in [-1,1]
    return x*2-1 if x.min() >= 0 and x.max() <= 1 else x

# ---------- training ----------
def diffusion_loss(model, x0, t, abar):
    """
    ε ~ N(0, I); x_t = sqrt(ᾱ_t) x0 + sqrt(1-ᾱ_t) ε
    Loss = MSE(ε_pred, ε)
    """
    B = x0.size(0)
    a_bar_t = abar[t].view(B,1,1,1)
    noise = torch.randn_like(x0)
    x_t = a_bar_t.sqrt()*x0 + (1.0 - a_bar_t).sqrt()*noise
    eps_pred = model(x_t, t)
    return F.mse_loss(eps_pred, noise)

def train_diffusion(model, loader, device="cpu", epochs=1, T=1000, lr=1e-3):
    model.to(device).train()
    betas, alphas, abar = linear_beta_schedule(T=T, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        for imgs, _ in loader:
            imgs = to_minus1_1(imgs).to(device)
            t = torch.randint(0, T, (imgs.size(0),), device=device, dtype=torch.long)
            loss = diffusion_loss(model, imgs, t, abar)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model

# ---------- sampling ----------
@torch.no_grad()
def sample_ddpm(model, num_samples=16, img_size=28, img_channels=1, device="cpu", T=1000):
    model.eval().to(device)
    betas, alphas, abar = linear_beta_schedule(T=T, device=device)

    x = torch.randn(num_samples, img_channels, img_size, img_size, device=device)
    for t in reversed(range(T)):
        t_b = torch.full((num_samples,), t, device=device, dtype=torch.long)
        eps = model(x, t_b)
        a_t = alphas[t]
        ab_t = abar[t]
        # mean per DDPM paper (simplified)
        mean = (1.0/torch.sqrt(a_t)) * (x - ((1-a_t)/torch.sqrt(1-ab_t)) * eps)
        if t > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(betas[t]) * z
        else:
            x = mean
    # back to [0,1] for viewing
    x = (x.clamp(-1,1) + 1)/2
    return x
