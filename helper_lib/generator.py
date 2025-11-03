import os, math
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from .diffusion_utils import sample_ddpm

# --------------------------
# 1) VAE sample generator
# --------------------------
@torch.no_grad()
def generate_vae_samples(
    model,
    device: str = "cpu",
    num_samples: int = 16,
    latent_dim: int | None = None,
    save_path: str = "models/vae_samples.png",
    show: bool = True,
):
    model.eval()
    model.to(device)

    # infer latent size
    if latent_dim is None:
        latent_dim = getattr(model, "latent_dim", 128)

    # sample z ~ N(0, I)
    z = torch.randn(num_samples, latent_dim, device=device)

    # ---- decode correctly for your architecture ----
    if hasattr(model, "decoder_input"):
        h = model.decoder_input(z)                      # (N, 128*4*4)
        first_deconv = model.decoder[0]                 # ConvTranspose2d(128, 64, ...)
        C = getattr(first_deconv, "in_channels", 128)   # 128
        S = int((h.shape[1] // C) ** 0.5)               # 4
        h = h.view(-1, C, S, S)                         # (N, 128, 4, 4)
        out = model.decoder(h)                          # (N, 3, 32, 32)
    else:
        out = model.decoder(z)

    # ensure [0,1]
    imgs = out
    if imgs.min() < 0 or imgs.max() > 1:
        imgs = torch.sigmoid(imgs)
    imgs = imgs.detach().cpu()

    # grid
    side = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(side, side, figsize=(side * 2, side * 2))
    axes = axes.flatten()
    for i in range(side * side):
        axes[i].axis("off")
        if i < num_samples:
            axes[i].imshow(imgs[i].permute(1, 2, 0))  # RGB

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[VAE] saved samples to: {save_path}")
    return save_path


# --------------------------
# 2) GAN sample generator
# --------------------------
@torch.no_grad()
def generate_gan_samples(
    model,
    device="cpu",
    num_samples: int = 16,
    z_dim: int | None = None,
    save_path: str | None = "outputs/gan_samples.png",
    show: bool = True,
):
    """
    Generate images from the trained GAN generator and display a grid.
    Assumes training normalized images to [-1, 1] with Tanh output.
    """
    device = torch.device(device)
    gen = model["gen"].to(device).eval()
    z_dim = int(z_dim or model.get("z_dim", 100))

    
    # ✅ 2-D noise — MNISTGenerator expects this; conv generators also handle it
    noise = torch.randn(num_samples, z_dim, device=device)
    fake = gen(noise).cpu()

    # de-normalize from [-1, 1] to [0, 1]
    imgs = (fake + 1) / 2
    nrow = int(math.sqrt(num_samples)) or 1
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2, normalize=False)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[GAN] saved samples to: {save_path}")

    if show:
        plt.show()
    plt.close()
    return imgs

def generate_samples(model, device="cpu", num_samples=16, img_size=28, img_channels=1, T=1000, show=True):
    imgs = sample_ddpm(model, num_samples=num_samples, img_size=img_size,
                       img_channels=img_channels, device=device, T=T)
    if show:
        cols = int(num_samples**0.5)
        rows = (num_samples + cols - 1)//cols
        plt.figure(figsize=(cols*2, rows*2))
        for i in range(num_samples):
            plt.subplot(rows, cols, i+1)
            plt.imshow(imgs[i, 0].cpu(), cmap="gray")
            plt.axis("off")
        plt.tight_layout(); plt.show()
    return imgs