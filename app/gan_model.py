import io
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from helper_lib.model import MNISTGenerator  # you already defined this

_DEVICE = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")

def load_gan(weights_path: str, z_dim: int = 100):
    gen = MNISTGenerator(z_dim=z_dim).to(_DEVICE)
    sd = torch.load(weights_path, map_location=_DEVICE)
    gen.load_state_dict(sd)
    gen.eval()
    return gen

@torch.no_grad()
def sample_png(gen, count: int = 16, z_dim: int = 100) -> bytes:
    z = torch.randn(count, z_dim, device=_DEVICE)
    imgs = gen(z).cpu()                    # (B,1,28,28), in [-1,1]
    imgs = (imgs + 1) / 2                  # to [0,1]
    grid = make_grid(imgs, nrow=int(count**0.5), padding=2)

    fig = plt.figure(figsize=(4,4))
    plt.axis("off"); plt.imshow(grid[0], cmap="gray")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
