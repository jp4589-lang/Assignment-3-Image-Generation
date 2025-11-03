import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_model import build_diffusion_model


# ----------------------
# 1) Fully-connected MLP
# ----------------------
class FCNN(nn.Module):
    """
    Simple MLP for CIFAR-10: Flatten -> [Linear-ReLU]*2 -> Linear(10)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

# ------------
# 2) Baseline CNN
# ------------
class CNN(nn.Module):
    """
    Two conv blocks with BN -> ReLU -> MaxPool, then FC.
    Output: 10 classes for CIFAR-10.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------
# 3) Enhanced CNN
# ----------------
class EnhancedCNN(nn.Module):
    """
    Deeper CNN with three blocks + dropout; slightly stronger than CNN.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 128 * 4 * 4 = 2048
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))



# -----------------
# 4) Variational Autoencoder (VAE)
# -----------------
class VAE(nn.Module):
    """
    Variational Autoencoder for CIFAR-10.
    Encoder -> latent mean/logvar -> reparam -> Decoder reconstructs image.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: input 3x32x32 -> compressed latent space
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 32x16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 64x8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 128x4x4
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder: latent vector -> reconstruct image
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 3x32x32
            nn.Sigmoid(),  # normalize output to [0,1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        h_decoded = self.decoder_input(z)
        h_decoded = h_decoded.view(-1, 128, 4, 4)
        recon_x = self.decoder(h_decoded)

        return recon_x, mu, logvar







# -----------------
# 5) GAN: Generator + Critic
# -----------------
class Generator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),  # images in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- MNIST 28x28 DCGAN ---
class MNISTGenerator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),

            nn.Unflatten(1, (128, 7, 7)),          # (B,128,7,7)

            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), -1)
        return self.net(z)


class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),   # 28->14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 14->7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128*7*7, 1)
        )

    def forward(self, x):  # returns logits
        return self.net(x)






# -----------------
# Factory function
# -----------------
# -----------------
# Factory function
# -----------------
def get_model(name: str, **kwargs) -> nn.Module:
    """
    Factory:
      - "fcnn" | "mlp"        -> FCNN()
      - "cnn"                 -> CNN()
      - "enhancedcnn" | "enhanced" -> EnhancedCNN()
      - "vae"                 -> VAE()
      - "gan"                 -> {"gen": Generator, "critic": Critic, "z_dim": int}  (dict)
      - "diffusion" | "ddpm"  -> UNetTiny() from diffusion_model (noise predictor)

    Extra kwargs:
      - img_channels (int) for diffusion model, default=1 (e.g., MNIST)
      - z_dim (int) for GAN, default=100
    """
    name = name.lower()

    if name in ("fcnn", "mlp"):
        return FCNN()
    if name == "cnn":
        return CNN()
    if name in ("enhancedcnn", "enhanced"):
        return EnhancedCNN()
    if name == "vae":
        return VAE()
    if name == "gan":
        z_dim = kwargs.get("z_dim", 100)
        return {"gen": Generator(z_dim), "critic": Critic(), "z_dim": z_dim}
    if name in ("diffusion", "ddpm"):
        img_channels = kwargs.get("img_channels", 1)
        return build_diffusion_model(img_channels=img_channels)
    if name in {"mnist_gan", "gan_mnist"}:
        z_dim = kwargs.get("z_dim", 100)
        return {"gen": MNISTGenerator(z_dim), "critic": MNISTDiscriminator(), "z_dim": z_dim}


    raise ValueError(f"Unknown model name: {name!r}")


