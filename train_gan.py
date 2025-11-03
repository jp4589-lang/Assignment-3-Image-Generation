# train_gan.py  (MNIST version for HW3)
import os, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_gan_samples

def main():
    # Prefer Apple MPS, else CUDA, else CPU
    device = ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available()
              else "cpu")
    print(f"Using device: {device}")

    # MNIST, 28x28, 1 channel in [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # single-channel mean/std
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,                          # safest on macOS
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )

    # Use the MNIST GAN you added in helper_lib.model
    gan = get_model("mnist_gan", z_dim=100)

    # Train (DCGAN defaults; no weight clipping / RMSprop)
    trained = train_gan(
        model=gan,
        data_loader=loader,
        device=device,
        epochs=5,               # bump if you want better samples
        n_critic=1,
        clip_value=None,        # no clipping for DCGAN
        lr=2e-4,
        betas=(0.5, 0.999),
        use_rmsprop=False
    )

    # Save a grid to outputs/
    generate_gan_samples(trained, device=device, num_samples=16,
                         save_path="outputs/gan_samples.png")
    
    os.makedirs("outputs", exist_ok=True)
    torch.save(trained["gen"].state_dict(), "outputs/gan_mnist.pt")
    print("[GAN] saved generator weights to: outputs/gan_mnist.pt")











if __name__ == "__main__":
    main()
