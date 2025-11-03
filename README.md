# Assignment-3-Image-Generation
Assignment-3: Image Generation — GAN + FastAPI + Docker

This project implements a Generative Adversarial Network (GAN) trained on MNIST to generate realistic handwritten digit images, and exposes the model via a FastAPI inference service.
A Docker container is provided for fully reproducible deployment.

✅ Repository Structure
HW3/
 ├─ app/
 │   ├─ main.py               # FastAPI application (image generation)
 │   └─ ...
 ├─ helper_lib/               # GAN model, trainer, preprocessing
 ├─ outputs/
 │   ├─ gan_mnist.pt          # ✅ Trained generator weights (required)
 │   ├─ gan_samples.png       # ✅ Demo grid of generated digits
 ├─ train_gan.py              # Training script
 ├─ test_diffusion.py         # (Optional experiment)
 ├─ Dockerfile
 ├─ requirements-docker.txt
 ├─ pyproject.toml / uv.lock  # Reproducible environment
 └─ README.md


✅ No datasets or notebooks included
✅ Only required outputs tracked

🔥 Run the API Locally (no Docker)

Requires Python 3.11 and uv installed

cd HW3
uv sync
uv run uvicorn app.main:app --reload --port 8000


Open the Swagger UI:
👉 http://localhost:8000/docs

Example endpoint:
GET /generate?digit=9 → returns a base64-encoded generated image of digit 9

🐳 Run with Docker (Recommended)

Build image:

cd HW3
docker build -t sps-hw3-api .


Run container:

docker run --rm -p 8000:8000 \
  -v "$(pwd)/outputs:/app/outputs" \
  --name sps-hw3-api \
  sps-hw3-api


Then visit:
👉 http://localhost:8000/docs

✅ No GPU required
✅ Fully containerized inference

🧠 Model Training (if needed)

The pre-trained GAN weights are included, but to regenerate them:

uv run python train_gan.py


Outputs are saved to:

outputs/gan_mnist.pt
outputs/gan_samples.png

✅ Submission Requirements Checklist
Requirement	Status
GAN trained on MNIST	✅
gan_mnist.pt & sample image committed	✅
FastAPI serving image generation	✅
Docker container provided	✅
README instructions accurate	✅
✨ Author

William