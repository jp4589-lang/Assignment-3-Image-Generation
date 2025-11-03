from typing import Union
from fastapi import FastAPI, Response
from pydantic import BaseModel
from .bigram_model import BigramModel
# from app.embeddings import word_vector, sentence_vector, cosine_similarity
from .lstm_model import LSTMModel
import torch
import base64
from io import BytesIO
from torchvision.utils import make_grid, save_image
from helper_lib.model import get_model
from pathlib import Path
import math
app = FastAPI()
# Sample corpus for the bigram model
corpus = [
"The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
"this is another example sentence",
"we are generating text based on bigram probabilities",
"bigram models are simple but effective"
]

# Select GPU if available
def _device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = _device()

# ---------- GAN (MNIST) lazy loader ----------
_gan_gen = None  # cached generator

def _weights_path() -> Path:
    """
    Try both repo-root/outputs and CWD/outputs for gan_mnist.pt.
    Works in dev and inside Docker if you COPY or mount outputs/.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../sps_genai
    candidates = [
        repo_root / "outputs/gan_mnist.pt",
        Path("outputs/gan_mnist.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "GAN weights not found. Expected outputs/gan_mnist.pt. "
        "Train with `python train_gan.py` first."
    )

def _load_gan():
    global _gan_gen
    if _gan_gen is not None:
        return _gan_gen

    dev = _device()
    bundle = get_model("mnist_gan", z_dim=100)
    gen = bundle["gen"].to(dev).eval()

    state = torch.load(_weights_path(), map_location=dev)
    gen.load_state_dict(state)
    _gan_gen = gen
    return _gan_gen




# print("Using device:", device)
# --- Build vocab from the same corpus ---
text = " ".join(corpus).lower().split()
vocab = sorted(set(text))

stoi = {w:i for i,w in enumerate(vocab)}
itos = {i:w for w,i in stoi.items()}
vocab_size = len(vocab)

lstm_model = LSTMModel(vocab_size=vocab_size).to(device)
lstm_model.eval()  # no training right now



bigram_model = BigramModel(corpus)
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.get("/generate")
def generate_text_get(start_word: str, length: int = 20):
    generated_text = bigram_model.generate_text(start_word, length)
    return {"generated_text": generated_text}

class RNNRequest(BaseModel):
    start_word: str
    length: int = 20

@app.post("/generate_with_rnn")
def generate_with_rnn(req: RNNRequest):
    if req.start_word not in stoi:
        return {"error": f"Start word '{req.start_word}' not in vocabulary."}

    ids = [stoi[req.start_word]]
    hidden = None

    for _ in range(max(1, req.length)):
        x = torch.tensor([ids[-1:]], dtype=torch.long).to(device)  # shape (1, 1)
        with torch.no_grad():
            logits, hidden = lstm_model(x, hidden)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs[0], num_samples=1).item()
        ids.append(next_id)

    words = [itos[i] for i in ids]
    return {"generated_text": " ".join(words)}


class GanSampleReq(BaseModel):
    n: int = 16        # number of images to sample (make it a square like 4, 9, 16)
    z_dim: int = 100   # latent size used during training

@app.post("/gan/sample", response_class=Response)
def gan_sample(req: GanSampleReq):
    """
    Returns a PNG grid of sampled MNIST digits from the trained generator.
    """
    gen = _load_gan()
    dev = _device()

    with torch.no_grad():
        z = torch.randn(req.n, req.z_dim, device=dev)
        imgs = gen(z).cpu()              # (N, 1, 28, 28) in [-1, 1]
        imgs = (imgs + 1) / 2.0          # -> [0,1] for display

        grid = make_grid(
            imgs,
            nrow=int(math.ceil(req.n ** 0.5)),   # <-- handle non-square n
            padding=2
        )

        buf = BytesIO()
        save_image(grid, buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")







# ---------------------- NEW: embeddings API ----------------------
class WordReq(BaseModel):
    word: str

class SentenceReq(BaseModel):
    text: str

class SimilarityReq(BaseModel):
    a: str
    b: str

# @app.post("/embed/word")
# def embed_word(req: WordReq):
#     vec = word_vector(req.word)
#     return {"word": req.word, "dim": len(vec), "vector": vec.tolist()}

# @app.post("/embed/sentence")
# def embed_sentence(req: SentenceReq):
#     vec = sentence_vector(req.text)
#     return {"text": req.text, "dim": len(vec), "vector": vec.tolist()}

# @app.post("/similarity")
# def similarity(req: SimilarityReq):
#     av = word_vector(req.a)
#     bv = word_vector(req.b)
#     sim = cosine_similarity(av, bv)
#     return {"a": req.a, "b": req.b, "cosine_similarity": sim}