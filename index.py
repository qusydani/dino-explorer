import os
import glob
import numpy as np
import faiss
import torch
from PIL import Image
from tqdm import tqdm

from model import load_model, preprocess, extract_features, DEVICE, EMBED_DIM


def build_index(normal_dir: str, model) -> faiss.IndexFlatL2:
    """
    Run all images in normal_dir through DINOv2, collect every patch
    token, and store them in a Faiss flat L2 index.

    Args:
        normal_dir : path to MVTec 'train/good' folder
        model      : loaded DINOv2 model

    Returns:
        index      : faiss.IndexFlatL2 containing all normal patch vectors
    """
    index = faiss.IndexFlatL2(EMBED_DIM)  # exact L2 search, no approximation

    image_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No .png images found in {normal_dir}")

    print(f"Building index from {len(image_paths)} normal images...")

    for path in tqdm(image_paths, desc="Indexing"):
        tensor = preprocess(path)
        _, patches = extract_features(model, tensor)
        # patches: (1, 256, 384) → (256, 384) → numpy float32
        vecs = patches.squeeze(0).cpu().numpy().astype("float32")
        index.add(vecs)

    print(f"Index built — {index.ntotal} patch vectors stored.")
    return index


def save_index(index: faiss.IndexFlatL2, path: str) -> None:
    """Persist index to disk so we don't rebuild every run."""
    faiss.write_index(index, path)
    print(f"Index saved → {path}")


def load_index(path: str) -> faiss.IndexFlatL2:
    """Load a previously saved index from disk."""
    index = faiss.read_index(path)
    print(f"Index loaded — {index.ntotal} patch vectors.")
    return index


def query_index(
    index: faiss.IndexFlatL2,
    patch_tokens: torch.Tensor,
    k: int = 9,
) -> np.ndarray:
    """
    For each of the 256 query patch tokens, find its k nearest
    normal neighbours and return the mean distance as the anomaly score.

    Args:
        index        : built Faiss index of normal patches
        patch_tokens : (1, 256, 384) tensor from extract_features
        k            : number of nearest neighbours

    Returns:
        scores : (256,) float32 array — one anomaly score per patch
                 higher = more anomalous
    """
    vecs = patch_tokens.squeeze(0).cpu().numpy().astype("float32")
    distances, _ = index.search(vecs, k)   # distances: (256, k)
    scores = distances.mean(axis=1)        # (256,) — mean distance to k neighbours
    return scores


if __name__ == "__main__":
    NORMAL_DIR  = os.path.expanduser("~/Downloads/metal_nut/train/good")
    INDEX_PATH  = "metal_nut.index"
    TEST_IMAGE  = os.path.expanduser("~/Downloads/metal_nut/test/bent/000.png")

    model = load_model()

    # Build and save (skip if already exists)
    if not os.path.exists(INDEX_PATH):
        index = build_index(NORMAL_DIR, model)
        save_index(index, INDEX_PATH)
    else:
        print("Index file found — loading from disk.")
        index = load_index(INDEX_PATH)

    # Query with a test image
    tensor = preprocess(TEST_IMAGE)
    _, patches = extract_features(model, tensor)
    scores = query_index(index, patches)

    print(f"\nAnomaly scores — shape : {scores.shape}")
    print(f"Min  : {scores.min():.4f}")
    print(f"Max  : {scores.max():.4f}")
    print(f"Mean : {scores.mean():.4f}")

    # Reshape to spatial grid to sanity-check
    grid = scores.reshape(16, 16)
    print(f"\nPatch score grid (16×16) — top-left corner:")
    print(grid[:4, :4].round(3))