import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from sklearn.decomposition import PCA

from model import GRID_SIZE, IMG_SIZE


def load_original(image_path: str) -> np.ndarray:
    """Load image as RGB numpy array at model input resolution."""
    img = Image.open(os.path.expanduser(image_path)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img)


def anomaly_heatmap(
    image_rgb: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay per-patch anomaly scores as a heatmap on the original image.

    Args:
        image_rgb : (224, 224, 3) uint8
        scores    : (256,) float32 — one score per patch
        alpha     : heatmap opacity

    Returns:
        overlay   : (224, 224, 3) uint8
    """
    # Reshape scores to 16×16 grid, upsample to 224×224
    grid = scores.reshape(GRID_SIZE, GRID_SIZE).astype("float32")

    # Normalise to [0, 1] for colormap
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    # Upsample via PIL for clean bilinear interpolation
    heat_img = Image.fromarray((grid * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    heat_arr = np.array(heat_img) / 255.0

    # Apply jet colormap → (224, 224, 3)
    heatmap_rgb = (cm.jet(heat_arr)[:, :, :3] * 255).astype(np.uint8)

    # Blend with original
    overlay = (
        (1 - alpha) * image_rgb.astype(np.float32)
        + alpha * heatmap_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return overlay


def pca_patch_map(patch_tokens: np.ndarray) -> np.ndarray:
    """
    Reduce 256 patch tokens to 3 PCA dimensions, map to RGB.
    Same semantic regions → same colour, zero labels needed.

    Args:
        patch_tokens : (256, 384) float32

    Returns:
        rgb_grid     : (224, 224, 3) uint8
    """
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(patch_tokens)   # (256, 3)

    # Normalise each channel to [0, 1]
    for i in range(3):
        ch = reduced[:, i]
        reduced[:, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

    # Reshape to 16×16×3 then upsample to 224×224
    rgb_small = (reduced.reshape(GRID_SIZE, GRID_SIZE, 3) * 255).astype(np.uint8)
    rgb_grid = np.array(
        Image.fromarray(rgb_small).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    )
    return rgb_grid


def attention_map(
    attn: np.ndarray,
    image_rgb: np.ndarray,
    head: int = -1,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Overlay attention weights on the original image.

    Args:
        attn      : (n_heads, 256) — from extract_attention
        image_rgb : (224, 224, 3) uint8
        head      : which head to show (-1 = mean across all heads)
        alpha     : attention overlay opacity

    Returns:
        overlay   : (224, 224, 3) uint8
    """
    if head == -1:
        weights = attn.mean(axis=0)   # (256,) — mean across heads
    else:
        weights = attn[head]

    # Normalise, reshape, upsample
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    grid = weights.reshape(GRID_SIZE, GRID_SIZE).astype("float32")

    attn_img = Image.fromarray((grid * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    attn_arr = np.array(attn_img) / 255.0
    attn_rgb = (cm.inferno(attn_arr)[:, :, :3] * 255).astype(np.uint8)

    overlay = (
        (1 - alpha) * image_rgb.astype(np.float32)
        + alpha * attn_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return overlay


def patch_score_grid(scores: np.ndarray) -> np.ndarray:
    """
    Render the raw 16×16 anomaly score grid as a colour image.
    No blending — pure score visualisation.

    Returns:
        rgb : (224, 224, 3) uint8
    """
    grid = scores.reshape(GRID_SIZE, GRID_SIZE).astype("float32")
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    rgb_small = (cm.RdYlGn_r(grid)[:, :, :3] * 255).astype(np.uint8)
    rgb = np.array(
        Image.fromarray(rgb_small).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    )
    return rgb


def plot_all(
    image_path: str,
    scores: np.ndarray,
    patch_tokens: np.ndarray,
    attn: np.ndarray,
    save_path: str = None,
) -> None:
    """
    Render all four visualisations side by side and display/save.
    """
    image_rgb = load_original(image_path)

    panels = {
        "Original":        image_rgb,
        "Anomaly heatmap": anomaly_heatmap(image_rgb, scores),
        "PCA patch map":   pca_patch_map(patch_tokens),
        "Attention (mean)":attention_map(attn, image_rgb),
        "Patch score grid":patch_score_grid(scores),
    }

    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4))
    fig.suptitle(os.path.basename(image_path), fontsize=11)

    for ax, (title, img) in zip(axes, panels.items()):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()


if __name__ == "__main__":
    import torch
    from model import load_model, preprocess, extract_features, extract_attention
    from index import load_index, query_index

    TEST_IMAGE = os.path.expanduser("~/Downloads/metal_nut/test/bent/000.png")
    INDEX_PATH = "metal_nut.index"

    model = load_model()
    index = load_index(INDEX_PATH)

    tensor  = preprocess(TEST_IMAGE)
    _, patches = extract_features(model, tensor)
    scores  = query_index(index, patches)
    attn    = extract_attention(model, tensor)

    # patch_tokens as numpy for PCA
    patch_np = patches.squeeze(0).cpu().numpy()

    plot_all(TEST_IMAGE, scores, patch_np, attn, save_path="output.png")