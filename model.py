import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE = 14
IMG_SIZE   = 224
GRID_SIZE  = IMG_SIZE // PATCH_SIZE   # 16
N_PATCHES  = GRID_SIZE ** 2           # 256
EMBED_DIM  = 384                      # dinov2_vits14

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

def load_model() -> nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model.to(DEVICE)
    return model

def preprocess(image_path: str) -> torch.Tensor:
    """Image path → normalised tensor (1, 3, 224, 224)."""
    img = Image.open(os.path.expanduser(image_path)).convert("RGB")
    return TRANSFORM(img).unsqueeze(0).to(DEVICE)

def extract_features(
    model: nn.Module,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        cls_token    (1, 384)      — global image summary
        patch_tokens (1, 256, 384) — one vector per 14×14 patch
    """
    with torch.no_grad():
        out = model.forward_features(tensor)
    cls    = out["x_norm_clstoken"]     # (1, 384)
    patches = out["x_norm_patchtokens"] # (1, 256, 384)
    return cls, patches

def extract_attention(
    model: nn.Module,
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Extracts CLS-to-patch attention from the last transformer block
    by temporarily patching it to compute standard (non-fused) attention
    and capturing the weights via a hook on the QKV projection.

    Returns:
        attn (n_heads, n_patches) — normalised to [0, 1] per head
    """
    _store = {}
    last_block = model.blocks[-1]
    n_heads = last_block.attn.num_heads

    def _qkv_hook(module, input, output):
        # output: (B, N, 3 * head_dim * n_heads)
        B, N, _ = output.shape
        head_dim = output.shape[-1] // (3 * n_heads)
        qkv = output.reshape(B, N, 3, n_heads, head_dim)
        q, k = qkv[:, :, 0], qkv[:, :, 1]  # (B, N, n_heads, head_dim)
        # Transpose to (B, n_heads, N, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        scale = head_dim ** -0.5
        # (B, n_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        _store["attn"] = attn.detach()

    hook = last_block.attn.qkv.register_forward_hook(_qkv_hook)

    with torch.no_grad():
        model(tensor)

    hook.remove()

    attn = _store["attn"]       # (1, n_heads, N, N)  N = n_patches + 1
    attn = attn[0, :, 0, 1:]   # (n_heads, n_patches) — CLS → patches
    attn = attn.cpu().numpy()

    # Normalise each head to [0, 1]
    mn = attn.min(axis=1, keepdims=True)
    mx = attn.max(axis=1, keepdims=True)
    attn = (attn - mn) / (mx - mn + 1e-8)

    return attn                 # (n_heads, 256)


if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print("Loading DINOv2 vits14...")
    model = load_model()
    print("Model loaded.\n")

    TEST = os.path.expanduser("~/Downloads/metal_nut/test/bent/000.png")

    tensor = preprocess(TEST)
    print(f"Input tensor    : {tensor.shape}")

    cls, patches = extract_features(model, tensor)
    print(f"CLS token       : {cls.shape}")
    print(f"Patch tokens    : {patches.shape}")

    attn = extract_attention(model, tensor)
    print(f"Attention maps  : {attn.shape}")
    print(f"Grid            : {GRID_SIZE} × {GRID_SIZE} = {N_PATCHES} patches")
    print("\nAll checks passed.")