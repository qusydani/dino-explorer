import os
import glob
import streamlit as st
import numpy as np
from PIL import Image

from model import load_model, preprocess, extract_features, extract_attention
from index import build_index, save_index, load_index, query_index
from visualize import (
    load_original,
    anomaly_heatmap,
    pca_patch_map,
    attention_map,
    patch_score_grid,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DINOv2 Patch Anomaly Explorer",
    layout="wide",
)

MVTEC_ROOT = os.path.expanduser("~/Downloads/metal_nut")
INDEX_PATH  = "metal_nut.index"


# ── Cached resources — only load once per session ──────────────────────────────
@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_index(_model):
    if os.path.exists(INDEX_PATH):
        return load_index(INDEX_PATH)
    normal_dir = os.path.join(MVTEC_ROOT, "train", "good")
    index = build_index(normal_dir, _model)
    save_index(index, INDEX_PATH)
    return index


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")

    # Discover defect categories from test folder
    test_root = os.path.join(MVTEC_ROOT, "test")
    categories = sorted([
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    ])
    category = st.selectbox("Defect category", categories)

    # Discover images within chosen category
    cat_dir    = os.path.join(test_root, category)
    image_paths = sorted(glob.glob(os.path.join(cat_dir, "*.png")))
    image_names = [os.path.basename(p) for p in image_paths]
    chosen_name = st.selectbox("Test image", image_names)
    chosen_path = os.path.join(cat_dir, chosen_name)

    st.markdown("---")

    k         = st.slider("k neighbours", min_value=1, max_value=20, value=9, step=1)
    threshold = st.slider("Anomaly threshold", min_value=0.0, max_value=1.0,
                          value=0.5, step=0.01)
    head      = st.selectbox("Attention head", ["mean"] + list(range(6)))

    st.markdown("---")
    rebuild = st.button("Rebuild index")


# ── Load model + index ─────────────────────────────────────────────────────────
model = get_model()

if rebuild:
    # Clear cache so get_index rebuilds from scratch
    get_index.clear()
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    st.toast("Index cleared — rebuilding on next run.")

index = get_index(model)


# ── Run inference ──────────────────────────────────────────────────────────────
tensor          = preprocess(chosen_path)
_, patches      = extract_features(model, tensor)
scores          = query_index(index, patches, k=k)
attn            = extract_attention(model, tensor)
patch_np        = patches.squeeze(0).cpu().numpy()
image_rgb       = load_original(chosen_path)

# Normalise scores to [0, 1] for threshold comparison
scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
n_anomalous = int((scores_norm > threshold).sum())


# ── Metric row ─────────────────────────────────────────────────────────────────
st.markdown("## DINOv2 patch anomaly explorer")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Category",         category)
m2.metric("Anomalous patches", f"{n_anomalous} / 256")
m3.metric("Max patch score",   f"{scores.max():.1f}")
m4.metric("Mean patch score",  f"{scores.mean():.1f}")


# ── Ground truth mask (if available) ──────────────────────────────────────────
mask_path = chosen_path.replace(
    os.path.join("test", category),
    os.path.join("ground_truth", category),
).replace(".png", "_mask.png")

has_mask = os.path.exists(mask_path)


# ── Visualisation grid ─────────────────────────────────────────────────────────
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Original**")
    st.image(image_rgb, use_container_width=True)

    if has_mask:
        st.markdown("**Ground truth mask**")
        mask_img = np.array(Image.open(mask_path).convert("L").resize((224, 224)))
        st.image(mask_img, use_container_width=True, clamp=True)

with col2:
    st.markdown("**Anomaly heatmap**")
    st.image(anomaly_heatmap(image_rgb, scores), use_container_width=True)

    st.markdown("**Patch score grid**")
    st.image(patch_score_grid(scores_norm), use_container_width=True)

with col3:
    st.markdown("**PCA patch map**")
    st.image(pca_patch_map(patch_np), use_container_width=True)

    st.markdown(f"**Attention ({'mean' if head == 'mean' else f'head {head}'})**")
    h = -1 if head == "mean" else int(head)
    st.image(attention_map(attn, image_rgb, head=h), use_container_width=True)


# ── Predicted mask ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Predicted anomaly mask**  "
            f"<span style='color:grey;font-size:12px'>threshold = {threshold:.2f}</span>",
            unsafe_allow_html=True)

pred_mask = (scores_norm > threshold).reshape(16, 16).astype(np.uint8) * 255
pred_mask_up = np.array(
    Image.fromarray(pred_mask).resize((224, 224), Image.NEAREST)
)

mc1, mc2, mc3 = st.columns(3)
with mc1:
    if has_mask:
        st.markdown("Ground truth")
        st.image(mask_img, use_container_width=True, clamp=True)
with mc2:
    st.markdown("Predicted")
    st.image(pred_mask_up, use_container_width=True, clamp=True)
with mc3:
    if has_mask:
        # Simple pixel-level overlap metric
        gt_bin   = (mask_img > 127).astype(np.uint8)
        pred_bin = (pred_mask_up > 127).astype(np.uint8)
        # Upsample gt to match pred resolution
        intersection = (gt_bin & pred_bin).sum()
        union        = (gt_bin | pred_bin).sum()
        iou = intersection / (union + 1e-8)
        st.markdown("Overlap metric")
        st.metric("IoU (patch-level)", f"{iou:.3f}")
        st.caption(
            "Note: IoU is approximate — GT mask is pixel-level, "
            "prediction is 16×16 patch-level."
        )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "DINOv2 ViT-S/14 · frozen · no training · "
    "patch anomaly detection via Faiss k-NN · MVTec AD dataset"
)