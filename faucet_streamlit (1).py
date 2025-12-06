import os
import json
import torch
import clip
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load metadata
with open("faucet_metadata.json", "r") as f:
    metadata = json.load(f)

# Prepare dataset embeddings
features = {}
image_folder = "images"
for fname in os.listdir(image_folder):
    if not fname.endswith(".png"):
        continue
    path = os.path.join(image_folder, fname)
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)[0].detach().cpu().numpy()
    emb /= np.linalg.norm(emb)
    features[path] = emb

# Streamlit UI
st.title("Faucet Image Identifier")
uploaded_file = st.file_uploader("Upload a faucet image (.png)", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model.encode_image(img_tensor)[0].detach().cpu().numpy()
    q_emb /= np.linalg.norm(q_emb)

    # Compute similarities
    sims = {path: float(cosine_similarity(q_emb.reshape(1, -1), emb.reshape(1, -1))[0][0])
            for path, emb in features.items()}
    best_match = max(sims, key=sims.get)
    score = sims[best_match]
    best_meta = metadata.get(best_match.split("/")[-1], {})

    st.markdown("### Best Match:")
    st.image(best_match, caption=f"Score: {score:.3f}", use_column_width=True)

    st.markdown("### Metadata:")
    st.json(best_meta)