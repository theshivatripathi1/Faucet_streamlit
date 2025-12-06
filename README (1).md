# Faucet Matching with CLIP + Streamlit

This web app uses OpenAI’s CLIP model to identify the most visually similar faucet image from a dataset, and displays its metadata including brand and article number.

## 🔍 Features

- Upload a faucet image and find the closest match
- Uses CLIP (ViT-B/32) for image embeddings
- Displays metadata from `faucet_metadata.json`
- Simple UI via Streamlit
- Ideal for product lookup or catalog integration

## 📁 Folder Structure

```
.
├── faucet_streamlit.py        # Main Streamlit app
├── faucet_metadata.json       # Metadata for reference images
├── requirements.txt           # Python dependencies
├── images/                    # Reference faucet images (.png)
├── testing_images/            # Query images (optional)
```

## 🚀 Running Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/faucet-matcher.git
   cd faucet-matcher
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run faucet_streamlit.py
   ```

## ☁️ Deploy on Streamlit Cloud

1. Push your repo to GitHub.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Click "New App" and select this repo.
4. Set the main file as `faucet_streamlit.py`.
5. Deploy and upload a query image to test.

## ⚠️ Notes

- GitHub has a 100MB file limit. If your `images/` dataset is large, only include a subset for deployment.
- Ensure all images are in `.png` format.

---

Built with ❤️ using Streamlit and OpenAI's CLIP.