# WIF3009 — Resale Apparel Price Prediction

Multi-agent AI system that predicts resale apparel prices using text, image, and context features combined with an XGBoost regression model.

## Prerequisites

- **Python 3.10+** (required by TensorFlow 2.15+)
- **pip**

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/shengFung/WIF3009.git
cd WIF3009/backend

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model files from Google Drive
#    See the section below — place all model files into backend/models/

# 5. Start the server
python run.py
```

The API will be available at `http://localhost:8000`.

Open `http://localhost:8000` in your browser for a simple test UI.

## Downloading Model Files

The trained model artifacts are hosted on Google Drive. Download the following files and place them under `backend/models/`:

| File | Purpose |
|------|---------|
| `hype_engine_model.json` | XGBoost price prediction model |
| `tfidf_vectorizer.pkl` | TF-IDF text vectorizer (5000 features) |
| `brand_encoder.pkl` | Label encoder for brand names |
| `cat_encoder.pkl` | Label encoder for sub-categories |
| `image_mean_vector.npy` | Global mean image feature vector (1280-dim) |

**Download link:** `[GOOGLE_DRIVE_LINK]`

> After downloading, the `backend/models/` directory should contain all five files listed above.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Predict resale price (multipart form: image, description, brand, category, condition) |
| `GET` | `/api/brands` | List known brand names |
| `GET` | `/api/categories` | List known sub-categories |
| `GET` | `/api/conditions` | List condition ID → label mapping |
| `GET` | `/api/health` | Health check |

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "description=Vintage denim jacket, excellent condition" \
  -F "brand=Levi's" \
  -F "category=Jackets" \
  -F "condition=2" \
  -F "image=@jacket.jpg"
```

## Project Structure

```
WIF3009/
├── WIF3009.ipynb          # Model training notebook (Google Colab)
├── backend/
│   ├── run.py             # Entry point
│   ├── requirements.txt   # Python dependencies
│   ├── models/            # Trained model artifacts (download from Drive)
│   ├── static/
│   │   └── index.html     # Browser-based test UI
│   └── app/
│       ├── main.py        # FastAPI application
│       ├── config.py      # Paths and constants
│       ├── agents/        # Text, Image, Context agents + orchestrator
│       ├── routes/        # API route handlers
│       ├── models/        # Pydantic request/response schemas
│       └── utils/         # Model loader utilities
```

## How It Works

1. **Context Agent** — encodes brand, category, and injects condition keywords into the description.
2. **Text Agent** — vectorizes the enriched description using TF-IDF.
3. **Image Agent** — extracts visual features from an uploaded image using MobileNetV2, or falls back to a global mean vector when no image is provided.
4. **Feature Fusion** — concatenates text, image, and metadata features.
5. **Hype Engine** — an XGBoost model predicts the log-transformed resale price (USD). The result is exponentiated back and returned with a confidence range.
