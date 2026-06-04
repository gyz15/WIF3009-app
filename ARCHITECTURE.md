# ARCHITECTURE.md — WIF3009 Backend

## Overview

Stateless ML inference microservice for predicting resale apparel prices. FastAPI + multi-agent pipeline + XGBoost regressor.

```
CLIENT (browser/curl)
    │
    ▼
┌─────────────────────────────────────────────┐
│  FastAPI (app/main.py)                       │
│  Lifespan · CORS · Static /                  │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────▼──────────────┐
    │  Routes (routes/predict.py)│
    │  POST /api/predict         │
    │  GET  /api/brands          │
    │  GET  /api/categories      │
    │  GET  /api/conditions      │
    │  GET  /api/health          │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────┐
    │  Orchestrator              │
    │  run_pipeline()            │
    │  1. Context Agent          │
    │  2. Text Agent             │
    │  3. Image Agent            │
    │  4. Feature Fusion         │
    │  5. XGBoost Predict        │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────┐
    │  Model Loader (singleton)  │
    │  backend/models/*          │
    └────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI >= 0.110.0 |
| Server | Uvicorn (standard extras) |
| ML - Regression | XGBoost >= 2.0.0 |
| ML - Text | scikit-learn (TF-IDF, LabelEncoder) |
| ML - Image | TensorFlow/Keras (MobileNetV2) |
| Serialization | joblib |
| Frontend | Vanilla HTML/JS (static/index.html) |
| Python | 3.10+ |

## Directory Structure

```
backend/
├── run.py                     # Entry point (uvicorn launcher)
├── requirements.txt           # Python dependencies
├── static/index.html          # Browser test UI
├── models/                    # Pretrained model artifacts
│   ├── hype_engine_model.json # XGBoost regressor
│   ├── tfidf_vectorizer.pkl   # TF-IDF (5000 features)
│   ├── brand_encoder.pkl      # LabelEncoder (brands)
│   ├── cat_encoder.pkl        # LabelEncoder (sub-categories)
│   └── image_mean_vector.npy  # Global mean image vector (1280-dim)
└── app/
    ├── main.py                # FastAPI app factory, lifespan, CORS
    ├── config.py              # Paths, constants, env vars
    ├── routes/
    │   └── predict.py         # API route handlers (4 endpoints)
    ├── models/
    │   └── schemas.py         # Pydantic request/response models
    ├── agents/
    │   ├── orchestrator.py    # Pipeline coordinator
    │   ├── context_agent.py   # Brand/cat encoding + condition injection
    │   ├── text_agent.py      # TF-IDF text vectorization
    │   └── image_agent.py     # MobileNetV2 image feature extraction
    └── utils/
        └── model_loader.py    # Thread-safe singleton model loader
```

## Architecture Pattern

**Layered + Agent-based Orchestration**

1. **Transport Layer** — FastAPI routes handle HTTP, multipart parsing, validation.
2. **Orchestration Layer** — `run_pipeline()` coordinates three agents, runs in separate thread via `asyncio.to_thread()` (30s timeout).
3. **Agent Layer** — Three specialized agents (Context, Text, Image) each produce a feature vector.
4. **Infrastructure Layer** — Singleton model loader, centralized config, static file serving.

## Data Flow (Prediction)

```
POST /api/predict (multipart/form-data)
  image (file) + description + product_name + brand + sub_category + condition_id
    │
    ▼
predict_price() — validates models loaded, reads image bytes, builds PredictionInput
    │
    ▼
run_pipeline() [in thread]:
    │
    ├─► Context Agent: inject_condition_text(desc, cond_id)
    │   Appends condition keywords (e.g. "new with tags nwt brand new never used")
    │
    ├─► Text Agent: vectorize_text(enriched_desc, product_name)
    │   TF-IDF → sparse CSR [1 × 5000]
    │
    ├─► Context Agent: encode_brand(brand) → int
    │                  encode_category(sub_cat) → int
    │                  make_meta_features(b, c, 0.0) → dense [1 × 3]
    │
    ├─► Image Agent: extract_features(image_bytes)
    │   MobileNetV2 (no top, GlobalAvgPool) → dense [1 × 1280]
    │   Fallback: global mean vector if image is None or fails
    │
    ├─► Fusion: sp.hstack([text, meta, image]) → sparse [1 × 6283]
    │
    ├─► XGBoost: xgb.DMatrix → xgb_model.predict() → log_price
    │
    ├─► Post-process: np.expm1(log_price) → predicted_price (USD)
    │
    └─► Confidence: predicted ± expm1(RMSE_LOG_SCALE)
    │
    ▼
PredictionResponse (JSON)
    { predicted_price, confidence_low, confidence_high, input_summary, warnings }
```

## API Endpoints

| Method | Path | Input | Output | Purpose |
|--------|------|-------|--------|---------|
| `GET` | `/` | — | HTML | Static test UI |
| `POST` | `/api/predict` | Multipart form | `PredictionResponse` | Price prediction |
| `GET` | `/api/brands?q=` | Query param | `[BrandItem]` | List known brands |
| `GET` | `/api/categories?q=` | Query param | `[CategoryItem]` | List known categories |
| `GET` | `/api/conditions` | — | `[ConditionItem]` | Condition ID→label map |
| `GET` | `/api/health` | — | `HealthResponse` | Health check + model status |

## Key Design Decisions

- **No database.** Application is fully stateless. All knowledge stored in serialized ML artifacts.
- **Thread-safe singleton** for model loading (`ModelLoader.__new__` + lock). Loaded once at startup; image model lazy-loaded on first request.
- **Prediction runs in thread** via `asyncio.to_thread()` to avoid blocking async event loop. 30-second timeout returns 504.
- **Image fallback.** If no image uploaded or MobileNetV2 fails, uses precomputed global mean vector (1280-dim zeros mean).
- **Log-space prediction.** XGBoost regressor trained on `log1p(price)`. Output exponentiated back to USD via `np.expm1()`.
- **RMSE-based confidence.** `confidence = predicted ± expm1(0.5442)` — simple heuristic from training RMSE.
- **CORS** configured for `localhost:5173` (Vite) and `localhost:3000` (CRA) for frontend dev.
- **Configuration** centralized in `app/config.py`. Supports `.env` via `python-dotenv` for `HOST`, `PORT`, `RELOAD`.

## Model Artifact Origins

All artifacts in `backend/models/` are produced by `WIF3009.ipynb` (Google Colab, GPU T4) and downloaded from Google Drive. Training uses:
- Kaggle Fashion Dataset (visual features)
- Mercari Dataset (text descriptions + pricing)
