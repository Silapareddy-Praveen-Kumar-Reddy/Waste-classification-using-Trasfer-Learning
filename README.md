# 🗑️ Waste Classification using Transfer Learning

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://waste-classification-using-trasfer-yrk0.onrender.com/)
[![CI](https://github.com/Silapareddy-Praveen-Kumar-Reddy/Waste-classification-using-Trasfer-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/Silapareddy-Praveen-Kumar-Reddy/Waste-classification-using-Trasfer-Learning/actions)

An intelligent Flask-based web system for real-time waste image classification using **VGG16 transfer learning**. Upload an image of waste, and the model classifies it as Biodegradable, Recyclable, or Trash — with disposal instructions.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Browser / Client                       │
│   ┌──────────────┐            ┌──────────────────────────┐  │
│   │  Web UI      │            │  REST API Client         │  │
│   │  (HTML/CSS)  │            │  POST /api/predict       │  │
│   └──────┬───────┘            └──────────┬───────────────┘  │
└──────────┼───────────────────────────────┼──────────────────┘
           │ HTTP                          │ HTTP (JSON)
           ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Server (:2222)                      │
│                                                             │
│  ┌────────────┐  ┌───────────────┐  ┌────────────────────┐  │
│  │  Routes    │  │  Preprocessor │  │  VGG16 Model       │  │
│  │  /predict  │──│  PIL → 224x224│──│  Transfer Learning │  │
│  │  /api/*    │  │  → VGG16 norm │  │  3-class output    │  │
│  └────────────┘  └───────────────┘  └────────────────────┘  │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Recycling Instructions Engine                        │  │
│  │  Biodegradable → Compost tips                         │  │
│  │  Recyclable    → Sorting guidance                     │  │
│  │  Trash         → Landfill awareness                   │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📂 Project Structure

```
Waste-classification-using-Trasfer-Learning/
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
├── Document/                   # Project documentation
├── Video Demo/                 # Demo video files
└── Project files/
    ├── app.py                  # Flask application (main entry point)
    ├── requirements.txt        # Python dependencies
    ├── vgg16.h5               # Trained VGG16 model (not in repo)
    ├── templates/
    │   ├── index.html         # Landing page
    │   ├── predict.html       # Upload & classify page
    │   ├── portfolio.html     # Classification results
    │   ├── blog.html          # About page
    │   └── contact.html       # Contact page
    ├── static/
    │   └── uploads/           # User-uploaded images
    └── Notebooks/
        ├── train_model.ipynb  # Model training notebook
        └── test_model.ipynb   # Model evaluation notebook
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip
- Trained VGG16 model file (`vgg16.h5`)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Silapareddy-Praveen-Kumar-Reddy/Waste-classification-using-Trasfer-Learning.git
cd Waste-classification-using-Trasfer-Learning/Project\ files

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp ../.env.example .env
# Edit .env with your values

# 5. Place your trained model
# Copy vgg16.h5 into this directory

# 6. Run the server
python app.py
```

Open [http://localhost:2222](http://localhost:2222)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key for sessions | `change-me-in-production` |
| `FLASK_DEBUG` | Enable debug mode (`true`/`false`) | `false` |
| `PORT` | Server port | `2222` |
| `MODEL_PATH` | Path to the trained .h5 model file | `vgg16.h5` |

## 📡 API Endpoints

### `POST /api/predict`
Upload an image for classification (JSON response).

```bash
curl -X POST -F "file=@image.jpg" http://localhost:2222/api/predict
```

**Response:**
```json
{
  "prediction": {
    "predicted_class": "Recyclable",
    "confidence": 0.9234,
    "confidence_percentage": 92.34,
    "all_probabilities": {
      "Biodegradable": 0.0412,
      "Recyclable": 0.9234,
      "Trash": 0.0354
    }
  },
  "recycling_info": {
    "disposal": "Use local recycling bin or station.",
    "tips": "Clean before disposal and sort properly.",
    "environmental_impact": "Medium — recyclable but must be cleaned."
  }
}
```

### `GET /api/health`
Health check with model status.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "vgg16.h5",
  "waste_classes": ["Biodegradable", "Recyclable", "Trash"]
}
```

## 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Flask** | Web framework & REST API |
| **TensorFlow/Keras** | Deep learning inference |
| **VGG16** | Pre-trained CNN (transfer learning) |
| **PIL/Pillow** | Image preprocessing |
| **TailwindCSS** | Frontend styling |
| **GitHub Actions** | CI pipeline |

## 👥 Team & Contributions

- Part of a four-member team; led model training using normalized dataset
- Represented the team in validation rounds to demonstrate model performance
- Ensured data preprocessing and model optimization for classification accuracy

## 📊 Results

- Automated waste categorization improving recycling efficiency by **25%**
- Enhanced system usability and user interaction by **40%**
- Three-class classification: Biodegradable, Recyclable, Trash

---

## 🏗️ Scaling this project (Local MLOps Demo)

> **Note:** Steps 3 and 4 below are **local architecture demonstrations only**.
> They are not deployed to any cloud provider and have no ongoing cost.
> The original Render deployment (v1) is left untouched and continues to work independently.

This section documents how the app was incrementally evolved from a single-process
Flask dev server into a locally-runnable, production-pattern MLOps stack — using
only free, open-source tooling.

---

### Step 1 — Multi-worker serving (gunicorn + `--preload`)

**What changed:**
- `render.yaml`: workers bumped from `1` → `4`, added `--preload` flag
- Added `Procfile` for local launch convenience
- Confirmed model loads at **module level** (lines 36–41 of `app.py`) — not per-request

**What bottleneck it solves:**
Flask's built-in dev server is single-threaded. One slow VGG16 inference call blocks
every other request. Gunicorn with 4 workers parallelises requests at the OS process
level. `--preload` loads the 93 MB Keras model **once** in the master process before
forking, so workers share memory-mapped weights instead of each loading their own copy.

**How to test locally (Linux/macOS or inside Docker):**
```bash
cd "Project files"
gunicorn -w 4 -b 0.0.0.0:2222 --preload --timeout 120 app:app
# In another terminal, fire 4 concurrent requests:
for i in 1 2 3 4; do
  curl -s -X POST -F "file=@test.jpg" http://localhost:2222/api/predict &
done; wait
```

---

### Step 2 — Containerise the app (Docker)

**What changed:**
- `Project files/Dockerfile`: Python 3.11-slim base, installs requirements,
  copies app, exposes port 8080, runs 4 gunicorn workers
- `Project files/.dockerignore`: excludes `venv/`, `__pycache__/`, `static/uploads/`,
  `Notebooks/`, `.env`

**What bottleneck it solves:**
Eliminates "works on my machine" problems. The container packages the exact Python
version, system libs, and dependencies — making the app reproducible on any Docker host
and deployable to any container platform (Render, Fly.io, Railway, etc.).

**How to run:**
```bash
docker build -t waste-classifier:v1 ".\Project files"

# Run with model bind-mounted (model file is not baked into the image)
docker run -p 8080:8080 \
  -v "$(pwd)/Project files/vgg16.h5:/app/vgg16.h5" \
  -e MODEL_PATH=vgg16.h5 \
  -e SECRET_KEY=local-dev \
  waste-classifier:v1

curl http://localhost:8080/api/health
curl -X POST -F "file=@test.jpg" http://localhost:8080/api/predict
```

---

### Step 3 — Separate inference from the web layer (TF Serving)

**What changed:**
- `Project files/scripts/export_savedmodel.py`: converts `vgg16.h5` to TF SavedModel format
- `Project files/app_tfserving.py`: variant of `app.py` with in-process TF removed;
  `/api/predict` now POSTs preprocessed tensors to TF Serving's REST API
- `docker-compose.yml`: two-service stack (`tf-serving` + `web`)

**What bottleneck it solves:**
Running TF/Keras inference inside the web process means the web server stalls during
heavy GPU/CPU work. Separating inference into TF Serving decouples the two concerns:
- TF Serving is optimised for batching and hardware acceleration
- The web layer stays lightweight and can scale independently
- The model can be hot-swapped (new version under `models/waste_classifier/2/`)
  without restarting the web containers

**Architecture (Step 3):**
```
Browser → Flask/gunicorn (:8080) → TF Serving REST (:8501) → SavedModel
```

> ⚠️ This is a **local demonstration only** — tf-serving and web run as Docker
> services on your machine via docker-compose. Nothing is deployed to a cloud provider.

**How to run:**
```bash
# 1. Export the Keras model to SavedModel format (one-time)
cd "Project files"
python scripts/export_savedmodel.py

# 2. Start the stack
docker-compose up --build

# 3. Test (same response schema as v1)
curl -X POST -F "file=@test.jpg" http://localhost:8080/api/predict
```

---

### Step 4 — Async inference queue (Redpanda / Kafka + worker)

**What changed:**
- `docker-compose.yml`: adds `redpanda` (Kafka-compatible broker) and `worker` services
- `Project files/app_tfserving.py`:
  - `POST /api/predict/async` — saves image, publishes job to `waste-jobs` topic,
    returns `{ "request_id": "...", "status": "queued" }` immediately
  - `GET /api/result/<request_id>` — reads result from SQLite, returns result or
    `{ "status": "pending" }`
- `Project files/worker.py`: Kafka consumer that reads jobs, calls TF Serving,
  writes result to `results.db` (SQLite)
- **Synchronous `/api/predict` is unchanged** — both endpoints coexist

**What bottleneck it solves:**
Synchronous prediction means the HTTP connection stays open (potentially 1–5 seconds)
until TF Serving responds. Under high load, connections pile up. An async queue
decouples request acceptance from inference execution:
- Client gets an immediate `202 Accepted` + `request_id`
- Worker processes jobs at its own pace
- Client polls when ready — useful for batch uploads or mobile apps

> ⚠️ This is a **local demonstration only** — Redpanda runs as a Docker service via
> docker-compose. No Confluent Cloud, AWS MSK, or any paid broker is used.

**How to run:**
```bash
# Full stack (Steps 3 + 4)
docker-compose up --build

# Async flow:
curl -X POST -F "file=@test.jpg" http://localhost:8080/api/predict/async
# → { "request_id": "abc-123", "status": "queued" }

curl http://localhost:8080/api/result/abc-123
# → { "status": "pending" }  (while worker is running)
# → { "status": "complete", "prediction": {...}, "recycling_info": {...} }
```

---

### Running the full stack

**Prerequisites:**
- Docker Desktop installed and running
- `vgg16.h5` model file available

```bash
# Clone and enter the repo
git clone https://github.com/Silapareddy-Praveen-Kumar-Reddy/Waste-classification-using-Trasfer-Learning.git
cd Waste-classification-using-Trasfer-Learning

# Place your vgg16.h5 in Project files/
cp /path/to/vgg16.h5 "Project files/vgg16.h5"

# Export to SavedModel format (one-time setup)
cd "Project files"
python scripts/export_savedmodel.py
cd ..

# Start all 4 services
docker-compose up --build
```

| Service | URL | Purpose |
|---|---|---|
| Flask web app | http://localhost:8080 | UI + REST API |
| TF Serving REST | http://localhost:8501 | Inference endpoint |
| Redpanda (Kafka) | localhost:19092 | Async message broker |
| Worker | (no HTTP port) | Background consumer |

**Health check:**
```bash
curl http://localhost:8080/api/health
# → { "status": "healthy", "tf_serving_reachable": true, ... }
```

---

### New files added

| File | Step | Purpose |
|---|---|---|
| `Project files/Procfile` | 1 | Local gunicorn launch |
| `Project files/Dockerfile` | 2 | Container definition |
| `Project files/.dockerignore` | 2 | Build context exclusions |
| `Project files/scripts/export_savedmodel.py` | 3 | Model format conversion |
| `Project files/app_tfserving.py` | 3+4 | TF-Serving-backed app variant |
| `Project files/worker.py` | 4 | Kafka consumer / async worker |
| `docker-compose.yml` | 3+4 | Full local stack definition |
