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
