# News Intelligence — NLP-Based News Classification System

## Project Overview

News Intelligence is a full-stack machine learning application that classifies news articles into four categories and provides NLP feature extraction and text similarity analysis.

The application uses FastAPI for backend services and React for the frontend interface.

## Features

- News article classification
- TF-IDF vectorization
- Bag of Words analysis
- N-gram analysis
- Keyword extraction
- Text similarity using cosine similarity
- Model-based feature importance ranking
- Interactive charts and visualizations

## Technology Stack

**Frontend:** React, Vite, Tailwind CSS, Axios, Recharts

**Backend:** Python, FastAPI, Pydantic

**Machine Learning:** scikit-learn, Logistic Regression, TF-IDF

**Dataset:** AG News

**Model Storage:** Local joblib artifact

## Dataset

The AG News dataset contains 120,000 training samples and 7,600 test samples across four categories:

1. World
2. Sports
3. Business
4. Technology (Sci/Tech)

For the initial model, 20,000 balanced training samples were selected, with 5,000 samples per category.

The complete 7,600-sample test split was used for evaluation.

## Model Development

The classification pipeline consists of:

1. Text preprocessing
2. TF-IDF feature extraction
3. Logistic Regression classification
4. Model evaluation
5. Model serialization

The TF-IDF vectorizer uses unigrams and bigrams with a maximum vocabulary size of 20,000 features.

## Model Performance

| Metric | Result |
|---|---|
| Test Accuracy | 89.63% |
| Macro F1-Score | 89.60% |
| Training Samples | 20,000 |
| Testing Samples | 7,600 |
| Number of Categories | 4 |

These results were obtained from the held-out AG News test dataset.

## Installation and Setup

### Backend

Open PowerShell in the project root.

```powershell
cd backend
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the model artifact is not already available, train it:

```powershell
python train_model.py
```

Start the API:

```powershell
python -m uvicorn app.main:app --reload
```

FastAPI runs at:

`http://localhost:8000`

API documentation:

`http://localhost:8000/docs`

### Frontend

Open another terminal in the project root.

```powershell
cd frontend
npm install
npm run dev
```

React runs at:

`http://localhost:5173`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend health |
| GET | `/api/nlp/status` | NLP model status |
| GET | `/api/nlp/metrics` | Model evaluation metrics |
| POST | `/api/nlp/classify` | Classify news |
| POST | `/api/nlp/tfidf` | Extract TF-IDF terms |
| POST | `/api/nlp/bow` | Bag of Words |
| POST | `/api/nlp/ngrams` | N-gram analysis |
| POST | `/api/nlp/keywords` | Keyword extraction |
| POST | `/api/nlp/similarity` | Text similarity |
| POST | `/api/nlp/features` | Model-based feature ranking |
| POST | `/api/nlp/analyze` | Combined NLP analysis |

## Testing

Run backend tests:

```powershell
cd backend
python -m pytest tests/ -v
```

Verify the frontend production build:

```powershell
cd frontend
npm run build
```

## Limitations

- The classifier supports four predefined news categories.
- Model confidence values are estimated probabilities, not guarantees.
- TF-IDF similarity measures lexical similarity rather than deep semantic similarity.
- Keyword extraction is based on trained TF-IDF weights.
- Feature importance ranking uses classifier coefficients rather than a dedicated feature-subset selection algorithm.
- The application currently uses local model storage rather than a database.

## Conclusion

The News Intelligence system demonstrates the integration of machine learning, NLP feature extraction, REST APIs, and a responsive web interface.

It provides an end-to-end implementation of news classification and the six required NLP feature extraction capabilities.