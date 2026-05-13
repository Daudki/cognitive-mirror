# Cognitive Mirror — Mind-State Analyzer

Cognitive Mirror is a lightweight Flask app that analyzes short text and returns an inferred emotion, sentiment, and a short human-readable "mind state" summary. The project pairs classical classifiers (TF-IDF + Logistic Regression) with an optional LLM to generate richer descriptions.

Key improvements made in this branch:
- Shared preprocessing between training and inference (`cognitive_mirror.preprocessing.clean_text`).
- Consolidated fine-grained emotion labels into a smaller canonical set for more reliable confidence scores.
- Frontend UX: top-emotions, tentative indicator for low-confidence readings, and smoother UI interactions.

Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train models (creates `models/model.pkl`):
   ```bash
   python -m ml.train
   ```
   Optionally include approved review data:
   ```bash
   python -m ml.train --use-approved
   ```
3. Run the web app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000/` in your browser.

API
POST `/predict` with JSON `{ "text": "..." }` returns:
```json
{
  "emotion": "joy",
  "sentiment": "positive",
  "confidence": 0.82,
  "mind_state": "The person appears joyful and optimistic.",
  "top_emotions": [{"emotion":"joy","probability":0.82}, ...]
}
```

LLM integration
Set `LLM_PROVIDER=api` and `OPENAI_API_KEY` to use OpenAI, or `LLM_PROVIDER=local` with a local model name. The code falls back to a `DummyAdapter` when no provider is configured.

How this improves predictions
- Consolidating labels reduces class cardinality so probability mass concentrates on plausible labels, increasing reported confidences.
- Shared `clean_text` ensures training and inference see the same tokenization.

Troubleshooting
- If predictions look incorrect, check `data/data.csv` for noisy labels and consider reviewing pending cases (`data/interaction_cases/pending.jsonl`).
- To debug locally, run the training script and inspect printed test predictions.

Tests
Run unit tests with `pytest`:
```bash
python -m pytest -q
```

