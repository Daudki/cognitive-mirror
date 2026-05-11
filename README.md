# Cognitive Mirror / mindReader

This project is a simple Flask-based mindstate inference tool that predicts emotion, sentiment, and confidence from a text input.

## What changed
- Added a `/predict` JSON API endpoint for programmatic access.
- Improved model persistence with a combined `models/model.pkl` checkpoint.
- Added a frontend script for AJAX inference and a cleaner UI flow.
- Added a mind-state summary field for more natural results.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the models:
   ```bash
   python ml/train.py
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open the browser at `http://127.0.0.1:5000/`.

## API
POST `/predict`

Request JSON:
```json
{ "text": "I am feeling very confused today" }
```

Response JSON:
```json
{
  "emotion": "fear",
  "sentiment": "negative",
  "confidence": 0.85,
  "mind_state": "The person is anxious, uneasy, and sensing uncertainty."
}
```
