#!/usr/bin/env python
import os
import json
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an expert psychologist and emotion classifier. Analyze the text and return ONLY valid JSON with these exact keys:
- emotion: one of [joy, sadness, anger, fear, surprise, disgust, neutral, pride, shame, relief, hope, love, envy, guilt, boredom, confusion, frustration, nostalgia, contentment, excitement]
- sentiment: one of [positive, negative, neutral, mixed]
- confidence: float between 0 and 1
- mind_state: a one-sentence empathetic description of the person's mental state

No explanations. No markdown. Pure JSON only."""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        result = json.loads(raw)

        return jsonify({
            "emotion": result.get("emotion", "neutral"),
            "sentiment": result.get("sentiment", "neutral"),
            "confidence": result.get("confidence", 0.5),
            "mind_state": result.get("mind_state", "Unable to determine mental state."),
            "top_emotions": [],
        })

    except json.JSONDecodeError:
        return jsonify({
            "emotion": "neutral",
            "sentiment": "neutral",
            "confidence": 0.0,
            "mind_state": "Analysis produced an unreadable result.",
            "top_emotions": [],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)