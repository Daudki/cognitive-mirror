from flask import Flask, render_template, request, jsonify
from ml.model import predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            result = predict(text)

    return render_template("index.html", result=result)

@app.route("/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text") if payload else request.form.get("text")
    if not text:
        return jsonify({"error": "No text provided."}), 400

    result = predict(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
