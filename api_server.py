
"""
api_server.py — API Flask pour l'inférence SoH
Déployable sur Render, Railway, Fly.io, etc.
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─── Activation CORS manuel (sans flask-cors) ──────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ─── Chargement du modèle ──────────────────────────────────────
_model  = None
_scaler = None

MODEL_PATH  = os.environ.get("MODEL_PATH",  "models/lstm_soh.pth")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.pkl")
FEATURES    = ["Voltage_measured", "Current_measured",
               "Temperature_measured", "SoC"]


def load_artifacts():
    global _model, _scaler
    if _model is not None:
        return _model, _scaler

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler introuvable : {SCALER_PATH}")
    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    try:
        import torch
        from src.model import LSTMSoH
        _model = LSTMSoH(
            input_size =len(FEATURES),
            hidden_size=int(os.environ.get("HIDDEN_SIZE", 64)),
            num_layers =int(os.environ.get("NUM_LAYERS",  2)),
        )
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
    except ImportError:
        _model = "demo"

    return _model, _scaler


def predict_soh(sequence):
    model, scaler = load_artifacts()
    X        = np.array(sequence, dtype=np.float32)
    X_scaled = scaler.transform(X)[np.newaxis, :, :]

    if model == "demo":
        v, soc = X[0, 0], X[0, 3]
        return float(np.clip(60 + (v - 3.0) * 20 + soc * 0.25, 50, 115))

    import torch
    with torch.no_grad():
        soh = model(torch.from_numpy(X_scaled).float()).item()
    return round(float(soh), 4)


def soh_status(soh):
    return "healthy" if soh >= 80 else "warning" if soh >= 65 else "critical"


# ─── Routes ────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Battery SoH Predictor API",
        "version": "1.0.0",
        "status" : "running",
        "endpoints": {
            "GET  /health"        : "Health check",
            "POST /predict"       : "Single sequence prediction",
            "POST /predict/batch" : "Batch prediction",
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Body JSON :
    { "sequence": [[V, I, T, SoC], [V, I, T, SoC], ...] }

    Réponse :
    { "soh": 87.42, "status": "healthy", "unit": "%" }
    """
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True)
    if not data or "sequence" not in data:
        return jsonify({
            "error"  : "Body JSON requis avec la clé 'sequence'.",
            "example": {"sequence": [[3.72, -1.50, 28.0, 75.0],
                                     [3.70, -1.51, 27.6, 72.0],
                                     [3.68, -1.52, 27.1, 69.0]]}
        }), 400

    seq = data["sequence"]
    if not isinstance(seq, list) or len(seq) == 0:
        return jsonify({"error": "'sequence' doit être une liste non vide."}), 400

    for i, pt in enumerate(seq):
        if not isinstance(pt, list) or len(pt) != len(FEATURES):
            return jsonify({
                "error": f"Point [{i}] invalide — attendu {len(FEATURES)} valeurs : {FEATURES}"
            }), 400

    try:
        soh = predict_soh(seq)
        return jsonify({"soh": soh, "status": soh_status(soh),
                        "unit": "%", "n_bins": len(seq)})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500


@app.route("/predict/batch", methods=["POST", "OPTIONS"])
def predict_batch():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True)
    if not data or "sequences" not in data:
        return jsonify({"error": "Body JSON requis avec la clé 'sequences'."}), 400

    try:
        results = [
            {"soh": predict_soh(seq), "status": soh_status(predict_soh(seq)), "unit": "%"}
            for seq in data["sequences"]
        ]
        return jsonify({"predictions": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Point d'entrée ─────────────────────────────────────────────
if __name__ == "__main__":
    # ⚠️ Render injecte $PORT — ne jamais hardcoder le port
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"

    print(f"  Battery SoH API — port {port} | debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)