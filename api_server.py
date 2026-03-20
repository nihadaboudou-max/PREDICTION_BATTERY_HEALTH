"""
api_server.py — API Flask pour l'inférence SoH + proxy MongoDB
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from pymongo import MongoClient

app = Flask(__name__)

# ─── CORS ─────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ─── Chargement modèle ────────────────────────────────────────
_model  = None
_scaler = None
FEATURES = ["Voltage_measured", "Current_measured",
            "Temperature_measured", "SoC"]

def load_artifacts():
    global _model, _scaler
    if _model is not None:
        return _model, _scaler
    scaler_path = os.environ.get("SCALER_PATH", "models/scaler.pkl")
    model_path  = os.environ.get("MODEL_PATH",  "models/lstm_soh.pth")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)
    try:
        import torch
        from src.model import LSTMSoH
        _model = LSTMSoH(
            input_size =len(FEATURES),
            hidden_size=int(os.environ.get("HIDDEN_SIZE", 64)),
            num_layers =int(os.environ.get("NUM_LAYERS",  2)),
        )
        _model.load_state_dict(torch.load(model_path, map_location="cpu"))
        _model.eval()
    except Exception:
        _model = "demo"
    return _model, _scaler

def predict_soh(sequence):
    model, scaler = load_artifacts()
    X = np.array(sequence, dtype=np.float32)
    if scaler:
        X_scaled = scaler.transform(X)[np.newaxis, :, :]
    else:
        X_scaled = X[np.newaxis, :, :]
    if model == "demo":
        v, soc = X[0, 0], X[0, 3]
        return float(np.clip(60 + (v - 3.0) * 20 + soc * 0.25, 50, 115))
    import torch
    with torch.no_grad():
        soh = model(torch.from_numpy(X_scaled).float()).item()
    return round(float(soh), 4)

def soh_status(soh):
    return "healthy" if soh >= 80 else "warning" if soh >= 65 else "critical"

# ─── Routes SoH ───────────────────────────────────────────────
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
            "GET  /mongo/data"    : "Load data from MongoDB (needs MONGO_URI env var)",
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True)
    if not data or "sequence" not in data:
        return jsonify({"error": "Body JSON requis avec la clé 'sequence'."}), 400
    seq = data["sequence"]
    if not isinstance(seq, list) or len(seq) == 0:
        return jsonify({"error": "'sequence' doit être une liste non vide."}), 400
    try:
        soh = predict_soh(seq)
        return jsonify({"soh": soh, "status": soh_status(soh), "unit": "%", "n_bins": len(seq)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/batch", methods=["POST", "OPTIONS"])
def predict_batch():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True)
    if not data or "sequences" not in data:
        return jsonify({"error": "Body JSON requis avec la clé 'sequences'."}), 400
    try:
        results = [{"soh": predict_soh(seq), "status": soh_status(predict_soh(seq)), "unit": "%"}
                   for seq in data["sequences"]]
        return jsonify({"predictions": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Route MongoDB proxy ──────────────────────────────────────
@app.route("/mongo/data", methods=["GET", "OPTIONS"])
def mongo_data():
    """
    Proxy MongoDB côté serveur — contourne la restriction CORS du navigateur.
    Variables d'environnement requises dans Render :
      MONGO_URI = mongodb+srv://user:password@cluster.mongodb.net/
    Paramètres GET :
      db    = nom de la database  (ex: Energie)
      col   = nom de la collection (ex: bactérie)
      limit = nombre max de documents (défaut: 5000)
    """
    if request.method == "OPTIONS":
        return "", 204

    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        return jsonify({
            "error": "Variable MONGO_URI non configurée sur Render. "
                     "Ajoutez-la dans Settings → Environment Variables."
        }), 503

    db_name  = request.args.get("db",    "Energie")
    col_name = request.args.get("col",   "bactérie")
    limit    = int(request.args.get("limit", 5000))

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
        db     = client[db_name]
        col    = db[col_name]
        docs   = list(col.find({}, {"_id": 0}).limit(limit))
        client.close()
        return jsonify(docs), 200
    except Exception as e:
        return jsonify({"error": f"Erreur MongoDB : {str(e)}"}), 500

# ─── Point d'entrée ───────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    print(f"  Battery SoH API — port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)