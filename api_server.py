"""
api_server.py — API Flask SoH minimale — garantie de démarrer
"""
import os
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ─── Fallback si modèle absent ────────────────────────────────
def predict_soh_fallback(sequence):
    X = np.array(sequence, dtype=np.float32)
    v   = float(X[-1, 0])
    soc = float(X[-1, 3])
    return float(np.clip(60 + (v - 3.0) * 20 + soc * 0.25, 50, 115))

def soh_status(soh):
    return "healthy" if soh >= 80 else "warning" if soh >= 65 else "critical"

# ─── Routes ───────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Battery SoH Predictor API",
        "version": "1.0.0",
        "status" : "running",
        "endpoints": {
            "GET  /health"       : "Health check",
            "POST /predict"      : "SoH prediction",
            "POST /predict/batch": "Batch prediction",
            "GET  /mongo/data"   : "MongoDB proxy (needs MONGO_URI)",
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
        return jsonify({"error": "Body JSON requis avec la cle 'sequence'."}), 400
    seq = data["sequence"]
    if not isinstance(seq, list) or len(seq) == 0:
        return jsonify({"error": "sequence doit etre une liste non vide."}), 400
    try:
        soh = predict_soh_fallback(seq)
        return jsonify({"soh": round(soh, 2), "status": soh_status(soh), "unit": "%"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/batch", methods=["POST", "OPTIONS"])
def predict_batch():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True)
    if not data or "sequences" not in data:
        return jsonify({"error": "Body JSON requis avec la cle 'sequences'."}), 400
    try:
        results = [
            {"soh": round(predict_soh_fallback(seq), 2),
             "status": soh_status(predict_soh_fallback(seq)), "unit": "%"}
            for seq in data["sequences"]
        ]
        return jsonify({"predictions": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mongo/data", methods=["GET", "OPTIONS"])
def mongo_data():
    if request.method == "OPTIONS":
        return "", 204
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        return jsonify({"error": "MONGO_URI non configure dans Render Environment Variables."}), 503
    db_name  = request.args.get("db",    "Energie")
    col_name = request.args.get("col",   "cycles")
    limit    = int(request.args.get("limit", 5000))
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
        docs   = list(client[db_name][col_name].find({}, {"_id": 0}).limit(limit))
        client.close()
        return jsonify(docs), 200
    except ImportError:
        return jsonify({"error": "pymongo non installe."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Démarrage local uniquement ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)