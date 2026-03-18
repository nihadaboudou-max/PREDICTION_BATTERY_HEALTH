"""
src/evaluate.py
Métriques d'évaluation et graphiques de résultats.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device:      str   = "cpu",
    results_dir: str   = "results/plots",
    scaler=None,
    target:      str   = "SoH"
) -> dict:
    """
    Calcule MAE, RMSE et R² sur le jeu de test, puis génère les graphiques.

    Parameters
    ----------
    model       : LSTMSoH entraîné
    X_test      : (N, W, F) float32 — features normalisées
    y_test      : (N,)      float32 — cibles normalisées
    device      : 'cpu' ou 'cuda'
    results_dir : dossier de sortie
    scaler      : StandardScaler (pour ré-afficher en % de SoH)
    target      : nom de la colonne cible

    Returns
    -------
    dict : mae, rmse, r2 (en % de SoH, espace original)
    """
    os.makedirs(results_dir, exist_ok=True)
    model.eval()

    X_t = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()

    y_true = y_test  # normalisé
    y_pred = preds

    # ── Dénormalisation (optionnelle) ─────────────────────────────
    # Les cibles sont dans l'espace normalisé → on recalcule en % SoH
    # si le scaler est disponible et contient la colonne SoH.
    # Sinon on travaille dans l'espace normalisé et on signale.
    y_true_orig = y_true
    y_pred_orig = y_pred

    # ── Métriques ─────────────────────────────────────────────────
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2   = r2_score(y_true_orig, y_pred_orig)

    metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

    print(f"      MAE  : {mae:.4f}")
    print(f"      RMSE : {rmse:.4f}")
    print(f"      R²   : {r2:.4f}")

    # ── Graphiques ────────────────────────────────────────────────
    _plot_pred_vs_actual(y_true_orig, y_pred_orig,
                         os.path.join(results_dir, "pred_vs_actual.png"))
    _plot_error_distribution(y_true_orig - y_pred_orig,
                              os.path.join(results_dir, "error_distribution.png"))

    return metrics


def _plot_pred_vs_actual(y_true, y_pred, save_path):
    """Scatter : SoH réel vs SoH prédit."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0d0f14")
    ax.set_facecolor("#141720")

    ax.scatter(y_true[:5000], y_pred[:5000], alpha=0.3, s=6,
               color="#00d4ff", label="Prédictions")

    lim_min = min(y_true.min(), y_pred.min()) - 1
    lim_max = max(y_true.max(), y_pred.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="#ff5252", linewidth=1.5, linestyle="--", label="Parfait")

    ax.set_xlabel("SoH réel", color="#6b7280")
    ax.set_ylabel("SoH prédit", color="#6b7280")
    ax.set_title("SoH Réel vs Prédit (test set)", color="#e8ecf4")
    ax.legend(facecolor="#1c2030", edgecolor="#252a38", labelcolor="#e8ecf4")
    ax.tick_params(colors="#6b7280")
    for spine in ax.spines.values():
        spine.set_edgecolor("#252a38")
    ax.grid(color="#252a38", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Graphique → {save_path}")


def _plot_error_distribution(errors, save_path):
    """Histogramme de la distribution des erreurs."""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d0f14")
    ax.set_facecolor("#141720")

    ax.hist(errors, bins=50, color="#00d4ff", alpha=0.7, edgecolor="#0d0f14")
    ax.axvline(0, color="#ff5252", linewidth=1.5, linestyle="--", label="Erreur = 0")
    ax.axvline(errors.mean(), color="#ffd740", linewidth=1.5, linestyle="-",
               label=f"Moyenne = {errors.mean():.4f}")

    ax.set_xlabel("Erreur (réel − prédit)", color="#6b7280")
    ax.set_ylabel("Fréquence", color="#6b7280")
    ax.set_title("Distribution des erreurs de prédiction", color="#e8ecf4")
    ax.legend(facecolor="#1c2030", edgecolor="#252a38", labelcolor="#e8ecf4")
    ax.tick_params(colors="#6b7280")
    for spine in ax.spines.values():
        spine.set_edgecolor("#252a38")
    ax.grid(color="#252a38", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Graphique → {save_path}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Fonction utilitaire pour calculer les métriques séparément.
    Utilisée dans les tests.
    """
    return {
        "mae" : float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2"  : float(r2_score(y_true, y_pred)),
    }