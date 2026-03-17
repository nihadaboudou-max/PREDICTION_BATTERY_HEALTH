# =============================================================================
# src/evaluate.py
# =============================================================================
# Responsabilité : Évaluer le modèle entraîné et générer les visualisations.
#
# Métriques calculées :
#   - MAE  : Mean Absolute Error → erreur moyenne en % de SoH
#   - RMSE : Root Mean Squared Error → pénalise les grandes erreurs
#   - R²   : coefficient de détermination → proportion de variance expliquée
#
# INTERPRETATION DES METRIQUES
# ------------------------------
# MAE  : "En moyenne, le modèle se trompe de X% sur le SoH"
# RMSE : toujours ≥ MAE. Si RMSE >> MAE, des erreurs ponctuelles sont grandes.
# R²   : 1.0 = parfait | 0.0 = aussi bien qu'une moyenne constante | <0 = inutile
#
# GRAPHIQUES GENERÉS
# -------------------
# 1. Courbes de loss (train vs test) → détection overfitting
# 2. SoH prédit vs SoH réel (scatter) → qualité globale
# 3. Histogramme des erreurs → distribution des résidus
# 4. SoH prédit vs réel dans le temps → évolution cycle par cycle
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error
from typing import List
import os


def predire(
    modele: nn.Module,
    X: np.ndarray
) -> np.ndarray:
    """
    Effectue les prédictions du modèle sur un jeu de données.

    Paramètres
    ----------
    modele : LSTMBatterie entraîné et en mode eval()
    X      : array de forme (N, seq_len, n_features)

    Retourne
    --------
    y_pred : array de forme (N,)  — SoH prédit pour chaque fenêtre
    """
    modele.eval()
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        preds = modele(X_tensor)  # (N, 1)

    return preds.squeeze().numpy()  # → (N,)


def calculer_metriques(y_reel: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcule MAE, RMSE et R².

    Paramètres
    ----------
    y_reel : valeurs réelles de SoH
    y_pred : valeurs prédites par le modèle

    Retourne
    --------
    dict : {"MAE": float, "RMSE": float, "R2": float}
    """
    mae  = mean_absolute_error(y_reel, y_pred)
    rmse = np.sqrt(np.mean((y_reel - y_pred) ** 2))
    r2   = r2_score(y_reel, y_pred)

    print("\n" + "=" * 50)
    print("  RESULTATS DE L'EVALUATION")
    print("=" * 50)
    print(f"  MAE  : {mae:.4f}%  (erreur absolue moyenne)")
    print(f"  RMSE : {rmse:.4f}%  (erreur quadratique moyenne)")
    print(f"  R²   : {r2:.4f}   (proportion de variance expliquee)")
    print("=" * 50 + "\n")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def tracer_courbes_loss(
    train_losses: List[float],
    test_losses: List[float],
    best_epoch: int,
    dossier_sortie: str = "results/plots"
) -> None:
    """
    Trace les courbes d'apprentissage (loss train et test).

    Ces courbes permettent de diagnostiquer :
    - Underfitting : les deux pertes restent hautes
    - Overfitting  : la loss train descend mais la loss test remonte
    - Bonne convergence : les deux descendent et se stabilisent

    Paramètres
    ----------
    train_losses  : liste des losses train par epoch
    test_losses   : liste des losses test par epoch
    best_epoch    : epoch de la meilleure loss test (marquée sur le graphe)
    dossier_sortie : où sauvegarder le graphique
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Loss Train (MSE)", color="#2196F3", linewidth=2)
    ax.plot(epochs, test_losses,  label="Loss Test (MSE)",  color="#F44336", linewidth=2)

    # Ligne verticale sur la meilleure epoch
    ax.axvline(x=best_epoch, color="green", linestyle="--",
               alpha=0.7, label=f"Meilleure epoch ({best_epoch})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Courbes d'apprentissage — LSTM SoH", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Échelle log pour mieux voir la convergence

    plt.tight_layout()
    chemin = os.path.join(dossier_sortie, "learning_curves.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Graphique sauvegarde : {chemin}")


def tracer_predictions(
    y_reel: np.ndarray,
    y_pred: np.ndarray,
    metriques: dict,
    dossier_sortie: str = "results/plots"
) -> None:
    """
    Génère un tableau de bord visuel des performances du modèle.

    Contient :
    1. Scatter plot : SoH prédit vs réel (points sur la diagonale = parfait)
    2. Histogramme des erreurs (résidus) : centré sur 0 = bon modèle
    3. Série temporelle : évolution du SoH prédit vs réel

    Paramètres
    ----------
    y_reel       : valeurs réelles de SoH
    y_pred       : valeurs prédites
    metriques    : dict avec MAE, RMSE, R2
    dossier_sortie : dossier de sauvegarde
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    erreurs = y_reel - y_pred  # Résidus (erreurs de prédiction)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ---- Graphique 1 : Scatter Prédit vs Réel ----
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(y_reel, y_pred, alpha=0.6, color="#2196F3", edgecolors="white",
                linewidth=0.5, s=50, label="Prédictions")

    # Ligne idéale y = x (prédiction parfaite)
    vmin, vmax = min(y_reel.min(), y_pred.min()), max(y_reel.max(), y_pred.max())
    ax1.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=2, label="Idéal (y=x)")

    ax1.set_xlabel("SoH réel (%)", fontsize=11)
    ax1.set_ylabel("SoH prédit (%)", fontsize=11)
    ax1.set_title(f"Prédit vs Réel\nR² = {metriques['R2']:.4f}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ---- Graphique 2 : Distribution des erreurs ----
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(erreurs, bins=20, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Erreur nulle")
    ax2.axvline(x=erreurs.mean(), color="orange", linestyle="-", linewidth=2,
                label=f"Moyenne ({erreurs.mean():.3f}%)")
    ax2.set_xlabel("Erreur de prédiction (%)", fontsize=11)
    ax2.set_ylabel("Fréquence", fontsize=11)
    ax2.set_title(f"Distribution des résidus\nMAE = {metriques['MAE']:.4f}%",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ---- Graphique 3 : Série temporelle ----
    ax3 = fig.add_subplot(gs[2])
    indices = range(len(y_reel))
    ax3.plot(indices, y_reel, color="#F44336", linewidth=2, label="SoH réel", alpha=0.8)
    ax3.plot(indices, y_pred, color="#2196F3", linewidth=2, label="SoH prédit",
             linestyle="--", alpha=0.8)
    ax3.fill_between(indices, y_reel, y_pred, alpha=0.15, color="gray")
    ax3.set_xlabel("Fenêtre temporelle (index)", fontsize=11)
    ax3.set_ylabel("SoH (%)", fontsize=11)
    ax3.set_title(f"Évolution du SoH\nRMSE = {metriques['RMSE']:.4f}%",
                  fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Titre global
    fig.suptitle("Évaluation du modèle LSTM — Prédiction du SoH des batteries",
                 fontsize=14, fontweight="bold", y=1.02)

    chemin = os.path.join(dossier_sortie, "evaluation.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Graphique sauvegarde : {chemin}")


def rapport_complet(
    modele: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    historique: dict,
    dossier_sortie: str = "results/plots"
) -> dict:
    """
    Génère le rapport d'évaluation complet.

    Combine :
    - Prédictions sur le jeu de test
    - Calcul des métriques
    - Génération de tous les graphiques

    Paramètres
    ----------
    modele       : LSTM entraîné
    X_test       : données de test (numpy)
    y_test       : cibles de test (numpy)
    historique   : dict retourné par train.entrainer()
    dossier_sortie : dossier pour les graphiques

    Retourne
    --------
    dict : métriques {MAE, RMSE, R2} + y_pred
    """
    print("[evaluate] Generation du rapport d'evaluation...")

    # 1. Prédictions
    y_pred = predire(modele, X_test)

    # 2. Métriques
    metriques = calculer_metriques(y_test, y_pred)

    # 3. Graphiques
    tracer_courbes_loss(
        historique["train_losses"],
        historique["test_losses"],
        historique["best_epoch"],
        dossier_sortie
    )
    tracer_predictions(y_test, y_pred, metriques, dossier_sortie)

    metriques["y_pred"] = y_pred
    return metriques