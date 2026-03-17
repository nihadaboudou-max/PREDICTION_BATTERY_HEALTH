# =============================================================================
# main.py
# =============================================================================
# Point d'entrée principal du projet de prédiction du SoH des batteries.
#
# Ce fichier orchestre tout le pipeline :
#   1. Chargement et validation des données
#   2. Prétraitement (normalisation + fenêtres glissantes)
#   3. Création et entraînement du modèle LSTM
#   4. Évaluation et génération du rapport
#   5. Sauvegarde du modèle pour réutilisation (dashboard)
#
# UTILISATION
# -----------
#   python main.py
#
# Les résultats sont sauvegardés dans :
#   - models/lstm_soh.pth      → poids du modèle
#   - models/scaler.pkl        → scaler de normalisation
#   - results/plots/           → graphiques d'évaluation
# =============================================================================

import os
import sys

# Ajoute le dossier courant au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader   import charger_donnees, afficher_apercu, verifier_qualite
from preprocessing import preparer_par_cycle
from model         import creer_modele, sauvegarder_modele
from train         import creer_dataloaders, entrainer
from evaluate      import rapport_complet


# =========================================================================
# HYPERPARAMETRES DU PROJET
# =========================================================================
# Ces valeurs sont ici centralisées pour faciliter les expérimentations.
# Modifier ces constantes change tout le comportement du pipeline.
# =========================================================================

CHEMIN_DONNEES   = "battery_health_dataset.csv"  # Fichier CSV source
TAILLE_FENETRE   = 3      # Nombre de bins par fenêtre glissante
RATIO_TEST       = 0.2    # 20% des cycles réservés pour l'évaluation

# Architecture LSTM
INPUT_SIZE       = 5      # Nombre de features : Voltage, Current, Temp, SoC, cycle
HIDDEN_SIZE      = 64     # Unités cachées du LSTM
NUM_LAYERS       = 2      # Couches LSTM empilées
DROPOUT          = 0.2    # Régularisation

# Entraînement
BATCH_SIZE       = 16     # Taille des mini-batches
N_EPOCHS         = 100    # Nombre maximum d'epochs
LEARNING_RATE    = 0.001  # Taux d'apprentissage initial
PATIENCE         = 15     # Early stopping : tolérance (epochs sans amélioration)


def main():
    """
    Pipeline complet d'entraînement et d'évaluation du modèle LSTM SoH.
    """

    print("\n" + "#" * 65)
    print("  PREDICTION DU STATE OF HEALTH (SoH) DES BATTERIES")
    print("  Modele : LSTM | Approche : Fenetres glissantes")
    print("#" * 65 + "\n")

    # ----------------------------------------------------------------
    # ETAPE 1 : Chargement et validation des données
    # ----------------------------------------------------------------
    print(">>> ETAPE 1 : Chargement des donnees")
    df = charger_donnees(CHEMIN_DONNEES)
    afficher_apercu(df)
    verifier_qualite(df)

    # ----------------------------------------------------------------
    # ETAPE 2 : Prétraitement
    # ----------------------------------------------------------------
    print(">>> ETAPE 2 : Pretraitement")
    donnees = preparer_par_cycle(
        df=df,
        taille_fenetre=TAILLE_FENETRE,
        ratio_test=RATIO_TEST
    )

    X_train = donnees["X_train"]  # (N_train, W, n_features)
    y_train = donnees["y_train"]  # (N_train,)
    X_test  = donnees["X_test"]   # (N_test, W, n_features)
    y_test  = donnees["y_test"]   # (N_test,)

    # ----------------------------------------------------------------
    # ETAPE 3 : Création du modèle
    # ----------------------------------------------------------------
    print(">>> ETAPE 3 : Creation du modele LSTM")
    modele = creer_modele(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    # ----------------------------------------------------------------
    # ETAPE 4 : Création des DataLoaders
    # ----------------------------------------------------------------
    print(">>> ETAPE 4 : Preparation des batches")
    loader_train, loader_test = creer_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=BATCH_SIZE
    )

    # ----------------------------------------------------------------
    # ETAPE 5 : Entraînement
    # ----------------------------------------------------------------
    print(">>> ETAPE 5 : Entrainement")
    historique = entrainer(
        modele=modele,
        loader_train=loader_train,
        loader_test=loader_test,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        patience=PATIENCE
    )

    # ----------------------------------------------------------------
    # ETAPE 6 : Sauvegarde du modèle
    # ----------------------------------------------------------------
    print(">>> ETAPE 6 : Sauvegarde")
    sauvegarder_modele(modele, chemin="models/lstm_soh.pth")

    # ----------------------------------------------------------------
    # ETAPE 7 : Évaluation et rapport
    # ----------------------------------------------------------------
    print(">>> ETAPE 7 : Evaluation")
    metriques = rapport_complet(
        modele=modele,
        X_test=X_test,
        y_test=y_test,
        historique=historique,
        dossier_sortie="results/plots"
    )

    # ----------------------------------------------------------------
    # BILAN FINAL
    # ----------------------------------------------------------------
    print("\n" + "#" * 65)
    print("  BILAN FINAL")
    print("#" * 65)
    print(f"  MAE  : {metriques['MAE']:.4f}%")
    print(f"  RMSE : {metriques['RMSE']:.4f}%")
    print(f"  R²   : {metriques['R2']:.4f}")
    print(f"\n  Modele sauvegarde : models/lstm_soh.pth")
    print(f"  Scaler sauvegarde : models/scaler.pkl")
    print(f"  Graphiques        : results/plots/")
    print(f"\n  Ouvrez dashboard.html pour les predictions en temps reel.")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    main()