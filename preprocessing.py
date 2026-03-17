# =============================================================================
# src/preprocessing.py
# =============================================================================
# Responsabilité : Transformer les données brutes en tenseurs prêts pour LSTM.
#
# Étapes réalisées dans ce module :
#   1. Sélection des features pertinentes
#   2. Normalisation (StandardScaler) pour stabiliser l'apprentissage
#   3. Découpage en fenêtres glissantes (sliding windows)
#   4. Séparation train/test par cycle (pas aléatoire → respecte l'ordre temporel)
#
# POURQUOI NORMALISER ?
# ---------------------
# La tension (~3.5V) et le courant (~1A) ont des échelles très différentes.
# Sans normalisation, le gradient descend mal et le modèle converge lentement.
# Le StandardScaler centre (moyenne=0) et réduit (écart-type=1) chaque feature.
#
# POURQUOI DES FENETRES GLISSANTES ?
# ------------------------------------
# Un cycle contient N mesures (bins). Plutôt que de donner tout le cycle d'un coup
# (séquence trop longue, données rares), on découpe en sous-séquences de taille W.
# Chaque sous-séquence → un exemple d'entraînement avec le SoH du cycle comme cible.
# Résultat : on multiplie le nombre d'exemples et on apprend des motifs locaux.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import pickle
import os


# Features utilisées comme entrées du LSTM
# On inclut le cycle_number car c'est un proxy direct du vieillissement
FEATURES = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "SoC",
    "cycle_number"
]

# Variable cible
TARGET = "SoH"


def selectionner_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extrait les colonnes features (X) et la colonne cible (y) du DataFrame.

    Paramètres
    ----------
    df : pd.DataFrame  DataFrame brut complet

    Retourne
    --------
    X : pd.DataFrame   Uniquement les colonnes features
    y : pd.Series      La colonne SoH
    """
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    print(f"[preprocessing] Features selectionnees : {FEATURES}")
    print(f"[preprocessing] Cible : {TARGET}")
    return X, y


def normaliser(X_train: np.ndarray,
               X_test: np.ndarray,
               chemin_scaler: str = "models/scaler.pkl"
               ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalise les features avec un StandardScaler.

    IMPORTANT : le scaler est AJUSTE sur le train uniquement,
    puis APPLIQUE sur le test. Cela évite la fuite de données (data leakage).
    Si on ajustait le scaler sur tout le dataset, les stats du test
    contamineraient l'entraînement → évaluation biaisée.

    Paramètres
    ----------
    X_train : array (n_train, n_features)
    X_test  : array (n_test, n_features)
    chemin_scaler : où sauvegarder le scaler pour réutilisation (dashboard)

    Retourne
    --------
    X_train_norm : array normalisé
    X_test_norm  : array normalisé
    scaler       : objet StandardScaler ajusté
    """

    scaler = StandardScaler()

    # fit_transform sur train : calcule mean/std ET transforme
    X_train_norm = scaler.fit_transform(X_train)

    # transform seulement sur test : applique les stats du train
    X_test_norm = scaler.transform(X_test)

    # Sauvegarde du scaler pour le dashboard (prédictions temps réel)
    os.makedirs(os.path.dirname(chemin_scaler), exist_ok=True)
    with open(chemin_scaler, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[preprocessing] Normalisation terminee | Scaler sauvegarde : {chemin_scaler}")
    return X_train_norm, X_test_norm, scaler


def creer_fenetres_glissantes(
    X: np.ndarray,
    y: np.ndarray,
    taille_fenetre: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Découpe une séquence en sous-séquences (fenêtres glissantes).

    Fonctionnement :
    ----------------
    Pour une séquence de 10 mesures et une fenêtre de taille 3 :
      - Fenêtre 1 : mesures [0,1,2] → cible y[2]
      - Fenêtre 2 : mesures [1,2,3] → cible y[3]
      - Fenêtre 3 : mesures [2,3,4] → cible y[4]
      - ...
      - Fenêtre 8 : mesures [7,8,9] → cible y[9]

    Le LSTM reçoit une matrice (taille_fenetre, n_features) pour chaque exemple.
    La cible est le SoH du dernier pas de la fenêtre (SoH constant par cycle).

    Paramètres
    ----------
    X : array (N, n_features)  Séquence de features normalisées
    y : array (N,)             Valeurs de SoH correspondantes
    taille_fenetre : int       Nombre de bins par fenêtre (hyperparamètre)

    Retourne
    --------
    X_fen : array (N - W + 1, W, n_features)  Fenêtres 3D pour LSTM
    y_fen : array (N - W + 1,)                Cibles associées
    """

    X_fen, y_fen = [], []

    # On glisse la fenêtre sur toute la séquence
    for i in range(len(X) - taille_fenetre + 1):
        # Extrait W mesures consécutives
        fenetre = X[i : i + taille_fenetre]
        # La cible est le SoH du dernier bin de la fenêtre
        cible = y[i + taille_fenetre - 1]
        X_fen.append(fenetre)
        y_fen.append(cible)

    return np.array(X_fen), np.array(y_fen)


def preparer_par_cycle(
    df: pd.DataFrame,
    taille_fenetre: int = 3,
    ratio_test: float = 0.2
) -> dict:
    """
    Pipeline complet de préparation des données.

    Stratégie de séparation train/test :
    -------------------------------------
    On sépare par CYCLE, pas aléatoirement.
    Les N derniers cycles (ratio_test%) constituent le jeu de test.
    Cela simule une situation réelle : le modèle est entraîné sur
    des cycles anciens, puis évalué sur des cycles futurs inconnus.
    Une séparation aléatoire des lignes fuiterait les infos du futur
    dans le passé → évaluation trop optimiste.

    Paramètres
    ----------
    df           : DataFrame brut
    taille_fenetre : taille des fenêtres glissantes
    ratio_test   : fraction des cycles réservée au test (ex: 0.2 = 20%)

    Retourne
    --------
    dict avec les clés :
        X_train, y_train : données d'entraînement (numpy arrays)
        X_test, y_test   : données de test
        scaler           : scaler ajusté sur le train
        cycles_train     : liste des numéros de cycles utilisés pour train
        cycles_test      : liste des numéros de cycles utilisés pour test
    """

    # 1. Récupérer la liste ordonnée des cycles
    cycles = sorted(df["cycle_number"].unique())
    n_cycles = len(cycles)
    n_test = max(1, int(n_cycles * ratio_test))  # au moins 1 cycle en test

    cycles_train = cycles[: n_cycles - n_test]
    cycles_test  = cycles[n_cycles - n_test :]

    print(f"\n[preprocessing] Repartition cycles :")
    print(f"   Train : {len(cycles_train)} cycles → {cycles_train}")
    print(f"   Test  : {len(cycles_test)} cycles  → {cycles_test}")

    # 2. Séparer le DataFrame selon les cycles
    df_train = df[df["cycle_number"].isin(cycles_train)].reset_index(drop=True)
    df_test  = df[df["cycle_number"].isin(cycles_test)].reset_index(drop=True)

    # 3. Extraire features et cibles
    X_train_raw = df_train[FEATURES].values
    y_train_raw = df_train[TARGET].values
    X_test_raw  = df_test[FEATURES].values
    y_test_raw  = df_test[TARGET].values

    # 4. Normalisation (scaler ajusté sur train uniquement)
    X_train_norm, X_test_norm, scaler = normaliser(
        X_train_raw, X_test_raw, chemin_scaler="models/scaler.pkl"
    )

    # 5. Découpage en fenêtres glissantes par cycle
    #    IMPORTANT : on crée les fenêtres DANS chaque cycle séparément
    #    pour éviter qu'une fenêtre enjambe deux cycles différents.
    X_train_fen, y_train_fen = _fenetrer_par_cycle(
        df_train, X_train_norm, y_train_raw, taille_fenetre
    )
    X_test_fen, y_test_fen = _fenetrer_par_cycle(
        df_test, X_test_norm, y_test_raw, taille_fenetre
    )

    print(f"\n[preprocessing] Apres fenetrage (W={taille_fenetre}) :")
    print(f"   X_train : {X_train_fen.shape}  →  (exemples, fenetre, features)")
    print(f"   X_test  : {X_test_fen.shape}")
    print(f"   y_train : {y_train_fen.shape}")
    print(f"   y_test  : {y_test_fen.shape}\n")

    return {
        "X_train": X_train_fen,
        "y_train": y_train_fen,
        "X_test": X_test_fen,
        "y_test": y_test_fen,
        "scaler": scaler,
        "cycles_train": cycles_train,
        "cycles_test": cycles_test,
    }


def _fenetrer_par_cycle(
    df_subset: pd.DataFrame,
    X_norm: np.ndarray,
    y: np.ndarray,
    taille_fenetre: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fonction interne : applique le fenetrage cycle par cycle.

    En travaillant cycle par cycle, on garantit qu'aucune fenêtre
    ne mélange les mesures de deux cycles différents (ce qui serait
    physiquement incohérent : chaque cycle est indépendant).

    Paramètres
    ----------
    df_subset    : DataFrame filtré (train ou test)
    X_norm       : array normalisé correspondant
    y            : array des cibles SoH
    taille_fenetre : taille de fenêtre

    Retourne
    --------
    Tuple (X_fenetres, y_fenetres) concatenés pour tous les cycles
    """
    X_all, y_all = [], []

    cycles = sorted(df_subset["cycle_number"].unique())

    for cycle in cycles:
        # Indices des lignes appartenant à ce cycle
        masque = df_subset["cycle_number"].values == cycle
        idx = np.where(masque)[0]

        X_cycle = X_norm[idx]
        y_cycle = y[idx]

        # Si le cycle est trop court pour créer au moins une fenêtre, on skip
        if len(X_cycle) < taille_fenetre:
            print(f"   [WARNING] Cycle {cycle} trop court "
                  f"({len(X_cycle)} bins < fenetre {taille_fenetre}) → ignore")
            continue

        X_fen, y_fen = creer_fenetres_glissantes(X_cycle, y_cycle, taille_fenetre)
        X_all.append(X_fen)
        y_all.append(y_fen)

    if len(X_all) == 0:
        raise ValueError("Aucune fenetre generee. Verifiez la taille de fenetre "
                         "par rapport au nombre de bins par cycle.")

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)