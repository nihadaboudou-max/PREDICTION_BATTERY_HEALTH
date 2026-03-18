"""
src/preprocessing.py
Normalisation StandardScaler + fenêtres glissantes.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    scaler_path: str = "models/scaler.pkl"
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Ajuste un StandardScaler UNIQUEMENT sur df_train, puis transforme
    df_train et df_test. Évite le data leakage.

    Parameters
    ----------
    df_train    : DataFrame d'entraînement
    df_test     : DataFrame de test
    features    : colonnes à normaliser
    scaler_path : chemin de sauvegarde du scaler

    Returns
    -------
    df_train_n, df_test_n, scaler
    """
    scaler = StandardScaler()
    scaler.fit(df_train[features].values)

    df_train_n = df_train.copy()
    df_test_n  = df_test.copy()

    df_train_n[features] = scaler.transform(df_train[features].values)
    df_test_n[features]  = scaler.transform(df_test[features].values)

    # Sauvegarde du scaler
    os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"      Scaler sauvegardé → {scaler_path}")

    return df_train_n, df_test_n, scaler


def create_windows(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Découpe chaque cycle en fenêtres glissantes de taille window_size.
    Les fenêtres ne chevauchent jamais deux cycles différents.

    Parameters
    ----------
    df          : DataFrame normalisé avec colonnes features + target + cycle_number
    features    : colonnes d'entrée
    target      : colonne cible
    window_size : nombre de bins par séquence (W)

    Returns
    -------
    X : (N, W, n_features)  — séquences d'entrée
    y : (N,)                — valeurs cibles correspondantes
    """
    X_list, y_list = [], []

    for cycle_id, group in df.groupby("cycle_number"):
        group = group.reset_index(drop=True)
        feat_vals = group[features].values       # (L, n_features)
        target_vals = group[target].values       # (L,)

        n = len(group)
        if n < window_size:
            # Cycle trop court : on l'ignore
            continue

        for start in range(n - window_size + 1):
            X_list.append(feat_vals[start : start + window_size])
            # Cible = SoH du dernier bin de la fenêtre
            y_list.append(target_vals[start + window_size - 1])

    if not X_list:
        raise ValueError(
            f"Aucune fenêtre générée. "
            f"Vérifiez que les cycles ont au moins {window_size} bins."
        )

    X = np.array(X_list, dtype=np.float32)   # (N, W, F)
    y = np.array(y_list,  dtype=np.float32)  # (N,)

    return X, y


def train_test_split_by_cycle(
    df: pd.DataFrame,
    test_ratio: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare le DataFrame en train/test en réservant les derniers
    test_ratio% des cycles pour le test.

    Cette approche évite la fuite temporelle : le modèle est évalué
    sur des cycles qu'il n'a jamais vus pendant l'entraînement.

    Parameters
    ----------
    df         : DataFrame complet
    test_ratio : fraction [0,1] des cycles allouée au test

    Returns
    -------
    df_train, df_test
    """
    cycles  = sorted(df["cycle_number"].unique())
    n_train = int(len(cycles) * (1 - test_ratio))
    train_cycles = set(cycles[:n_train])
    test_cycles  = set(cycles[n_train:])

    df_train = df[df["cycle_number"].isin(train_cycles)].copy()
    df_test  = df[df["cycle_number"].isin(test_cycles)].copy()

    print(f"      Train : {len(train_cycles)} cycles · Test : {len(test_cycles)} cycles")
    return df_train, df_test