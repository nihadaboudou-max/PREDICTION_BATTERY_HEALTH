"""
src/data_loader.py
Chargement et validation du fichier CSV battery_data.csv.
"""

import pandas as pd
import numpy as np


def load_and_validate(
    path: str,
    features: list[str],
    target: str,
    warn_soh_bounds: bool = True
) -> pd.DataFrame:
    """
    Charge le CSV, vérifie les colonnes requises et signale les anomalies.

    Parameters
    ----------
    path    : chemin vers le fichier CSV
    features: liste des colonnes features attendues
    target  : colonne cible (SoH) — optionnelle dans le CSV
    warn_soh_bounds : si True, signale les SoH hors [0, 100]

    Returns
    -------
    df : DataFrame nettoyé (lignes incomplètes supprimées)

    Raises
    ------
    FileNotFoundError  : si path n'existe pas
    ValueError         : si des colonnes obligatoires sont manquantes
    """
    # ── Chargement ────────────────────────────────────────────────
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Vérifiez que battery_data.csv est bien dans le dossier data/"
        )

    # ── Colonnes obligatoires ─────────────────────────────────────
    required = features + ["cycle_number", "battery_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {missing}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )

    # ── Colonne cible (optionnelle) ───────────────────────────────
    has_target = target in df.columns
    if not has_target:
        print(f"  [WARN] Colonne '{target}' absente — mode prédiction pure activé.")

    # ── Suppression des lignes incomplètes ────────────────────────
    cols_to_check = required + ([target] if has_target else [])
    n_before = len(df)
    df = df.dropna(subset=cols_to_check)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  [WARN] {n_dropped} lignes supprimées (valeurs manquantes).")

    # ── Conversion des types ──────────────────────────────────────
    for col in features + (["cycle_number"] if "cycle_number" in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=features)

    # ── Vérification SoH hors plage ───────────────────────────────
    if has_target and warn_soh_bounds:
        out_of_range = df[(df[target] < 0) | (df[target] > 100)]
        if len(out_of_range) > 0:
            print(
                f"  [WARN] {len(out_of_range)} valeurs SoH hors [0, 100] détectées.\n"
                f"         (min={df[target].min():.2f}%, max={df[target].max():.2f}%)\n"
                "         Ces valeurs sont conservées — filtrez-les si nécessaire."
            )

    # ── Tri par batterie et cycle ─────────────────────────────────
    df = df.sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)

    return df


def verifier_qualite(df: pd.DataFrame, target: str = "SoH") -> dict:
    """
    Retourne un résumé de qualité du dataset.

    Returns
    -------
    dict avec clés : n_rows, n_batteries, n_cycles, has_target,
                     soh_out_of_range, missing_pct
    """
    has_target = target in df.columns
    return {
        "n_rows"          : len(df),
        "n_batteries"     : df["battery_id"].nunique(),
        "n_cycles"        : df["cycle_number"].nunique(),
        "has_target"      : has_target,
        "soh_out_of_range": int(((df[target] < 0) | (df[target] > 100)).sum()) if has_target else 0,
        "missing_pct"     : float(df.isnull().mean().max() * 100),
    }