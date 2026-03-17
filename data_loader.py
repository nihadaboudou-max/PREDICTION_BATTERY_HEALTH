# =============================================================================
# src/data_loader.py
# =============================================================================
# Responsabilité : Charger le fichier CSV brut et vérifier sa cohérence.
#
# Ce module fait UNIQUEMENT la lecture et la validation basique.
# Aucune transformation ici — c'est le rôle de preprocessing.py.
#
# Principe de séparation des responsabilités (SRP) :
#   chaque fichier a UN rôle précis, ce qui facilite la maintenance.
# =============================================================================

import pandas as pd
import os


# Colonnes attendues dans le fichier CSV
# Si une colonne est absente, le chargement échoue avec un message clair.
COLONNES_REQUISES = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "SoC",
    "cycle_number",
    "battery_id",
    "SoH"
]


def charger_donnees(chemin_csv: str) -> pd.DataFrame:
    """
    Charge le fichier CSV de données de batteries.

    Paramètres
    ----------
    chemin_csv : str
        Chemin vers le fichier CSV (ex: "data/battery_data.csv")

    Retourne
    --------
    pd.DataFrame
        DataFrame avec toutes les colonnes brutes

    Erreurs
    -------
    FileNotFoundError : si le fichier n'existe pas
    ValueError        : si des colonnes sont manquantes
    """

    # Vérification de l'existence du fichier avant toute tentative de lecture
    if not os.path.exists(chemin_csv):
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin_csv}\n"
            f"Vérifiez que le chemin est correct depuis le répertoire courant."
        )

    print(f"[data_loader] Lecture du fichier : {chemin_csv}")
    df = pd.read_csv(chemin_csv)

    # Validation : toutes les colonnes attendues doivent être présentes
    colonnes_manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {colonnes_manquantes}\n"
            f"Colonnes trouvées : {list(df.columns)}"
        )

    print(f"[data_loader] {len(df)} lignes chargées | "
          f"{df['battery_id'].nunique()} batterie(s) | "
          f"{df['cycle_number'].nunique()} cycle(s)")

    return df


def afficher_apercu(df: pd.DataFrame) -> None:
    """
    Affiche un résumé statistique du dataset pour une première inspection.
    Permet de détecter des anomalies évidentes (valeurs aberrantes, NaN, etc.)

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame brut chargé par charger_donnees()
    """

    print("\n" + "=" * 60)
    print("  APERCU DU DATASET")
    print("=" * 60)

    # Dimensions
    print(f"\n  Dimensions          : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    # Batteries présentes
    batteries = df["battery_id"].unique()
    print(f"  Batteries           : {list(batteries)}")

    # Plage de cycles
    print(f"  Cycles              : {df['cycle_number'].min()} → {df['cycle_number'].max()}")

    # Plage de SoH (variable cible)
    print(f"  SoH (cible)         : min={df['SoH'].min():.2f}%  "
          f"max={df['SoH'].max():.2f}%  "
          f"moy={df['SoH'].mean():.2f}%")

    # Valeurs manquantes — critique pour la modélisation
    nb_nan = df.isnull().sum().sum()
    if nb_nan == 0:
        print(f"  Valeurs manquantes  : aucune (données completes)")
    else:
        print(f"  Valeurs manquantes  : {nb_nan} detectees !")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    # Statistiques descriptives des variables numériques
    print("\n  Statistiques descriptives :")
    print(df[["Voltage_measured", "Current_measured",
              "Temperature_measured", "SoC", "SoH"]].describe().round(3))

    print("=" * 60 + "\n")


def verifier_qualite(df: pd.DataFrame) -> dict:
    """
    Effectue des contrôles qualité sur les données.
    Retourne un dictionnaire avec les problèmes détectés.

    Paramètres
    ----------
    df : pd.DataFrame

    Retourne
    --------
    dict : résumé des problèmes détectés
        {
          "nan_count": int,
          "soh_hors_plage": int,   # SoH < 0 ou SoH > 100
          "soc_hors_plage": int,   # SoC < 0 ou SoC > 100
          "doublons": int
        }
    """

    problemes = {}

    # 1. Valeurs manquantes
    problemes["nan_count"] = int(df.isnull().sum().sum())

    # 2. SoH doit être entre 0 et 100 (c'est un pourcentage)
    soh_invalide = df[(df["SoH"] < 0) | (df["SoH"] > 100)]
    problemes["soh_hors_plage"] = len(soh_invalide)

    # 3. SoC doit être entre 0 et 100
    soc_invalide = df[(df["SoC"] < 0) | (df["SoC"] > 100)]
    problemes["soc_hors_plage"] = len(soc_invalide)

    # 4. Lignes dupliquées
    problemes["doublons"] = int(df.duplicated().sum())

    # Affichage du bilan
    print("[data_loader] Bilan qualite :")
    for cle, val in problemes.items():
        statut = "OK" if val == 0 else f"ATTENTION : {val} cas"
        print(f"   {cle:<25} → {statut}")

    return problemes