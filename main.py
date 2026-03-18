"""
main.py — Point d'entrée du pipeline SoH LSTM
Exécutez : python main.py
"""

import os
import torch
import numpy as np

# ─── HYPERPARAMÈTRES (à modifier ici) ──────────────────────────
DATA_PATH    = "data/battery_data.csv"
MODEL_DIR    = "models"
RESULTS_DIR  = "results/plots"

WINDOW_SIZE  = 3       # Taille de la fenêtre glissante (bins)
TEST_RATIO   = 0.20    # Fraction des cycles réservés au test
HIDDEN_SIZE  = 64      # Nombre d'unités LSTM
NUM_LAYERS   = 2       # Couches LSTM empilées
LEARNING_RATE= 1e-3    # Taux d'apprentissage Adam
MAX_EPOCHS   = 100     # Epochs maximum
PATIENCE     = 15      # Early stopping patience
BATCH_SIZE   = 64      # Taille des mini-batchs

FEATURES = ["Voltage_measured", "Current_measured",
            "Temperature_measured", "SoC"]
TARGET   = "SoH"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ─── IMPORTS MODULES ───────────────────────────────────────────
from data_loader    import load_and_validate
from preprocessing  import normalize_features, create_windows, train_test_split_by_cycle
from model          import LSTMSoH
from train          import train_model
from evaluate       import evaluate_model, plot_results


def main():
    print("=" * 60)
    print("  Battery SoH Predictor — Pipeline LSTM")
    print(f"  Device : {DEVICE}")
    print("=" * 60)

    # ── 1. Chargement et validation ──────────────────────────────
    print("\n[1/6] Chargement du CSV...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_and_validate(DATA_PATH, FEATURES, TARGET)
    print(f"      {len(df)} lignes · {df['battery_id'].nunique()} batteries · {df['cycle_number'].nunique()} cycles")

    # ── 2. Normalisation ─────────────────────────────────────────
    print("\n[2/6] Normalisation StandardScaler (fit sur train uniquement)...")
    cycles      = sorted(df["cycle_number"].unique())
    n_train     = int(len(cycles) * (1 - TEST_RATIO))
    train_cycles= cycles[:n_train]
    test_cycles = cycles[n_train:]

    df_train = df[df["cycle_number"].isin(train_cycles)].copy()
    df_test  = df[df["cycle_number"].isin(test_cycles)].copy()

    df_train_n, df_test_n, scaler = normalize_features(
        df_train, df_test, FEATURES,
        scaler_path=os.path.join(MODEL_DIR, "scaler.pkl")
    )
    print(f"      Train : {len(train_cycles)} cycles · Test : {len(test_cycles)} cycles")

    # ── 3. Fenêtres glissantes ───────────────────────────────────
    print(f"\n[3/6] Fenêtres glissantes W={WINDOW_SIZE}...")
    X_train, y_train = create_windows(df_train_n, FEATURES, TARGET, WINDOW_SIZE)
    X_test,  y_test  = create_windows(df_test_n,  FEATURES, TARGET, WINDOW_SIZE)
    print(f"      Fenêtres train : {X_train.shape[0]} · test : {X_test.shape[0]}")

    # ── 4. Split (déjà fait par cycle ci-dessus) ─────────────────
    print(f"\n[4/6] Split train/test par cycle :")
    print(f"      Train cycles : {train_cycles[0]}→{train_cycles[-1]}")
    print(f"      Test  cycles : {test_cycles[0]}→{test_cycles[-1]}")

    # ── 5. Entraînement ──────────────────────────────────────────
    print(f"\n[5/6] Entraînement LSTM ({HIDDEN_SIZE} unités, {NUM_LAYERS} couches)...")
    model = LSTMSoH(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Paramètres entraînables : {n_params:,}")

    model = train_model(
        model, X_train, y_train, X_test, y_test,
        lr=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        model_path=os.path.join(MODEL_DIR, "lstm_soh.pth"),
        results_dir=RESULTS_DIR
    )

    # ── 6. Évaluation ────────────────────────────────────────────
    print("\n[6/6] Évaluation sur le jeu test...")
    metrics = evaluate_model(
        model, X_test, y_test,
        device=DEVICE,
        results_dir=RESULTS_DIR,
        scaler=scaler,
        target=TARGET
    )

    print("\n" + "=" * 60)
    print("  RÉSULTATS FINAUX")
    print(f"  MAE  : {metrics['mae']:.4f}%")
    print(f"  RMSE : {metrics['rmse']:.4f}%")
    print(f"  R²   : {metrics['r2']:.4f}")
    print("=" * 60)
    print(f"\n  Modèle   → {MODEL_DIR}/lstm_soh.pth")
    print(f"  Scaler   → {MODEL_DIR}/scaler.pkl")
    print(f"  Graphiques → {RESULTS_DIR}/")
    print("\nPipeline terminé.\n")


if __name__ == "__main__":
    main()