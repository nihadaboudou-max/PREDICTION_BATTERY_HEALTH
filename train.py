"""
src/train.py
Boucle d'entraînement avec early stopping et sauvegarde du meilleur modèle.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    lr:           float = 1e-3,
    max_epochs:   int   = 100,
    patience:     int   = 15,
    batch_size:   int   = 64,
    device:       str   = "cpu",
    model_path:   str   = "models/lstm_soh.pth",
    results_dir:  str   = "results/plots",
) -> nn.Module:
    """
    Entraîne le modèle LSTM avec :
      - Optimiseur Adam
      - Loss MSE
      - Early stopping sur la val loss
      - Sauvegarde des meilleurs poids

    Parameters
    ----------
    model       : instance LSTMSoH
    X_train     : (N_train, W, F) float32
    y_train     : (N_train,)      float32
    X_val       : (N_val, W, F)   float32
    y_val       : (N_val,)        float32
    lr          : learning rate Adam
    max_epochs  : nombre maximum d'epochs
    patience    : patience early stopping
    batch_size  : taille des mini-batchs
    device      : 'cpu' ou 'cuda'
    model_path  : chemin de sauvegarde du meilleur modèle
    results_dir : dossier de sortie pour les graphiques

    Returns
    -------
    model chargé avec les meilleurs poids
    """
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────
    def to_tensor(arr):
        return torch.from_numpy(arr).float().to(device)

    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    val_ds   = TensorDataset(to_tensor(X_val),   to_tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ── Optimiseur + Loss ─────────────────────────────────────────
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.MSELoss()
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, verbose=False
    )

    # ── Boucle d'entraînement ─────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        # — Train —
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * len(X_batch)
        epoch_train_loss /= len(train_ds)

        # — Validation —
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                epoch_val_loss += loss.item() * len(X_batch)
        epoch_val_loss /= len(val_ds)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        # ─ Affichage ─
        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/{max_epochs} | "
                  f"train={epoch_train_loss:.4f} | val={epoch_val_loss:.4f}"
                  + (" ← meilleur" if epoch_val_loss < best_val_loss else ""))

        # ─ Early stopping ─
        if epoch_val_loss < best_val_loss:
            best_val_loss  = epoch_val_loss
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"      Early stopping déclenché à l'epoch {epoch} "
                      f"(patience={patience}).")
                break

    # ── Rechargement des meilleurs poids ─────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"      Meilleurs poids restaurés → {model_path}")

    # ── Courbe d'apprentissage ────────────────────────────────────
    _plot_learning_curves(train_losses, val_losses,
                          os.path.join(results_dir, "learning_curves.png"))

    return model


def _plot_learning_curves(train_losses, val_losses, save_path):
    """Trace et sauvegarde les courbes train/val loss."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d0f14")
    ax.set_facecolor("#141720")
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="#00d4ff", linewidth=1.5, label="Train Loss")
    ax.plot(epochs, val_losses,   color="#00e676", linewidth=1.5, linestyle="--", label="Val Loss")
    ax.set_xlabel("Epoch", color="#6b7280")
    ax.set_ylabel("MSE Loss", color="#6b7280")
    ax.set_title("Courbes d'apprentissage", color="#e8ecf4")
    ax.legend(facecolor="#1c2030", edgecolor="#252a38", labelcolor="#e8ecf4")
    ax.tick_params(colors="#6b7280")
    for spine in ax.spines.values():
        spine.set_edgecolor("#252a38")
    ax.grid(color="#252a38", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      Courbes d'apprentissage → {save_path}")