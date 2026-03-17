# =============================================================================
# src/train.py
# =============================================================================
# Responsabilité : Gérer la boucle d'entraînement du modèle LSTM.
#
# Ce module contient :
#   - La création des DataLoaders (mini-batches)
#   - La boucle epoch par epoch (train + validation)
#   - Le calcul de la loss à chaque étape
#   - L'early stopping pour arrêter avant l'overfitting
#   - La sauvegarde des courbes d'apprentissage
#
# CHOIX DE LA LOSS : MSE (Mean Squared Error)
# --------------------------------------------
# Pour un problème de régression, on minimise l'erreur quadratique.
# Le MSE pénalise plus fortement les grandes erreurs (utile ici :
# une erreur de 10% sur le SoH est bien plus grave qu'une erreur de 1%).
#
# OPTIMISEUR : Adam
# -----------------
# Adam est un optimiseur adaptatif qui ajuste le taux d'apprentissage
# pour chaque paramètre. Il converge plus vite que le SGD classique
# sur des données temporelles avec LSTM.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List


def creer_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """
    Convertit les arrays numpy en DataLoaders PyTorch.

    Un DataLoader :
    - Découpe les données en mini-batches de taille batch_size
    - Mélange les données train à chaque epoch (shuffle=True)
    - Gère la parallélisation (num_workers)

    Paramètres
    ----------
    X_train, y_train : arrays numpy train
    X_test, y_test   : arrays numpy test
    batch_size       : nombre d'exemples par batch

    Retourne
    --------
    (loader_train, loader_test) : deux DataLoaders PyTorch
    """

    # Conversion numpy → PyTorch tensors (float32 requis par le LSTM)
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)  # (N,) → (N, 1) pour correspondre à la sortie du modèle
    X_test_t  = torch.FloatTensor(X_test)
    y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

    # TensorDataset associe X et y pour les itérer ensemble
    dataset_train = TensorDataset(X_train_t, y_train_t)
    dataset_test  = TensorDataset(X_test_t, y_test_t)

    # shuffle=True sur le train : mélanger évite que le modèle
    # apprenne l'ordre des exemples plutôt que les patterns réels
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

    print(f"[train] DataLoaders crees | batch={batch_size} | "
          f"train: {len(loader_train)} batchs | test: {len(loader_test)} batchs")

    return loader_train, loader_test


def entrainer(
    modele: nn.Module,
    loader_train: DataLoader,
    loader_test: DataLoader,
    n_epochs: int = 100,
    lr: float = 0.001,
    patience: int = 15
) -> dict:
    """
    Boucle d'entraînement principale avec early stopping.

    EARLY STOPPING :
    ----------------
    Si la loss de validation ne s'améliore pas pendant `patience` epochs,
    on arrête l'entraînement. Cela évite l'overfitting :
    le modèle mémoriserait le train au détriment de la généralisation.
    On restaure les meilleurs poids trouvés avant l'arrêt.

    Paramètres
    ----------
    modele      : modèle LSTM PyTorch
    loader_train : DataLoader d'entraînement
    loader_test  : DataLoader de validation
    n_epochs    : nombre maximum d'epochs
    lr          : taux d'apprentissage initial
    patience    : tolérance early stopping (epochs sans amélioration)

    Retourne
    --------
    dict avec :
        "train_losses"  : historique des losses train par epoch
        "test_losses"   : historique des losses test par epoch
        "best_epoch"    : epoch où la meilleure loss test a été atteinte
    """

    # Fonction de perte : MSE pour la régression
    critere = nn.MSELoss()

    # Optimiseur Adam avec weight_decay (régularisation L2 légère)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=lr, weight_decay=1e-5)

    # Scheduler : réduit lr de moitié si la loss ne baisse plus pendant 10 epochs
    # Aide à sortir des plateaux d'apprentissage
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiseur, mode="min", patience=10, factor=0.5
    )

    # Historique pour visualisation
    train_losses: List[float] = []
    test_losses: List[float]  = []

    # Variables pour l'early stopping
    meilleure_loss_test = float("inf")
    compteur_patience   = 0
    meilleurs_poids     = None  # Sauvegarde en mémoire des meilleurs poids
    meilleure_epoch     = 0

    print(f"\n[train] Demarrage de l'entrainement")
    print(f"   Epochs max : {n_epochs} | LR : {lr} | Patience : {patience}")
    print(f"   {'Epoch':>6} | {'Loss Train':>12} | {'Loss Test':>12} | {'Statut'}")
    print("   " + "-" * 50)

    for epoch in range(1, n_epochs + 1):

        # ---- Phase TRAIN ----
        modele.train()  # Active le dropout
        loss_train_total = 0.0

        for X_batch, y_batch in loader_train:
            optimiseur.zero_grad()           # Remet les gradients à zéro
            prediction = modele(X_batch)     # Passe avant
            loss = critere(prediction, y_batch)  # Calcul de l'erreur
            loss.backward()                  # Rétropropagation des gradients
            nn.utils.clip_grad_norm_(modele.parameters(), max_norm=1.0)  # Clip gradients
            optimiseur.step()                # Mise à jour des poids
            loss_train_total += loss.item()

        loss_train_moy = loss_train_total / len(loader_train)

        # ---- Phase VALIDATION ----
        modele.eval()   # Désactive le dropout
        loss_test_total = 0.0

        with torch.no_grad():  # Pas de gradients en validation = plus rapide
            for X_batch, y_batch in loader_test:
                prediction = modele(X_batch)
                loss = critere(prediction, y_batch)
                loss_test_total += loss.item()

        loss_test_moy = loss_test_total / len(loader_test)

        # Mise à jour du scheduler
        scheduler.step(loss_test_moy)

        # Sauvegarde des historiques
        train_losses.append(loss_train_moy)
        test_losses.append(loss_test_moy)

        # ---- Early Stopping ----
        if loss_test_moy < meilleure_loss_test:
            meilleure_loss_test = loss_test_moy
            meilleurs_poids     = {k: v.clone() for k, v in modele.state_dict().items()}
            meilleure_epoch     = epoch
            compteur_patience   = 0
            statut = "* meilleur"
        else:
            compteur_patience += 1
            statut = f"patience {compteur_patience}/{patience}"

        # Affichage tous les 10 epochs ou quand c'est le meilleur
        if epoch % 10 == 0 or "meilleur" in statut:
            print(f"   {epoch:>6} | {loss_train_moy:>12.6f} | "
                  f"{loss_test_moy:>12.6f} | {statut}")

        # Arrêt si patience dépassée
        if compteur_patience >= patience:
            print(f"\n[train] Early stopping a l'epoch {epoch} "
                  f"(meilleur: epoch {meilleure_epoch})")
            break

    # Restauration des meilleurs poids
    if meilleurs_poids is not None:
        modele.load_state_dict(meilleurs_poids)
        print(f"[train] Poids restaures depuis l'epoch {meilleure_epoch}")

    print(f"[train] Entrainement termine | Meilleure loss test : {meilleure_loss_test:.6f}\n")

    return {
        "train_losses": train_losses,
        "test_losses":  test_losses,
        "best_epoch":   meilleure_epoch
    }