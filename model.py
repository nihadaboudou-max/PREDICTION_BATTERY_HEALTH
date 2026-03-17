# =============================================================================
# src/model.py
# =============================================================================
# Responsabilité : Définir l'architecture du réseau de neurones LSTM.
#
# ARCHITECTURE CHOISIE
# --------------------
# Input  → LSTM (64 unités, 2 couches) → Dropout (0.2) → Dense → SoH prédit
#
# POURQUOI LSTM ET PAS UN MLP CLASSIQUE ?
# -----------------------------------------
# Un MLP (réseau dense) traiterait chaque bin de manière indépendante.
# Il ignorerait que la mesure au temps t dépend des mesures t-1, t-2, etc.
# Le LSTM a une "mémoire" interne : il lit la séquence bin après bin
# et accumule de l'information sur la dynamique temporelle.
# C'est exactement ce qu'on veut : capturer comment la tension CHUTE
# progressivement pendant une décharge, ce qui est révélateur du SoH.
#
# HYPERPARAMÈTRES IMPORTANTS
# ---------------------------
# hidden_size : nombre d'unités LSTM (capacité mémorielle)
# num_layers  : profondeur → capture des patterns plus abstraits
# dropout     : régularisation pour éviter l'overfitting
# =============================================================================

import torch
import torch.nn as nn
import os


class LSTMBatterie(nn.Module):
    """
    Réseau LSTM pour la prédiction du SoH d'une batterie.

    Forme des entrées attendues :
        (batch_size, sequence_length, input_size)
        ex: (32, 3, 5)  → 32 exemples, fenêtres de 3 bins, 5 features

    Forme de la sortie :
        (batch_size, 1)  → une valeur de SoH par exemple
    """

    def __init__(
        self,
        input_size: int = 5,      # Nombre de features (Voltage, Current, Temp, SoC, cycle)
        hidden_size: int = 64,    # Taille de l'état caché du LSTM
        num_layers: int = 2,      # Nombre de couches LSTM empilées
        dropout: float = 0.2,     # Taux de dropout entre les couches LSTM
        output_size: int = 1      # On prédit une seule valeur : le SoH
    ):
        super(LSTMBatterie, self).__init__()

        # Sauvegarde des hyperparamètres pour pouvoir reconstruire le modèle
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # --- Couche LSTM ---
        # batch_first=True : les tenseurs ont la forme (batch, seq, features)
        # au lieu de (seq, batch, features). Plus intuitif.
        # dropout s'applique ENTRE les couches (pas après la dernière)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # dropout inutile si 1 seule couche
        )

        # Dropout supplémentaire après le LSTM (régularisation plus forte)
        self.dropout = nn.Dropout(p=dropout)

        # --- Couche de sortie ---
        # On prend uniquement la sortie du DERNIER pas de temps du LSTM
        # (représentation compacte de toute la séquence) et on la projette en 1 valeur
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du réseau.

        Paramètres
        ----------
        x : Tensor de forme (batch_size, seq_len, input_size)

        Retourne
        --------
        Tensor de forme (batch_size, 1)  — SoH prédit
        """

        # Initialisation des états cachés à zéro au début de chaque batch
        # h0 : état caché (mémoire court terme)
        # c0 : état de cellule (mémoire long terme)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Passage dans le LSTM
        # lstm_out : (batch, seq_len, hidden_size) — sortie à chaque pas de temps
        # (hn, cn) : états finaux (non utilisés ici)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # On utilise uniquement la sortie du DERNIER pas de temps
        # C'est la représentation résumant toute la séquence
        dernier_pas = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Régularisation
        dernier_pas = self.dropout(dernier_pas)

        # Projection vers la prédiction du SoH
        sortie = self.fc(dernier_pas)  # (batch_size, 1)

        return sortie


def creer_modele(
    input_size: int = 5,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2
) -> LSTMBatterie:
    """
    Instancie et retourne un nouveau modèle LSTM.

    Paramètres
    ----------
    input_size  : nombre de features d'entrée
    hidden_size : nombre d'unités cachées LSTM
    num_layers  : profondeur du LSTM
    dropout     : taux de dropout

    Retourne
    --------
    LSTMBatterie : modèle PyTorch initialisé
    """
    modele = LSTMBatterie(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    nb_params = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    print(f"[model] Architecture : LSTM({hidden_size} unites, {num_layers} couches)")
    print(f"[model] Nombre de parametres entrainables : {nb_params:,}")
    return modele


def sauvegarder_modele(modele: LSTMBatterie, chemin: str = "models/lstm_soh.pth") -> None:
    """
    Sauvegarde les poids du modèle sur disque.

    On sauvegarde le state_dict (dictionnaire des poids)
    plutôt que l'objet complet, ce qui est plus portable.

    Paramètres
    ----------
    modele : LSTMBatterie entraîné
    chemin : chemin de sauvegarde
    """
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    torch.save(modele.state_dict(), chemin)
    print(f"[model] Poids sauvegardes → {chemin}")


def charger_modele(
    chemin: str = "models/lstm_soh.pth",
    input_size: int = 5,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2
) -> LSTMBatterie:
    """
    Charge un modèle pré-entraîné depuis un fichier .pth.
    Utile pour faire des prédictions sans ré-entraîner.

    Paramètres
    ----------
    chemin      : chemin vers le fichier .pth
    (autres)    : mêmes hyperparamètres que lors de la création

    Retourne
    --------
    LSTMBatterie : modèle avec poids chargés, en mode évaluation
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Modele introuvable : {chemin}\n"
                                f"Lancez d'abord main.py pour entraîner le modele.")

    modele = LSTMBatterie(input_size, hidden_size, num_layers, dropout)
    modele.load_state_dict(torch.load(chemin, map_location="cpu"))
    modele.eval()  # Mode évaluation : désactive le dropout
    print(f"[model] Modele charge depuis {chemin}")
    return modele