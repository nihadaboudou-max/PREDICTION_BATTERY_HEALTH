"""
src/model.py
Architecture LSTM pour la prédiction du SoH (State of Health).
"""

import torch
import torch.nn as nn


class LSTMSoH(nn.Module):
    """
    LSTM empilé pour la régression du SoH.

    Architecture :
        LSTM(input_size, hidden_size, num_layers, dropout)
          → couche de dropout finale
          → Linear(hidden_size, 1)

    Entrée  : (batch, seq_len, input_size)  — batch de fenêtres
    Sortie  : (batch, 1)                    — SoH prédit [0–1], à rescaler

    Pourquoi un LSTM ?
    ------------------
    Un MLP traiterait chaque mesure indépendamment. Le LSTM lit la
    séquence bin par bin et accumule de l'information sur la dynamique
    de décharge (chute de tension, évolution du courant), qui est
    révélatrice de l'état de dégradation.
    """

    def __init__(
        self,
        input_size : int   = 4,
        hidden_size: int   = 64,
        num_layers : int   = 2,
        dropout    : float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        """
        # Initialisation des états cachés à zéro
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)

        # Passage dans le LSTM
        out, _ = self.lstm(x, (h0, c0))   # out : (batch, seq_len, hidden_size)

        # On utilise uniquement la sortie du dernier pas de temps
        last_hidden = out[:, -1, :]        # (batch, hidden_size)

        last_hidden = self.dropout(last_hidden)
        pred        = self.fc(last_hidden) # (batch, 1)

        return pred.squeeze(-1)            # (batch,)


def count_parameters(model: nn.Module) -> int:
    """Retourne le nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)