from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

# Registry per attivazioni e ottimizzatori
ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "selu": nn.SELU,
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

class PriorEstimator:
    """
    Stima μ e σ in modo efficiente senza caricare tutto il dataset in memoria.
    """
    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit(self, train_loader: torch.utils.data.DataLoader) -> None:
        """Calcola statistiche con algoritmo Welford o passaggi cumulativi."""
        sum_y = 0.0
        sum_sq_y = 0.0
        n = 0
        
        # Primo passaggio: accumulo somme per evitare torch.cat
        for _, y in train_loader:
            sum_y += y.sum(dim=0)
            sum_sq_y += (y**2).sum(dim=0)
            n += y.size(0)
            
        self.mean = sum_y / n
        # Varianza = E[X^2] - (E[X])^2
        var = (sum_sq_y / n) - (self.mean ** 2)
        self.std = var.clamp(min=1e-6).sqrt()
        
        print(f"[PriorEstimator] μ stimata su {n} campioni.")

    @property
    def fitted(self) -> bool:
        return self.mean is not None

class MrELoss(nn.Module):
    """
    Loss ottimizzata per MrE.
    """
    def __init__(self, lambda_entropy: float = 0.1, lambda_moment: float = 0.1) -> None:
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_moment = lambda_moment

    def _kl_gaussian(self, pred: torch.Tensor, prior_mean: torch.Tensor, prior_std: torch.Tensor) -> torch.Tensor:
        # Pre-calcoliamo la varianza e log_var per evitare operazioni ridondanti
        var = prior_std.pow(2).clamp(min=1e-8)
        # KL Divergence analitica
        kl = 0.5 * (torch.log(var) + (pred - prior_mean).pow(2) / var + 1.0 / var - 1.0)
        return kl.mean()

    def forward(self, pred, target, prior_mean=None, prior_std=None, moment_target=None, data_weight=1.0):
        info = {}
        
        # Termine Likelihood (Data)
        if data_weight > 0:
            data_loss = F.mse_loss(pred, target)
            total = data_loss * data_weight
            info["data"] = data_loss.item()
        else:
            total = torch.tensor(0.0, device=pred.device)
            info["data"] = 0.0

        # Termine Entropico (KL con Prior)
        if prior_mean is not None and self.lambda_entropy > 0:
            ent = self._kl_gaussian(pred, prior_mean, prior_std)
            total = total + self.lambda_entropy * ent
            info["entropy"] = ent.item()

        # Vincolo sui Momenti (Batch Mean vs Target F)
        if moment_target is not None and self.lambda_moment > 0:
            pred_mean = pred.mean(dim=0)
            mom = F.mse_loss(pred_mean, moment_target)
            total = total + self.lambda_moment * mom
            info["moment"] = mom.item()

        info["total"] = total.item()
        return total, info

class Pred(nn.Module):
    def __init__(self, num_features, window_size, dimension=[64, 32, 16], dilations=[1, 2, 4], kernel_size=3, activation="leaky_relu"):
        super().__init__()
        act_fn = ACTIVATIONS.get(activation, nn.LeakyReLU)
        
        layers = []
        in_ch = num_features
        for i, out_ch in enumerate(dimension):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=dilations[i], dilation=dilations[i]))
            layers.append(act_fn())
            in_ch = out_ch
            
        self.conv_block = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dimension[-1] * window_size, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # (B, T, F) -> (B, F, T)
        x = self.conv_block(x)
        x = self.flatten(x)
        return self.fc(x)

    def _run_phase(self, train_loader, training_cfg, criterion, prior_mean, prior_std, moment_target, data_weight=1.0, label="", epochs=None):
        n_epochs = epochs or training_cfg.epochs
        optimizer = OPTIMIZERS.get(training_cfg.optimizer, torch.optim.Adam)(
            self.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.weight_decay
        )
        
        self.train()
        device = next(self.parameters()).device
        
        # Spostiamo i target sui device corretti una volta sola
        if prior_mean is not None: prior_mean = prior_mean.to(device)
        if prior_std is not None: prior_std = prior_std.to(device)
        if moment_target is not None: moment_target = moment_target.to(device)

        for epoch in range(n_epochs):
            total_l = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True) # set_to_none è più veloce
                pred = self(bx)
                loss, _ = criterion(pred, by, prior_mean, prior_std, moment_target, data_weight)
                loss.backward()
                optimizer.step()
                total_l += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"[{label}] Epoch {epoch+1}/{n_epochs} | Loss: {total_l/len(train_loader):.6f}")

    def fit(self, train_loader, training_cfg, moment_target=None):
        mre_cfg = getattr(training_cfg, "mre", None)
        if not mre_cfg or not mre_cfg.enabled:
            # Fallback MSE veloce
            self._run_phase(train_loader, training_cfg, MrELoss(0, 0), None, None, None, 1.0, "MSE")
            return

        criterion = MrELoss(mre_cfg.lambda_entropy, mre_cfg.lambda_moment)
        prior = PriorEstimator()
        prior.fit(train_loader)

        mode = getattr(mre_cfg, "update_mode", "simultaneous")
        if mode == "sequential":
            # Fase 1: Solo momenti
            ep1 = getattr(mre_cfg, "epochs_phase1", training_cfg.epochs // 3)
            self._run_phase(train_loader, training_cfg, criterion, prior.mean, prior.std, moment_target, 0.0, "SEQ-P1", ep1)
            # Update prior (necessario per paper, ma lento)
            prior.fit(train_loader) 
            # Fase 2: Solo dati
            self._run_phase(train_loader, training_cfg, criterion, prior.mean, prior.std, None, 1.0, "SEQ-P2", training_cfg.epochs - ep1)
        else:
            self._run_phase(train_loader, training_cfg, criterion, prior.mean, prior.std, moment_target, 1.0, "SIMULT")