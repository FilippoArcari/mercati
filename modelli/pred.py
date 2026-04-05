from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ACTIVATIONS = {
    "relu":       nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu":       nn.GELU,
    "elu":        nn.ELU,
    "selu":       nn.SELU,
}

OPTIMIZERS = {
    "adam":    torch.optim.Adam,
    "adamw":   torch.optim.AdamW,
    "sgd":     torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


# ---------------------------------------------------------------------------
# Prior Estimator
# ---------------------------------------------------------------------------
class PriorEstimator:
    """
    Stima mu e sigma senza caricare tutto il dataset in memoria.
    Compatibile con target sia (B, F) che (B, steps, F).
    """
    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None
        self.std:  Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit(self, train_loader: torch.utils.data.DataLoader) -> None:
        sum_y    = 0.0
        sum_sq_y = 0.0
        n        = 0
        for _, y in train_loader:
            flat      = y.reshape(-1, y.shape[-1])
            sum_y    += flat.sum(dim=0)
            sum_sq_y += (flat ** 2).sum(dim=0)
            n        += flat.size(0)
        self.mean = sum_y / n
        var       = (sum_sq_y / n) - (self.mean ** 2)
        self.std  = var.clamp(min=1e-6).sqrt()
        print(f"[PriorEstimator] mu stimata su {n} campioni.")

    @property
    def fitted(self) -> bool:
        return self.mean is not None


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class MrELoss(nn.Module):
    def __init__(
        self,
        lambda_entropy: float = 0.1,
        lambda_moment:  float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_moment  = lambda_moment

    def _kl_gaussian(
        self,
        pred:       torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std:  torch.Tensor,
    ) -> torch.Tensor:
        var = prior_std.pow(2).clamp(min=0.1)
        return (0.5 * ((pred - prior_mean).pow(2) / var)).mean()

    @staticmethod
    def _align(pred: torch.Tensor, target: torch.Tensor):
        """
        Allinea pred e target per il calcolo della loss.

        Casi:
          (A) pred (B, F)        + target (B, F)        -> nessun cambio
          (B) pred (B, steps, F) + target (B, steps, F) -> nessun cambio
          (C) pred (B, steps, F) + target (B, F)        -> usa solo pred[:, 0, :]

        Il caso (C) si verifica quando make_windows restituisce target single-step
        ma prediction_steps > 1.  La loss viene calcolata solo sul primo step
        (il prossimo bar), che è l'unico target disponibile.
        Per supervisionare tutti gli step usa make_windows_multistep in utils.py.
        """
        if pred.dim() == 3 and target.dim() == 2:
            pred = pred[:, 0, :]   # (B, F) — supervisione solo su step 0
        return pred, target

    def forward(
        self,
        pred:          torch.Tensor,
        target:        torch.Tensor,
        prior_mean:    Optional[torch.Tensor] = None,
        prior_std:     Optional[torch.Tensor] = None,
        moment_target: Optional[torch.Tensor] = None,
        data_weight:   float = 1.0,
    ):
        info = {}

        # Allineamento shape
        pred_a, target_a = self._align(pred, target)

        # Data loss (MSE)
        if data_weight > 0:
            data_loss = F.mse_loss(pred_a, target_a)
            total     = data_loss * data_weight
            info["data"] = data_loss.item()
        else:
            total        = torch.tensor(0.0, device=pred.device)
            info["data"] = 0.0

        # Entropy loss (KL vs prior)
        if prior_mean is not None and self.lambda_entropy > 0:
            ent   = self._kl_gaussian(pred_a, prior_mean, prior_std)
            total = total + self.lambda_entropy * ent
            info["entropy"] = ent.item()

        # Moment constraint
        if moment_target is not None and self.lambda_moment > 0:
            mom   = F.mse_loss(pred_a.mean(dim=0), moment_target)
            total = total + self.lambda_moment * mom
            info["moment"] = mom.item()

        info["total"] = total.item()
        return total, info


# ---------------------------------------------------------------------------
# Tabular Positional Encoding  (CNN-Trans-SPP paper, Section 3.2, eq. 3.4)
# ---------------------------------------------------------------------------
class TabularPositionalEncoding(nn.Module):
    """
    Distribuisce i valori di posizione uniformemente in [0, 1].
    Superiore al sinusoidale per serie temporali finanziarie perché
    preserva l'informazione di posizione assoluta (non solo relativa).

        y_i = 0               if i == 0
        y_i = y_{i-1} + 1/k   if 0 < i < k
        y_i = 1               if i == k-1
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T   = x.size(1)
        pos = torch.zeros(1, 1, 1, device=x.device) if T == 1 else \
              torch.linspace(0.0, 1.0, T, device=x.device).view(1, T, 1)
        return x + self.proj(pos)


# ---------------------------------------------------------------------------
# Pred — CNN-Trans-SPP ibrido (CNN locale + Transformer globale)
# ---------------------------------------------------------------------------
class Pred(nn.Module):
    """
    Modello ibrido CNN + Transformer per predizione di serie finanziarie.

    Architettura ispirata a CNN-Trans-SPP (Li et al., ERA 2024):
      1. Blocco CNN dilated  -> feature locali e pattern di breve termine
      2. Proiezione lineare  -> porta le feature a dimensione d_model
      3. Tabular PE          -> informazione di posizione assoluta in [0,1]
      4. Transformer Encoder -> dipendenze temporali globali (self-attention)
      5. Transformer Decoder -> SOLO per multi-step; query learnable per step
      6. Head lineare        -> predizione finale

    Parametri ottimali da ablation study del paper (Section 4.3.5):
      nhead=8, num_encoder_layers=3, d_model=512

    Gestione multi-step:
      - prediction_steps=1 -> output (B, F)
      - prediction_steps>1 -> output (B, steps, F)
      Se il DataLoader fornisce target (B, F), MrELoss._align supervisiona
      automaticamente solo lo step 0.
    """

    def __init__(
        self,
        num_features:       int,
        window_size:        int,
        dimension:          list[int] = None,
        dilations:          list[int] = None,
        kernel_size:        int   = 3,
        activation:         str   = "leaky_relu",
        # Transformer (parametri ottimali da paper)
        d_model:            int   = 512,
        nhead:              int   = 8,
        num_encoder_layers: int   = 3,
        num_decoder_layers: int   = 3,
        dim_feedforward:    int   = 256,
        dropout:            float = 0.1,
        prediction_steps:   int   = 1,
        # Gradient clipping (0 = disabilitato)
        max_grad_norm:      float = 1.0,
    ) -> None:
        super().__init__()
        if dimension is None:
            dimension = [64, 32, 16]
        if dilations is None:
            dilations = [1, 2, 4]
        assert len(dimension) == len(dilations), \
            "dimension e dilations devono avere la stessa lunghezza"

        self.prediction_steps = prediction_steps
        self.num_features     = num_features
        self.max_grad_norm    = max_grad_norm
        act_fn                = ACTIVATIONS.get(activation, nn.LeakyReLU)

        # 1. CNN Embedding Block
        cnn_layers: list[nn.Module] = []
        in_ch = num_features
        for i, out_ch in enumerate(dimension):
            cnn_layers.extend([
                nn.Conv1d(
                    in_ch, out_ch, kernel_size,
                    padding=dilations[i], dilation=dilations[i],
                ),
                nn.BatchNorm1d(out_ch),
                act_fn(),
            ])
            in_ch = out_ch
        self.conv_block = nn.Sequential(*cnn_layers)

        # 2. Proiezione CNN output -> d_model
        self.input_proj = nn.Linear(dimension[-1], d_model)

        # 3. Tabular Positional Encoding
        self.pos_encoding = TabularPositionalEncoding(d_model)

        # 4. Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        # 5. Transformer Decoder (solo multi-step)
        if prediction_steps > 1:
            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True,
                ),
                num_layers=num_decoder_layers,
            )
            self.query_embed = nn.Embedding(prediction_steps, d_model)

        # 6. Output head
        self.fc = nn.Linear(d_model, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            (B, F)           se prediction_steps == 1
            (B, steps, F)    se prediction_steps >  1
        """
        B = x.size(0)

        # CNN: (B, T, F) -> permute -> conv -> permute -> (B, T, d_cnn)
        h = self.conv_block(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Proiezione + Tabular PE -> (B, T, d_model)
        h = self.pos_encoding(self.input_proj(h))

        # Transformer Encoder -> (B, T, d_model)
        memory = self.transformer_encoder(h)

        if self.prediction_steps > 1:
            queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
            out     = self.transformer_decoder(queries, memory)   # (B, steps, d_model)
            return self.fc(out)                                    # (B, steps, F)
        else:
            return self.fc(memory[:, -1, :])                      # (B, F)

    def _run_phase(
        self,
        train_loader,
        training_cfg,
        criterion:     MrELoss,
        prior_mean:    Optional[torch.Tensor],
        prior_std:     Optional[torch.Tensor],
        moment_target: Optional[torch.Tensor],
        data_weight:   float = 1.0,
        label:         str   = "",
        epochs:        Optional[int] = None,
    ) -> None:
        n_epochs  = epochs or training_cfg.epochs
        optimizer = OPTIMIZERS.get(training_cfg.optimizer, torch.optim.Adam)(
            self.parameters(),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )
        self.train()
        device = next(self.parameters()).device

        # ── Multi-GPU: avvolge il modello con DataParallel se disponibili >1 GPU ──
        # DataParallel divide il batch tra le GPU disponibili automaticamente.
        # Nota: usa `self` (non il wrapper) per optimizer e state_dict.
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            model = torch.nn.DataParallel(self)
            if not getattr(self, "_multigpu_logged", False):
                print(f"[{label}] Multi-GPU attivo: {n_gpus} GPU con DataParallel")
                self._multigpu_logged = True
        else:
            model = self

        if prior_mean    is not None: prior_mean    = prior_mean.to(device)
        if prior_std     is not None: prior_std     = prior_std.to(device)
        if moment_target is not None: moment_target = moment_target.to(device)

        for epoch in range(n_epochs):
            total_l      = 0.0
            total_data_l = 0.0
            nan_batches  = 0

            for bx, by in train_loader:
                # pin_memory=True nel DataLoader rende questo trasferimento asincrono
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = model(bx)
                loss, info = criterion(
                    pred, by, prior_mean, prior_std, moment_target, data_weight
                )

                # Salta il batch se la loss è NaN/inf (difesa a strato finale)
                if not torch.isfinite(loss):
                    nan_batches += 1
                    continue

                loss.backward()

                # Gradient clipping — previene NaN da gradienti esplosi
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=self.max_grad_norm
                    )

                optimizer.step()
                total_l      += loss.item()
                total_data_l += info.get("data", 0.0)

            if (epoch + 1) % 10 == 0:
                n = max(len(train_loader) - nan_batches, 1)
                nan_msg = f" | NaN batch saltati: {nan_batches}" if nan_batches else ""
                print(
                    f"[{label}] Epoch {epoch+1}/{n_epochs} "
                    f"| Tot Loss: {total_l/n:.6f} "
                    f"| Data(MSE): {total_data_l/n:.6f}"
                    f"{nan_msg}"
                )

    def fit(
        self,
        train_loader,
        training_cfg,
        moment_target: Optional[torch.Tensor] = None,
    ) -> None:
        mre_cfg = getattr(training_cfg, "mre", None)
        if not mre_cfg or not mre_cfg.enabled:
            self._run_phase(
                train_loader, training_cfg,
                MrELoss(0, 0), None, None, None, 1.0, "MSE",
            )
            return

        criterion = MrELoss(mre_cfg.lambda_entropy, mre_cfg.lambda_moment)
        prior     = PriorEstimator()
        prior.fit(train_loader)

        mode = getattr(mre_cfg, "update_mode", "simultaneous")
        if mode == "sequential":
            ep1 = getattr(mre_cfg, "epochs_phase1", training_cfg.epochs // 3)
            self._run_phase(
                train_loader, training_cfg, criterion,
                prior.mean, prior.std, moment_target, 0.0, "SEQ-P1", ep1,
            )
            prior.fit(train_loader)
            self._run_phase(
                train_loader, training_cfg, criterion,
                prior.mean, prior.std, None, 1.0, "SEQ-P2",
                training_cfg.epochs - ep1,
            )
        else:
            self._run_phase(
                train_loader, training_cfg, criterion,
                prior.mean, prior.std, moment_target, 1.0, "SIMULT",
            )