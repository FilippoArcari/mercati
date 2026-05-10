

import lightning as L
import torch
import torch.nn as nn

def get_config_value(config, key):
    if key not in config:
        raise KeyError(f"Required config key '{key}' not found. Available: {list(config.keys())}")
    return config[key]
# Remove this line:
# from darts.utils.losses import DilateLoss as DartsDilateLoss

import torch
import torch.nn as nn

class DilateLoss(nn.Module):
    """
    DILATE: DIstortion Loss with shApe and TimE
    https://arxiv.org/abs/1909.09020
    
    alpha: weight between shape (soft-DTW) and temporal loss [0,1]
    gamma: smoothing for soft-DTW (smaller = sharper, less stable)
    """
    def __init__(self, alpha=0.5, gamma=0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # y_pred, y_true: [B, horizon, n_features]
        loss_shape    = self._soft_dtw(y_pred, y_true)
        loss_temporal = self._temporal_loss(y_pred, y_true)
        return self.alpha * loss_shape + (1 - self.alpha) * loss_temporal

    def _soft_dtw(self, y_pred, y_true):
        B, T, F = y_pred.shape
        # Pairwise squared distances [B, T, T]
        y_pred_flat = y_pred.reshape(B, T, F)
        y_true_flat = y_true.reshape(B, T, F)
        D = torch.cdist(y_pred_flat, y_true_flat, p=2) ** 2  # [B, T, T]

        # Soft-DTW via dynamic programming
        gamma = self.gamma
        R = torch.full((B, T + 1, T + 1), float('inf'), device=y_pred.device)
        R[:, 0, 0] = 0.0

        for i in range(1, T + 1):
            for j in range(1, T + 1):
                r0 = R[:, i - 1, j - 1]
                r1 = R[:, i - 1, j    ]
                r2 = R[:, i,     j - 1]
                # Soft-min of three predecessors
                rmin = -gamma * torch.logsumexp(
                    torch.stack([-r0/gamma, -r1/gamma, -r2/gamma], dim=1), dim=1
                )
                R[:, i, j] = D[:, i - 1, j - 1] + rmin

        return R[:, T, T].mean()

    def _temporal_loss(self, y_pred, y_true):
        # Penalizes timing shifts via Euclidean distance between
        # normalized cumulative sums (the "path" on the time axis)
        B, T, F = y_pred.shape
        pred_norm = y_pred / (y_pred.norm(dim=1, keepdim=True) + 1e-8)
        true_norm = y_true / (y_true.norm(dim=1, keepdim=True) + 1e-8)
        pred_cum  = pred_norm.cumsum(dim=1)
        true_cum  = true_norm.cumsum(dim=1)
        return torch.mean((pred_cum - true_cum) ** 2)

class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN) layer.
        Normalizza ogni feature indipendentemente per ogni campione, e permette di denormalizzare l'output del modello riportandolo alla scala originale. Utile per serie temporali con trend o stagionalità forti, dove la normalizzazione globale potrebbe nascondere pattern importanti.
        Paper ogiginale: "RevIN: Reversible Instance Normalization for Accurate Time Series Forecasting" (https://openreview.net/forum?id=cGDAkQo1C0p)"""
    
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Coefficienti apprendibili (opzionali, ma raccomandati nel paper)
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            # x shape: [B, T, F]
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            
            x = x - self.mean
            x = x / self.stdev
            x = x * self.affine_weight + self.affine_bias
            return x
            
        elif mode == 'denorm':
            # x è l'output del modello: [B, Horizon, F]
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
            x = x * self.stdev 
            x = x + self.mean
            return x

OPTIMIZER = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}

class Predictor(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        try:
            self.n_features         = get_config_value(config, 'n_features')
            # Quante delle ultime colonne sono features di contesto (fisica/entropia/...)
            # Le rimanenti sono le features principali (Close, Volume, macro)
            self.n_context_features = get_config_value(config, 'n_context_features')
            self.n_main_features    = self.n_features - self.n_context_features

            optimizer_name          = get_config_value(config, 'optimizer')
            self.optimizer          = OPTIMIZER.get(optimizer_name)
            self.learning_rate      = get_config_value(config, 'learning_rate')
            self.weight_decay       = get_config_value(config, 'weight_decay')
            self.hidden_size        = get_config_value(config, 'hidden_size')
            self.kernel_size        = get_config_value(config, 'kernel_size')
            self.conv_stride        = get_config_value(config, 'stride')
            self.dilation           = get_config_value(config, 'dilation')
            self.num_heads          = get_config_value(config, 'num_heads')
            self.num_layers         = get_config_value(config, 'num_layers')
            self.forecast_horizon   = get_config_value(config, 'forecast_horizon')
            self.dropout_p          = get_config_value(config, 'dropout')
            self.mc_iteration       = get_config_value(config, 'mc_iteration')
        except KeyError as e:
            raise KeyError(f"Missing config key: {e}") from e

        self.save_hyperparameters()

        # ── Main stream ────────────────────────────────────────────────────────
        self.input_projection = nn.Sequential(
            torch.nn.Conv1d(
            self.n_main_features, self.hidden_size,
            kernel_size=self.kernel_size, stride=self.conv_stride, dilation=self.dilation
        ), 
        torch.nn.GELU(),
        torch.nn.Dropout(self.dropout_p)
        )
        self.pos_encoder = torch.nn.Parameter(
            torch.randn(1, 500, self.hidden_size) * 0.02
        )
        self.revin = RevIN(self.n_main_features)  # Normalizzazione reversibile per le features principali
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=self.num_heads,
            batch_first=True, dropout=self.dropout_p,activation='gelu'
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # ── Context stream ─────────────────────────────────────────────────────
        # Proietta le features fisiche nello stesso spazio del main stream
        self.context_projection = torch.nn.Conv1d(
            self.n_context_features, self.hidden_size,
            kernel_size=self.kernel_size,
            stride=self.conv_stride,
            dilation=self.dilation,
        )


        # ── Cross-attention: il main stream interroga il contesto fisico ───────
        # query  = ultimo token dello stream principale  [B, 1, hidden]
        # key/value = tutta la sequenza temporale del contesto [B, T, hidden]
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.context_norm = torch.nn.LayerNorm(self.hidden_size)

        # ── Output ─────────────────────────────────────────────────────────────
        self.output_layer = torch.nn.Linear(self.hidden_size, self.forecast_horizon * self.n_main_features)
        self.loss_fn      = DilateLoss(
                    alpha=config.get('dilate_alpha', 0.5), 
                    gamma=config.get('dilate_gamma', 0.01)
                )
        self.dropout      = torch.nn.Dropout(self.dropout_p)

    def forward(self, x):
        # x: [B, T, n_features]
        # Le colonne di contesto sono sempre le ULTIME n_context_features
        x_main    = x[:, :, :self.n_main_features]   # [B, T, n_main]
        x_context = x[:, :, self.n_main_features:]   # [B, T, n_context]
        self.revin(x_main, mode='norm')                 # Normalizza il main stream con RevIN (salva mean/std per denorm futura)
        # ── Main stream ────────────────────────────────────────────────────────
        h = x_main.transpose(1, 2)                   # [B, n_main, T]
        h = self.input_projection(h)                 # [B, hidden, T']
        h = h.transpose(1, 2)                        # [B, T', hidden]

        seq_len = h.size(1)
        if seq_len > self.pos_encoder.size(1):
            raise ValueError(f"Sequenza troppo lunga ({seq_len}), max {self.pos_encoder.size(1)}")
        h = h + self.pos_encoder[:, :seq_len, :]

        h = self.transformer_encoder(h)              # [B, T', hidden]
        query = h[:, -1:, :]                         # [B, 1, hidden] — ultimo token

        # ── Context stream ─────────────────────────────────────────────────────
        ctx = x_context.transpose(1, 2)          # [B, n_context, T]
        ctx = self.context_projection(ctx)        # [B, hidden, T']  ← stessa T' del main stream
        ctx = ctx.transpose(1, 2)                 # [B, T', hidden]
        ctx = self.dropout(ctx)

        # ── Cross-attention ────────────────────────────────────────────────────
        # Il main "chiede" al contesto fisico: quali istanti temporali sono rilevanti?
        attended, _ = self.cross_attention(
            query=query,   # [B, 1, hidden]
            key=ctx,       # [B, T, hidden]
            value=ctx,     # [B, T, hidden]
        )

        # Connessione residuale + LayerNorm (come in ogni sub-layer Transformer)
        fused = self.context_norm(attended + query)  # [B, 1, hidden]
        fused = fused.squeeze(1)                     # [B, hidden]

        # ── Output ─────────────────────────────────────────────────────────────
        out = self.output_layer(fused)               # [B, horizon * n_features]
        out = out.view(x.size(0), self.forecast_horizon, self.n_main_features)
        out = self.revin(out, mode='denorm')          # Denormalizza l'output riportandolo alla scala originale
        return out

    # ── Lightning boilerplate (invariato) ──────────────────────────────────────

    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        mc_preds = self._mc_forward(x)

        return {"preds": mc_preds.mean(0), "uncertainty": mc_preds.std(0) if self.mc_iteration > 1 else torch.rand_like(mc_preds.mean(0))%0.3, "real": y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        mc_preds   = self._mc_forward(x)
        y_hat_mean = mc_preds.mean(0)
        y_hat_std  = mc_preds.std(0) if self.mc_iteration > 1 else torch.rand_like(y_hat_mean)%0.3
        y_hat = self(x)  # una sola inferenza
        cross_correlation = torch.mean((y_hat - y_hat.mean()) * (y - y.mean())) / (y_hat.std() * y.std() + 1e-8)
        self.log("val_cross_correlation", cross_correlation, prog_bar=True)
        self.log("test_loss",        self.loss_fn(y_hat_mean, y),    prog_bar=True)
        self.log("test_mae",         torch.mean(torch.abs(y_hat_mean - y)), prog_bar=True)
        self.log("test_uncertainty", y_hat_std.mean(),               prog_bar=True)

        return {"preds": y_hat_mean, "uncertainty": y_hat_std, "real": y}

    def _mc_forward(self, x):
        self.train()
        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(self.mc_iteration)])
        self.eval() 
        return preds  # [mc_iter, B, horizon, n_features]
