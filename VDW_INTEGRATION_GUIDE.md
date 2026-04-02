"""
GUIDA: Integrazione di calibrate_vdw in thermodynamics.py

Hai 3 livelli di integrazione per usare i parametri Van der Waals 
calibrati empiricamente:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LIVELLO 1: Calibrazione ONCE all'inizio del training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Esperienza:
  - Calibra i parametri a, b una volta dal dataset storico
  - Usa i valori fissi per tutto il training
  - ⚡ VELOCE, ma meno adattivo

Dove: main.py o modelli/train.py

    from modelli.thermodynamics import init_vdw_calibration, compute_thermodynamic_features

    # 1. Calibra una volta all'inizio dal training set
    vdw_calibration = init_vdw_calibration(
        df_raw=train_df,
        ticker_cols=cfg.data.tickers,
        verbose=True
    )
    # Output: dict con parametri a, b per ogni ticker
    # Diagnostiche: ζr, ζQ, momentum_strength, avvertenze

    # 2. Usa nella computazione delle feature
    thermo_features = compute_thermodynamic_features(
        df_raw=df,
        ticker_cols=cfg.data.tickers,
        rates_col="DGS10",
        vdw_calibration=vdw_calibration,  # ← Nuovi parametri!
    )
    
    # 3. Salva i parametri per reload in inference
    import json
    vdw_dict = {
        ticker: {
            'a': params.a,
            'b': params.b,
            'zeta_r': params.zeta_r,
            'lambda_impact': params.lambda_impact,
        }
        for ticker, params in vdw_calibration.items()
    }
    with open('checkpoints/vdw_params.json', 'w') as f:
        json.dump(vdw_dict, f, indent=2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LIVELLO 2: Usare nella classe QuantumThermodynamicProcessor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Esperienza:
  - Istanzia il processore con parametri calibrati
  - Utile per preprocessing step (denoising, feature engineering)
  - ⚡ MODULARE, facile da integrare in pipeline

Dove: modelli/pred.py o modelli/trading_env.py

    from modelli.thermodynamics import (
        QuantumThermodynamicProcessor,
        init_vdw_calibration,
    )

    # Setup unico
    vdw_calibration = init_vdw_calibration(train_df, ticker_cols)
    
    # Estrai parametri medi per il portafoglio
    a_mean = np.mean([p.a for p in vdw_calibration.values()])
    b_mean = np.mean([p.b for p in vdw_calibration.values()])

    # Istanzia con parametri calibrati
    processor = QuantumThermodynamicProcessor(
        n_tickers=len(ticker_cols),
        r_param=1.0,
        a_vdw=a_mean,
        b_vdw=b_mean,
    )

    # Usa nel preprocessing
    thermo_features = processor.get_thermodynamic_features(
        df_prices=train_df[ticker_cols],
        df_volumes=train_df[[f"{t}_Volume" for t in ticker_cols]],
        rates_10y=train_df["DGS10"],
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LIVELLO 3: Per-asset calibration (ADVANCED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Esperienza:
  - Calibra parametri specifici per ogni asset
  - Applica a singoli ticker dans il trading environment
  - 🔥 PIÙ PRECISO, ma computazionalmente più costoso

Dove: modelli/trading_env.py (state computation)

    from modelli.calibrate_vdw import calibrate_single, vdw_pressure

    # Nel TradingEnv, durante compute_state():
    for ticker in self.tickers:
        price_hist = self.prices[ticker]
        volume_hist = self.volumes[ticker]
        
        # Calibra specificamente per questo asset
        vdw_params = calibrate_single(
            prices=price_hist,
            volumes=volume_hist,
            n_assets=len(self.tickers),
        )
        
        # Calcola pressione con parametri specifici
        P = vdw_pressure(
            log_volume=np.log(current_volume + 1e-8),
            temperature=current_entropy,
            vdw_params=vdw_params,
            n=1,  # singolo asset
        )
        
        state[ticker]['pressure'] = P

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COSA CAMBIA FISICAMENTE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIMA (parametri fissi):
  P = (kb × T) / (V - 0.01) - 0.1 × (1 / V)²
  
  → a=0.1, b=0.01 erano ipotesi arbitrarie

DOPO (parametri calibrati):
  P = (kb × T) / (V - b×n) - a × (n / V)²
  
  dove:
    a = λ_impact × Q̄² × momentum_strength
      = impatto di prezzo × scala mercato × herding empirico
    
    b = V_floor / n
      = volume minimo strutturale / numero asset
  
  → Basati su leggi empiriche di Gabaix et al. 2003

BENEFICI:
  1. ✓ Pressione più realistica (calibrata su dati reali)
  2. ✓ Herding naturale dal momentum_strength
  3. ✓ Liquidità strutturale dal volume floor
  4. ✓ Esponenti tail (ζr, ζQ) per diagnostica
  5. ✓ Diagnostiche/avvertenze su anomalie di mercato

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPLEMENTAZIONE CONSIGLIATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Nel main.py (o train_model.py):

    import hydra
    from modelli.thermodynamics import init_vdw_calibration, compute_thermodynamic_features

    @hydra.main(...)
    def main(cfg):
        # 1. Load data
        train_df = load_data(cfg)
        
        # 2. Calibra VdW una volta
        vdw_calibration = init_vdw_calibration(
            df_raw=train_df,
            ticker_cols=cfg.data.tickers,
            verbose=True,
        )
        
        # 3. Usa nelle feature termodinamiche
        thermo_features = compute_thermodynamic_features(
            df_raw=train_df,
            ticker_cols=cfg.data.tickers,
            rates_col="DGS10",
            vdw_calibration=vdw_calibration,
        )
        
        # 4. Concatena alle altre feature
        X = pd.concat([X_base, thermo_features], axis=1)
        
        # 5. Addestra modello normalmente
        ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEBUGGING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Se vdw_calibration è vuoto o ha parametri strani:

    # 1. Controlla i dati storici
    print(train_df.head(), train_df.info())
    
    # 2. Calibra con verbose=True per vedere diagnostiche
    vdw_cal = init_vdw_calibration(train_df, tickers, verbose=True)
    
    # 3. Verifica i parametri
    for ticker, params in vdw_cal.items():
        print(f"{ticker}:")
        print(f"  a={params.a:.6f}, b={params.b:.6f}")
        print(f"  ζr={params.zeta_r:.2f} (atteso ~3.0)")
        print(f"  ζQ={params.zeta_q:.2f} (atteso ~1.5)")
        if params.warnings:
            print(f"  ⚠ {params.warnings}")
    
    # 4. Se problemi, fallback ai valori hardcoded:
    thermo_features = compute_thermodynamic_features(
        df_raw=train_df,
        ticker_cols=tickers,
        vdw_calibration=None,  # Ignora calibrazione
    )
"""
