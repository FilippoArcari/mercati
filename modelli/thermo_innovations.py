"""
modelli/thermo_innovations.py

INNOVAZIONI TERMODINAMICHE PER DDPG
====================================

Sistema avanzato di analisi termodinamica che risolve:
- Sbilanciamento BUY/SELL (23:3 ratio)
- Varianza alta tra fold
- Mancanza di segnali oggettivi di vendita

NOVITÀ:
1. Adaptive Monetary Lag (Kalman Filter)
2. Work-Price Efficiency Detector
3. Thermodynamic Phase Detector + Dynamic Thresholds

Autore: Analisi termodinamica avanzata v3.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import Enum


class MarketPhase(Enum):
    """Fasi termodinamiche del mercato"""
    EXPANSION = "Espansione"      # W↑ P↑ T moderata
    COMPRESSION = "Compressione"  # W↓ P↑ T↑
    TRANSITION = "Transizione"    # Cambio regime
    CHAOS = "Caos"                # Alta entropia


@dataclass
class ThermoState:
    """Stato termodinamico completo"""
    pressure: float
    temperature: float
    entropy: float
    work_cumulative: float
    work_derivative: float
    phase: MarketPhase
    stress_zscore: float
    efficiency_index: float
    monetary_lag: int


class AdaptiveLagEstimator:
    """
    INNOVAZIONE 1: Stima adattiva del lag monetario con Kalman Filter
    
    Invece di lag fisso (58-72 giorni), stima dinamica tra 30-120 giorni
    che si adatta al regime di mercato.
    """
    
    def __init__(self, min_lag: int = 30, max_lag: int = 120):
        self.min_lag = min_lag
        self.max_lag = max_lag
        
        # Kalman state: [lag_estimate, lag_velocity]
        self.state = np.array([60.0, 0.0])
        self.P = np.eye(2) * 10
        
        # Process noise
        self.Q = np.array([[1.0, 0.0],
                           [0.0, 0.5]])
        
        # Measurement noise
        self.R = np.array([[5.0]])
        
    def update(self, pressure_series: pd.Series, rates_series: pd.Series) -> int:
        """Aggiorna stima del lag ottimale"""
        # Prediction step
        F = np.array([[1.0, 1.0],
                      [0.0, 0.95]])
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
        # Measurement
        best_lag, max_corr = self._find_best_lag(pressure_series, rates_series)
        
        if max_corr < 0.3:
            return int(np.clip(self.state[0], self.min_lag, self.max_lag))
        
        # Update step
        H = np.array([[1.0, 0.0]])
        y = best_lag - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        
        self.state = self.state + K.flatten() * y
        self.P = (np.eye(2) - K @ H) @ self.P
        
        return int(np.clip(self.state[0], self.min_lag, self.max_lag))
    
    def _find_best_lag(self, pressure: pd.Series, rates: pd.Series) -> Tuple[int, float]:
        """Trova lag con massima cross-correlation"""
        inv_rates = 1 / (rates + 1e-5)
        
        test_lags = range(self.min_lag, self.max_lag + 1, 5)
        correlations = []
        
        for lag in test_lags:
            shifted = inv_rates.shift(lag).dropna()
            aligned = pressure.loc[shifted.index]
            
            if len(aligned) < 50:
                correlations.append(0.0)
                continue
            
            corr = aligned.corr(shifted)
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        max_idx = np.argmax(np.abs(correlations))
        return list(test_lags)[max_idx], abs(correlations[max_idx])


class WorkPriceEfficiencyDetector:
    """
    INNOVAZIONE 2: Oscillatore di Efficienza Work-Price
    
    Identifica "Rally da Esaurimento" quando il mercato compie molto
    lavoro ma il prezzo non sale proporzionalmente.
    
    Efficiency > 1.5  → Dissipazione (SELL)
    Efficiency ~ 1.0  → Efficiente (HOLD)
    Efficiency < -0.5 → Accumulo (BUY)
    """
    
    def __init__(self, window: int = 20):
        self.window = window
        
    def compute_efficiency(
        self, 
        work_series: pd.Series, 
        price_series: pd.Series
    ) -> pd.Series:
        """Calcola indice di efficienza Work/Price"""
        delta_work = work_series.diff(self.window)
        delta_price = price_series.pct_change(self.window)
        
        delta_price = delta_price.replace(0, 1e-8)
        
        raw_efficiency = delta_work / (delta_price.abs() + 1e-8)
        
        # Normalizza con z-score robusto (MAD)
        median = raw_efficiency.rolling(120).median()
        mad = (raw_efficiency - median).abs().rolling(120).median()
        
        normalized = (raw_efficiency - median) / (1.4826 * mad + 1e-8)
        
        return normalized.fillna(0)
    
    def detect_exhaustion_rally(self, efficiency: pd.Series, threshold: float = 1.5) -> pd.Series:
        """Identifica periodi di rally da esaurimento"""
        is_high = efficiency > threshold
        persistent = is_high.rolling(5).sum() >= 3
        
        return persistent


class ThermodynamicPhaseDetector:
    """
    INNOVAZIONE 3: Rilevatore di Fase Termodinamica
    
    Classifica mercato in 4 fasi con soglie stress adattive:
    
    EXPANSION:    W↑ P↑ T moderata → stress threshold = 2.0
    COMPRESSION:  W↓ P↑ T↑ → stress threshold = 0.5
    TRANSITION:   Intermedio → stress threshold = 1.0
    CHAOS:        S↑↑ T↑↑ → NO TRADE
    """
    
    def detect_phase(
        self,
        pressure: float,
        temperature: float,
        work_delta: float,
        entropy: float
    ) -> MarketPhase:
        """Classifica fase termodinamica"""
        def norm(x):
            return (x + 3) / 6
        
        p = norm(pressure)
        t = norm(temperature)
        w = norm(work_delta)
        s = norm(entropy)
        
        if s > 0.8 and t > 0.8:
            return MarketPhase.CHAOS
        
        elif w > 0.6 and p > 0.5 and t < 0.7 and s < 0.5:
            return MarketPhase.EXPANSION
        
        elif w < 0.4 and p > 0.6 and t > 0.6:
            return MarketPhase.COMPRESSION
        
        else:
            return MarketPhase.TRANSITION
    
    def get_stress_threshold(self, phase: MarketPhase) -> float:
        """Soglia di stress adattiva per fase"""
        thresholds = {
            MarketPhase.EXPANSION: 2.0,
            MarketPhase.COMPRESSION: 0.5,
            MarketPhase.TRANSITION: 1.0,
            MarketPhase.CHAOS: -999,
        }
        return thresholds[phase]


class AdvancedThermoAnalyzer:
    """Sistema integrato che combina tutte le innovazioni"""
    
    def __init__(self):
        self.lag_estimator = AdaptiveLagEstimator()
        self.efficiency_detector = WorkPriceEfficiencyDetector()
        self.phase_detector = ThermodynamicPhaseDetector()
        
    def analyze(
        self,
        pressure: pd.Series,
        temperature: pd.Series,
        entropy: pd.Series,
        work: pd.Series,
        price: pd.Series,
        rates: pd.Series,
    ) -> pd.DataFrame:
        """Analisi termodinamica completa"""
        results = pd.DataFrame(index=pressure.index)
        
        # 1. Lag monetario adattivo
        lag = self.lag_estimator.update(pressure, rates)
        results['Thm_MonetaryLag'] = lag
        
        # 2. Efficienza Work-Price
        efficiency = self.efficiency_detector.compute_efficiency(work, price)
        results['Thm_Efficiency'] = efficiency
        
        # 3. Fase termodinamica
        phases = []
        stress_thresholds = []
        
        p_norm = (pressure - pressure.mean()) / (pressure.std() + 1e-8)
        t_norm = (temperature - temperature.mean()) / (temperature.std() + 1e-8)
        s_norm = (entropy - entropy.mean()) / (entropy.std() + 1e-8)
        w_delta = work.diff(20)
        w_norm = (w_delta - w_delta.mean()) / (w_delta.std() + 1e-8)
        
        for i in range(len(pressure)):
            phase = self.phase_detector.detect_phase(
                p_norm.iloc[i],
                t_norm.iloc[i],
                w_norm.iloc[i],
                s_norm.iloc[i]
            )
            phases.append(phase.value)
            stress_thresholds.append(self.phase_detector.get_stress_threshold(phase))
        
        results['Thm_Phase'] = phases
        results['Thm_StressThreshold'] = stress_thresholds
        
        # 4. Segnale di vendita combinato
        exhaustion = self.efficiency_detector.detect_exhaustion_rally(efficiency)
        
        rates_lagged = rates.shift(lag)
        p_expected = 1 / (rates_lagged + 1e-5)
        stress_div = pressure - p_expected
        stress_zscore = (stress_div - stress_div.mean()) / (stress_div.std() + 1e-8)
        
        stress_exceeded = stress_zscore > pd.Series(stress_thresholds, index=pressure.index)
        
        chaos_phase = pd.Series(phases) == MarketPhase.CHAOS.value
        
        results['Thm_SellSignal'] = (exhaustion | stress_exceeded) & (~chaos_phase)
        results['Thm_StressZScore'] = stress_zscore
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONI DI INTEGRAZIONE
# ═══════════════════════════════════════════════════════════════════════════

def compute_advanced_thermo_features(
    df: pd.DataFrame,
    pressure_col: str = 'Market_Pressure',
    temp_col: str = 'Market_Temperature', 
    entropy_col: str = 'Market_Entropy',
    work_col: str = 'Market_Work_Cum',
    price_col: str = 'Close',
    rates_col: str = 'DGS10',
) -> pd.DataFrame:
    """
    Calcola feature termodinamiche avanzate da aggiungere al dataset
    
    IMPORTANTE: Questa funzione ESTENDE il DataFrame esistente con nuove colonne.
    Usala DOPO aver calcolato le feature termodinamiche base.
    
    Returns:
        DataFrame con colonne aggiunte:
        - Thm_Phase
        - Thm_Efficiency  
        - Thm_MonetaryLag
        - Thm_StressThreshold
        - Thm_SellSignal
        - Thm_StressZScore
    """
    analyzer = AdvancedThermoAnalyzer()
    
    pressure = df[pressure_col]
    temperature = df[temp_col]
    entropy = df[entropy_col]
    work = df[work_col]
    price = df[price_col]
    
    # Se non ci sono tassi, usa default
    if rates_col in df.columns:
        rates = df[rates_col]
    else:
        print(f"[ThermoInnovations] rates_col '{rates_col}' non trovato, uso default 0.045")
        rates = pd.Series(0.045, index=df.index)
    
    thermo_features = analyzer.analyze(
        pressure=pressure,
        temperature=temperature,
        entropy=entropy,
        work=work,
        price=price,
        rates=rates,
    )
    
    # Merge con dataset originale
    df = pd.concat([df, thermo_features], axis=1)
    
    return df


def should_sell_now(thermo_state: pd.Series) -> bool:
    """
    Decisione di vendita basata su analisi termodinamica
    
    Usabile nel reward shaping del DDPG per guidare l'agente.
    
    Args:
        thermo_state: riga del DataFrame con feature termodinamiche
        
    Returns:
        True se segnali indicano vendita
    """
    if 'Thm_SellSignal' in thermo_state:
        return bool(thermo_state['Thm_SellSignal'])
    else:
        return False


def get_dynamic_sell_threshold(thermo_state: pd.Series) -> float:
    """
    Soglia di stress adattiva per il timestep corrente
    
    Args:
        thermo_state: riga del DataFrame con feature termodinamiche
        
    Returns:
        soglia Z-score sopra la quale vendere
    """
    if 'Thm_StressThreshold' in thermo_state:
        return thermo_state['Thm_StressThreshold']
    else:
        return 1.0


def get_phase_aware_noise_scale(current_phase: str, base_sigma: float = 0.2) -> float:
    """
    Adatta rumore di esplorazione (OU) in base alla fase termodinamica
    
    In CHAOS → riduci esplorazione (mercato imprevedibile)
    In EXPANSION → aumenta esplorazione (opportunità)
    
    Args:
        current_phase: fase corrente (str)
        base_sigma: sigma base dell'OU noise
        
    Returns:
        sigma adattato
    """
    phase_multipliers = {
        'Espansione': 1.2,
        'Compressione': 0.8,
        'Transizione': 1.0,
        'Caos': 0.3,
    }
    
    multiplier = phase_multipliers.get(current_phase, 1.0)
    return base_sigma * multiplier


# ═══════════════════════════════════════════════════════════════════════════
# STANDARD FEATURE SET
# ═══════════════════════════════════════════════════════════════════════════

# Feature termodinamiche standard da usare come stato per DDPG
# IMPORTANTE: Questo set FISSO previene shape mismatch tra fold
STANDARD_THERMO_FEATURES = [
    'Market_Pressure',
    'Market_Temperature',
    'Market_Entropy',
    'Market_Work_Cum',
    'Thm_Efficiency',
    'Thm_MonetaryLag',
    'Thm_StressZScore',
]


if __name__ == '__main__':
    print("=" * 70)
    print("THERMO INNOVATIONS - Modulo caricato correttamente")
    print("=" * 70)
    print("\nFunzioni disponibili:")
    print("  • compute_advanced_thermo_features(df)")
    print("  • should_sell_now(thermo_state)")
    print("  • get_dynamic_sell_threshold(thermo_state)")
    print("  • get_phase_aware_noise_scale(phase, sigma)")
    print("\nConstanti:")
    print(f"  • STANDARD_THERMO_FEATURES: {STANDARD_THERMO_FEATURES}")
    print("=" * 70)