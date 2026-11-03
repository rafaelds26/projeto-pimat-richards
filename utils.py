# utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Richards(t, A, K, B, Q, nu):
    t_clipped = np.clip(t, -500, 500)
    exp_term = np.exp(-B * t_clipped)
    base = 1.0 + Q * exp_term
    inv_nu = 1.0 / np.maximum(nu, 1e-5)
    return A + (K - A) / np.power(base, inv_nu)

def Richards_Derivative(t, A, K, B, Q, nu):
    clipped_t = np.clip(t, -50, 50)
    exp_term = np.exp(-B * clipped_t)
    base = 1.0 + Q * exp_term
    inv_nu = 1.0 / np.maximum(nu, 1e-5)
    numerator = (K - A) * Q * exp_term * B * np.power(base, -inv_nu - 1.0)
    denominator = np.maximum(nu, 1e-5)
    return numerator / denominator
    
def Adjustment_Metrics(y_true, y_pred, num_params=5):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    sse = np.sum((y_true - y_pred) ** 2)
    rmse = np.sqrt(sse / len(y_true))
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_total == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - (sse / ss_total)
        
    return sse, rmse, r2

def PlotResults(data, Window, TrainLimit, InitIndex, FinalIndex, n_prediction, nr2, vr2, ar2, filename, TIMEFRAME_LABEL, REF_SERIES):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    y_global = data['Acumulados'].values
    t_global = np.arange(1, len(y_global) + 1)

    # -----------------------------------------
    # PAINEL 1: Série Diária / Variação
    # -----------------------------------------
    ax1.plot(t_global, np.diff(y_global, prepend=y_global[0]), color='lightgray', linestyle='-', linewidth=1.5, label='Dados Reais')
    ax1.axvline(x=TrainLimit, color='orange', linestyle='--', linewidth=2, label=f'Limite Treino')
    ax1.plot(t_global, np.diff(n_prediction, prepend=n_prediction[0]), color='red', linestyle='-', linewidth=1.5, label='Previsão')
    ax1.set_title(f'Série Diária ({REF_SERIES.upper()}) - TimeFrame: {TIMEFRAME_LABEL} | Janela: {Window}')
    ax1.set_ylabel('Variação / Preço')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # -----------------------------------------
    # PAINEL 2: Dados Acumulados + Projeção
    # -----------------------------------------
    ax2.plot(t_global, y_global, color='lightgray', linestyle='-', linewidth=1.5, label='Acumulado Real')
    ax2.axvline(x=TrainLimit, color='orange', linestyle='--', linewidth=2, label=f'Limite Treino')
    ax2.plot(t_global, n_prediction, color='royalblue', linestyle='-', linewidth=1.5, label='Projeção Richards')
    ax2.set_title(f'Dados Acumulados | R² Treino: {nr2:.3f} | R² Acumulado: {vr2:.3f}')
    ax2.set_ylabel('Valor Acumulado')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # -----------------------------------------
    # PAINEL 3: 1ª Derivada Global
    # -----------------------------------------
    ax3.plot(t_global, y_global, color='lightgray', linestyle='-', linewidth=1.5, label='Série Real')
    ax3.axvline(x=TrainLimit, color='orange', linestyle='--', linewidth=2, label=f'Limite Treino')
    
    ax3_sec = ax3.twinx()
    ax3_sec.plot(t_global, np.diff(np.diff(n_prediction, prepend=n_prediction[0]), prepend=n_prediction[0]), color='green', linestyle='-.', linewidth=1.5, label='1ª Derivada')
    ax3_sec.tick_params(axis='y', labelcolor='green')
    ax3_sec.grid(False)

    ax3.set_title(f'Análise de Derivada | R² Derivada: {ar2:.3f}')
    ax3.set_ylabel('Preço Acumulado')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    plt.close(fig)
    print(f'Gráfico com 3 painéis salvo com sucesso em: {filename}')