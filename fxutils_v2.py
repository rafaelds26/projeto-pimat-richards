import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

def ReadDataFromDataBase(db_parametres):
    con = duckdb.connect(db_parametres['Path'])
    #--------------------------------------------------------------------
    tf_mapping = {'Tick-By-Tick': None, '1M': '1 minute', '5M': '5 minutes', '15M': '15 minutes',
                  '30M': '30 minutes', '1H': '1 hour', '4H': '4 hours', '1D': '1 day', 
                  '1W': '1 week', '1MW': '1 month'}
    interval_sql = tf_mapping.get(db_parametres['dtframe'], '')
    #--------------------------------------------------------------------
    if db_parametres['dtframe'] == 'Tick-By-Tick':
        query = f"SELECT timestamp AS Date, {db_parametres['dtserie']} AS val FROM forex_ticks WHERE pair = 'EURUSD' ORDER BY timestamp ASC LIMIT {db_parametres['dtsize']};"
    else:
        query = f"""
        WITH ohlc AS (
            SELECT time_bucket(INTERVAL '{interval_sql}', timestamp) AS Date,
                FIRST({db_parametres['dtserie']}) AS Open, MAX({db_parametres['dtserie']}) AS High,
                MIN({db_parametres['dtserie']}) AS Low, LAST({db_parametres['dtserie']}) AS Close
            FROM forex_ticks WHERE pair = 'EURUSD' GROUP BY Date ORDER BY Date ASC
        )
        SELECT Date, {db_parametres['dttype']} AS val FROM ohlc LIMIT {db_parametres['dtsize']};
        """
    #--------------------------------------------------------------------
    df = con.execute(query).df()
    con.close()
    
    # Extrai os valores brutos
    valores_brutos = df['val'].to_numpy(dtype=float) - 1.0
    
    # APLICANDO SUAVIZAÇÃO (Savitzky-Golay) para reduzir ruído de alta frequência
    # window_length deve ser ímpar (ex: 11) e polyorder é o grau do polinômio (ex: 3)
    if len(valores_brutos) >= 15:
        valores_suavizados = savgol_filter(valores_brutos, window_length=11, polyorder=3)
    else:
        valores_suavizados = valores_brutos # Se houver poucos dados, mantém o bruto

    rdf = pd.DataFrame({
        'Date': df['Date'].values,
        'Observed': valores_suavizados,
        'Accumulated': np.cumsum(valores_suavizados)
    })
    return rdf

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

def PlotResults(data, Window, TrainLimit, InitIndex, FinalIndex, n_prediction, nr2, vr2, ar2, filename):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    y_global = data['Accumulated'].values
    t_global = np.arange(1, len(y_global) + 1)

    # PAINEL 1
    ax1.plot(t_global, np.diff(y_global, prepend=y_global[0]), color='lightgray', linestyle='-', linewidth=1.5, label='Dados Reais')
    ax1.axvline(x=TrainLimit, color='orange', linestyle='--', linewidth=2, label=f'Limite Treino')
    ax1.plot(t_global, np.diff(n_prediction, prepend=n_prediction[0]), color='red', linestyle='-', linewidth=1.5, label='Previsão')
    ax1.set_title(f'Série Diária | Janela: {Window} | Limite Treino: {TrainLimit}')
    ax1.set_ylabel('Variação / Preço')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # PAINEL 2
    ax2.plot(t_global, y_global, color='lightgray', linestyle='-', linewidth=1.5, label='Acumulado Real')
    ax2.axvline(x=TrainLimit, color='orange', linestyle='--', linewidth=2, label=f'Limite Treino')
    ax2.plot(t_global, n_prediction, color='royalblue', linestyle='-', linewidth=1.5, label='Projeção Richards')
    ax2.set_title(f'Dados Acumulados | R² Treino: {nr2:.3f} | R² Acumulado: {vr2:.3f}')
    ax2.set_ylabel('Valor Acumulado')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # PAINEL 3
    ax3.plot(t_global, y_global, color='lightgray', linestyle='-', linewidth=1.5, label='Série Real')
    ax3.axvline(x=TrainLimit, color='orange', linestyle='--', linewidth=2, label=f'Limite Treino')
    
    ax3_sec = ax3.twinx()
    ax3_sec.plot(t_global, np.diff(np.diff(n_prediction, prepend=n_prediction[0]), prepend=n_prediction[0]), color='green', linestyle='-.', linewidth=1.5, label='1ª Derivada')
    ax3_sec.tick_params(axis='y', labelcolor='green')
    ax3_sec.grid(False)

    ax3.set_title(f'EUR/USD - Análise de Derivada | R² Derivada: {ar2:.3f}')
    ax3.set_ylabel('Preço Acumulado')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    plt.close(fig)
    print(f'Gráfico com 3 painéis salvo com sucesso em: {filename}')