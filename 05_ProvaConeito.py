import argparse
import duckdb
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotting_utils as pu 

# ==========================================
# CONFIGURAÇÃO DE ARGUMENTOS DA LINHA DE COMANDO
# ==========================================
parser = argparse.ArgumentParser(description='Experimento Forex com Curva de Richards e Derivada.')
parser.add_argument('--ref_series', type=str, default='bid', choices=['ask', 'bid'], help="Série de referência")
parser.add_argument('--data_type', type=str, default='Close', choices=['Open', 'High', 'Low', 'Close'])
parser.add_argument('--timeframe', type=str, default='1D', choices=['Tick-By-Tick', '1M', '5M', '15M', '30M', '1H', '4H', '1D', '1W', '1MW'])
parser.add_argument('--data_size', type=int, default=365, help="Conjunto de dados utilizado")
parser.add_argument('--window_size', type=int, default=36, help="Tamanho da Janela (X)")
parser.add_argument('--know_data', type=int, default=15, help="Quantidade de dados de treino (Y)")

args = parser.parse_args()

DB_PATH = 'data/forex.duckdb'
REF_SERIES      = args.ref_series
DATA_TYPE       = args.data_type
DATA_SIZE       = args.data_size
TIMEFRAME_LABEL = args.timeframe
WINDOW_SIZE_X   = args.window_size
KNOW_DATA_Y     = args.know_data

if KNOW_DATA_Y >= WINDOW_SIZE_X:
    print("Erro: A quantidade de dados de treino deve ser menor que o tamanho da janela.")
    exit(1)

tf_mapping = {
    'Tick-By-Tick': None, '1M': '1 minute', '5M': '5 minutes', '15M': '15 minutes',
    '30M': '30 minutes', '1H': '1 hour', '4H': '4 hours', '1D': '1 day', '1W': '1 week', '1MW': '1 month'
}
interval_sql = tf_mapping.get(TIMEFRAME_LABEL, '1 day')

# ==========================================
# CONEXÃO E CONSULTA AO DUCKDB
# ==========================================
con = duckdb.connect(DB_PATH)
print(f"Extraindo dados de {REF_SERIES.upper()} para {TIMEFRAME_LABEL} ({DATA_TYPE})...")

if TIMEFRAME_LABEL == 'Tick-By-Tick':
    query = f"SELECT timestamp AS Date, {REF_SERIES} AS val FROM forex_ticks WHERE pair = 'EURUSD' ORDER BY timestamp ASC LIMIT {DATA_SIZE};"
else:
    query = f"""
    WITH ohlc AS (
        SELECT time_bucket(INTERVAL '{interval_sql}', timestamp) AS Date,
            FIRST({REF_SERIES}) AS Open, MAX({REF_SERIES}) AS High,
            MIN({REF_SERIES}) AS Low, LAST({REF_SERIES}) AS Close
        FROM forex_ticks WHERE pair = 'EURUSD' GROUP BY Date ORDER BY Date ASC
    )
    SELECT Date, {DATA_TYPE} AS val FROM ohlc LIMIT {DATA_SIZE};
    """
df = con.execute(query).df()
con.close()

if df.empty or len(df) < KNOW_DATA_Y:
    print("Aviso: Dados insuficientes para a janela ou treino definidos.")
    exit()

# ==========================================
# FUNÇÕES DE REGRESSÃO
# ==========================================
def richards_curve(t, A, K, B, Q, nu):
    exp_term = np.exp(-B * np.clip(t, -50, 50))
    return A + (K - A) / np.power(1.0 + Q * exp_term, 1.0 / np.maximum(nu, 1e-5))

def richards_derivative(t, A, K, B, Q, nu):
    clipped_t = np.clip(t, -50, 50)
    exp_term = np.exp(-B * clipped_t)
    base = 1.0 + Q * exp_term
    inv_nu = 1.0 / np.maximum(nu, 1e-5)
    numerator = (K - A) * Q * exp_term * B * np.power(base, -inv_nu - 1.0)
    denominator = np.maximum(nu, 1e-5)
    return numerator / denominator

# ==========================================
# INICIO PROCESSAMENTO
# ==========================================
Dados_Disponíveis  = len(df)
Janela_Treinamento = KNOW_DATA_Y
Janela_de_Dados    = WINDOW_SIZE_X
Janelas_Analisadas = Dados_Disponíveis - Janela_de_Dados + 1

print(f"Dados Disponíveis.................:{Dados_Disponíveis}")
print(f"Tamanho da Janela de Dados........:{Janela_de_Dados}")
print(f"Tamanho da Janela de Treinamento..:{Janela_Treinamento}")
print(f"Quantidade Janelas Analisadas.....:{Janelas_Analisadas}")

df['t'] = np.arange(len(df))
y_data = df['val'].values
t_data = df['t'].values

metrics_history      = []
N_RichardsParametres = [] 
D_RichardsParametres = []

startIndice = 0

for analysis in range(Janelas_Analisadas):
    data_x_train = t_data[startIndice : startIndice + KNOW_DATA_Y]
    data_y_train = y_data[startIndice : startIndice + KNOW_DATA_Y]
    
    p0 = [data_y_train.min(), data_y_train.max() * 1.05, 0.1, 1.0, 1.0]
    
    try:
        poptN, _ = curve_fit(richards_curve, data_x_train, data_y_train, p0=p0, maxfev=1000000)
        poptD, _ = curve_fit(richards_derivative, data_x_train, data_y_train, p0=p0, maxfev=1000000)
        
        N_RichardsParametres.append(poptN)
        D_RichardsParametres.append(poptD)
        fit_success = True
        
        # Calcular predição no treino para métricas
        y_pred_train = richards_curve(data_x_train, *poptN)
        sse_train, rmse_train, r2_train = pu.calcular_metricas_ajuste(data_y_train, y_pred_train)
        
    except Exception as e:
        print(f"Erro no ajuste da Curva de Richards na janela {analysis}: {e}")
        poptN, poptD = np.zeros(5), np.zeros(5)
        N_RichardsParametres.append(poptN)
        D_RichardsParametres.append(poptD)
        fit_success = False
        sse_train, rmse_train, r2_train = np.nan, np.nan, np.nan
        
    # Guardar histórico para o gráfico resumo
    metrics_history.append({
        'SSE_Train': sse_train,
        'RMSE_Train': rmse_train,
        'R2_Train': r2_train
    })
    
    # Chamada da função externa de plotagem
    file_name = f'Analisys_{TIMEFRAME_LABEL}_{DATA_TYPE}_{WINDOW_SIZE_X}_{KNOW_DATA_Y}_Window_{analysis}.png'
    pu.plotar_janela_forex(df_completo=df,
                        start_idx=startIndice,
                        window_size=WINDOW_SIZE_X,
                        know_data=KNOW_DATA_Y,
                        poptN=poptN,
                        poptD=poptD,
                        fit_success=fit_success,
                        file_name=file_name,
                        TIMEFRAME_LABEL=TIMEFRAME_LABEL,
                        DATA_TYPE=DATA_TYPE,
                        REF_SERIES=REF_SERIES,
                        analysis_idx=analysis
                    )
    
    startIndice += 1

print("Processamento de janelas e geração de gráficos concluídos com sucesso!")
# ==========================================
# GERAÇÃO DO GRÁFICO RESUMO GERAL AO FINAL
# ==========================================
print("\nGerando gráfico consolidado de parâmetros e métricas...")
pu.plotar_resumo_parametros_metricas(N_RichardsParametres, metrics_history, TIMEFRAME_LABEL, DATA_TYPE)
# ==========================================
# Comparativo de parâmetros
# ==========================================
print("\nGerando gráfico resumo comparativo de parâmetros...")
pu.plotar_resumo_parametros_comparados(
    N_RichardsParametres=N_RichardsParametres,
    D_RichardsParametres=D_RichardsParametres,
    metrics_history=metrics_history,
    TIMEFRAME_LABEL=TIMEFRAME_LABEL,
    DATA_TYPE=DATA_TYPE
)