from scipy.optimize import curve_fit
import argparse
import duckdb
import pandas as pd
import numpy as np
import utils as ut  # Certifique-se de que o nome do arquivo de utilitários corresponde ao import

# ==========================================
# CONFIGURAÇÃO DE ARGUMENTOS DA LINHA DE COMANDO
# ==========================================
parser = argparse.ArgumentParser(description='Experimento Forex com Curva de Richards e Derivada.')
parser.add_argument('--ref_series', type=str, default='bid', choices=['ask', 'bid'], help="Série de referência")
parser.add_argument('--data_type', type=str, default='Close', choices=['Open', 'High', 'Low', 'Close'])
parser.add_argument('--timeframe', type=str, default='1D', choices=['Tick-By-Tick', '1M', '5M', '15M', '30M', '1H', '4H', '1D', '1W', '1MW'])
parser.add_argument('--data_size', type=int, default=200, help="Tamanho total de dados extraídos")
parser.add_argument('--window_size', type=int, default=100, help="Tamanho da Janela (X)")
parser.add_argument('--train_data', type=int, default=50, help="Quantidade de dados de treino (Y)")

args = parser.parse_args()

DB_PATH         = 'data/forex.duckdb'
REF_SERIES      = args.ref_series
DATA_TYPE       = args.data_type
DATA_SIZE       = args.data_size
TIMEFRAME_LABEL = args.timeframe
WINDOW          = args.window_size
TRAIN_DATA      = args.train_data

tf_mapping = {
    'Tick-By-Tick': None, '1M': '1 minute', '5M': '5 minutes', '15M': '15 minutes',
    '30M': '30 minutes', '1H': '1 hour', '4H': '4 hours', '1D': '1 day', '1W': '1 week', '1MW': '1 month'
}
interval_sql = tf_mapping.get(TIMEFRAME_LABEL, '1 day')

# ==========================================
# CONEXÃO E CONSULTA AO DUCKDB (Dados Reais)
# ==========================================
con = duckdb.connect(DB_PATH)
print(f"Extraindo dados reais de {REF_SERIES.upper()} para {TIMEFRAME_LABEL} ({DATA_TYPE})...")

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
df_forex = con.execute(query).df()
con.close()

if df_forex.empty:
    print("Erro: Nenhum dado retornado do banco DuckDB.")
    exit(1)

# ==========================================
# ESTRUTURAÇÃO DOS DADOS
# ==========================================
FileDF = pd.DataFrame()
FileDF['Data'] = df_forex['Date']
FileDF['Diarios'] = df_forex['val']
# Convertendo para acumulados para manter compatibilidade com o modelo de Richards
FileDF['Acumulados'] = np.cumsum(df_forex['val'].values)

y_data = FileDF['Acumulados'].values
x_data = np.arange(1, len(y_data) + 1, dtype=float)

StartIndex = 0
FinalIndex = TRAIN_DATA

x_train_data = x_data[StartIndex:FinalIndex+StartIndex]
y_train_data = y_data[StartIndex:FinalIndex+StartIndex]

# Parâmetros iniciais e limites para o ajuste de Richards
p0 = [y_train_data[0], y_train_data[0]*1.4, 0.01, 1.0, 0.5] 
bounds = ([0.0, 0.0, 0.0, 0.0, 0.0], [y_train_data[-1] * 5, y_train_data[-1] * 2, 10.0, 1000.0, 1000.0])
maxfev = 90000000

fit_success = True
try:
    poptN, _ = curve_fit(ut.Richards, x_train_data, y_train_data, p0=p0, bounds=bounds, maxfev=maxfev)
    
    y_pred_train = ut.Richards(x_train_data, *poptN)
    sse_train, rmse_train, r2_train = ut.Adjustment_Metrics(y_train_data, y_pred_train)
    
    Normal_y_predicted = ut.Richards(x_data, *poptN)
    sse_acc, rmse_acc, r2_acc = ut.Adjustment_Metrics(y_data, Normal_y_predicted)
    
    FirstDervative_y_predicted = ut.Richards_Derivative(x_data, *poptN)
    sse_vel, rmse_vel, r2_vel = ut.Adjustment_Metrics(np.diff(y_data, prepend=y_data[0]), FirstDervative_y_predicted)
    
except Exception as e:
    fit_success = False
    print(f"Erro no ajuste da Curva de Richards: {e}")

if fit_success:
    print("Parâmetros otimizados:", *poptN)
    print("--------------------")
    print(f"R² Treino:{r2_train:.3f}")
    print(f"R² Normal:{r2_acc:.3f}")
    print(f"R² 1ª Derivada:{r2_vel:.3f}")

    # ==========================================
    # GERAÇÃO DO GRÁFICO COM OS 3 PAINÉIS
    # ==========================================
    ut.PlotResults(
        FileDF.iloc[:WINDOW], 
        WINDOW,
        TRAIN_DATA, 
        0, 
        FinalIndex, 
        Normal_y_predicted[:WINDOW],
        r2_train, 
        r2_acc, 
        r2_vel, 
        f"Resultado_Forex_{TIMEFRAME_LABEL}_{DATA_TYPE}.png",
        TIMEFRAME_LABEL,
        REF_SERIES
    )