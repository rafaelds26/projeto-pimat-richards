from scipy.optimize import curve_fit
import argparse
import pandas as pd
import numpy as np
import fxutils_v2 as ut

# ==========================================
# CONFIGURAÇÃO DE CORES DA LINHA DE COMANDO
# ==========================================
VERDE = "\033[32m"
VERMELHO = "\033[31m"
AZUL = "\033[34m"
AMARELO = "\033[33m"
RESET = "\033[0m"

# ==========================================
# CONFIGURAÇÃO DE ARGUMENTOS DA LINHA DE COMANDO
# ==========================================
parser = argparse.ArgumentParser(description='Experimento Forex com Curva de Richards e Derivada.')
parser.add_argument('--ref_series', type=str, default='bid', choices=['ask', 'bid'], help="Série de referência")
parser.add_argument('--data_type', type=str, default='Close', choices=['Open', 'High', 'Low', 'Close'])
parser.add_argument('--timeframe', type=str, default='1D', choices=['Tick-By-Tick', '1M', '5M', '15M', '30M', '1H', '4H', '1D', '1W', '1MW'])
parser.add_argument('--data_size', type=int, default=365, help="Conjunto de dados utilizado")
parser.add_argument('--window_size', type=int, default=100, help="Tamanho da Janela (X)")
parser.add_argument('--train_data', type=int, default=50, help="Quantidade de dados de treino (Y)")

args = parser.parse_args()

REF_SERIES      = args.ref_series
DATA_TYPE       = args.data_type
DATA_SIZE       = args.data_size
TIMEFRAME_LABEL = args.timeframe
TRAIN_DATA      = args.train_data
WINDOW          = args.window_size

DB_PATH = 'data\\forex.duckdb'
db_parametres = {
    "Path": DB_PATH,
    "dtserie": REF_SERIES,
    "dttype": DATA_TYPE,
    "dtsize": DATA_SIZE,
    "dtframe": TIMEFRAME_LABEL,
    "dttrain": TRAIN_DATA,
    "dtwindow": WINDOW
}

print(f"{VERDE}Lendo dados de {db_parametres['dtserie'].upper()} para {db_parametres['dtframe']} ({db_parametres['dttype']})...{RESET}")

SystemDF = ut.ReadDataFromDataBase(db_parametres)

if SystemDF is None or SystemDF.empty:
    print(f"{VERMELHO}Erro: Nenhum dado retornado do banco DuckDB.{RESET}")
    exit(1)

# Seleção da janela de análise
df = SystemDF.iloc[0:WINDOW] if len(SystemDF) >= WINDOW else SystemDF

y_data = df['Accumulated'].values
x_data = np.arange(1, len(y_data) + 1, dtype=float)

StartIndex = 0
FinalIndex = TRAIN_DATA

x_train_data = x_data[StartIndex:FinalIndex+StartIndex]
y_train_data = y_data[StartIndex:FinalIndex+StartIndex]

# ==========================================
# AJUSTE DA CURVA DE RICHARDS
# ==========================================
p0 = [y_train_data[0], y_train_data[0]*1.4, 0.01, 1.0, 0.5] # A, K, B, Q, nu
bounds = ([0.0, 0.0, 0.0, 0.0, 0.0], [y_train_data[-1] * 5, y_train_data[-1] * 2, 10.0, 1000.0, 1000.0])
maxfev = 90000000

fit_success = True
try:
    poptN, _ = curve_fit(ut.Richards, x_train_data, y_train_data, p0=p0, bounds=bounds, maxfev=maxfev)
    
    # Métricas de Treino
    y_pred_train = ut.Richards(x_train_data, *poptN)
    sse_train, rmse_train, r2_train = ut.Adjustment_Metrics(y_train_data, y_pred_train)
    
    # Previsão Normal (Acumulada)
    Normal_y_predicted = ut.Richards(x_data, *poptN)
    sse_acc, rmse_acc, r2_acc = ut.Adjustment_Metrics(y_data, Normal_y_predicted)
    
    # Métricas da 1ª Derivada (Necessário para o PlotResults)
    # Nota: caso queira usar a função de derivada, certifique-se de tê-la no fxutils_v2
    # Se não tiver a função Richards_Derivative no fxutils_v2, podemos usar uma aproximação por diff
    try:
        FirstDerivative_y_predicted = ut.Richards_Derivative(x_data, *poptN)
    except AttributeError:
        # Fallback caso a derivada analítica não esteja no fxutils_v2
        FirstDerivative_y_predicted = np.gradient(Normal_y_predicted)
        
    sse_vel, rmse_vel, r2_vel = ut.Adjustment_Metrics(np.diff(y_data, prepend=y_data[0]), FirstDerivative_y_predicted)
    
except Exception as e:
    fit_success = False
    print(f"{AMARELO}Erro no ajuste da Curva de Richards na janela: {e}{RESET}")

# ==========================================
# RESULTADOS E EXIBIÇÃO DOS GRÁFICOS
# ==========================================
if fit_success:
    print(f"{AZUL}Parâmetros otimizados (A, K, B, Q, nu):{RESET}", *poptN)
    print("--------------------")
    print(f"R² Treino:{r2_train:.3f}")
    print(f"R² Normal:{r2_acc:.3f}")
    print(f"R² 1ª Derivada:{r2_vel:.3f}")

    # Chamada da função PlotResults com os 3 painéis
    try:
        ut.PlotResults(
            df, 
            WINDOW,
            TRAIN_DATA, 
            StartIndex, 
            FinalIndex, 
            Normal_y_predicted,
            r2_train, 
            r2_acc, 
            r2_vel, 
            f"Resultado_Forex_{TIMEFRAME_LABEL}_{DATA_TYPE}.png"
        )
        print(f"{VERDE}Gráfico com 3 painéis gerado com sucesso!{RESET}")
    except Exception as plot_err:
        print(f"{AMARELO}Aviso ao plotar resultados: {plot_err}{RESET}")