# plotting_utils.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calcular_metricas_ajuste(y_true, y_pred, num_params=5):
    """
    Calcula SSE, RMSE e R² entre os dados reais e a predição.
    """
    # Remover eventuais NaNs ou infinitos para evitar erros
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    # SSE (Soma dos Erros Quadráticos)
    sse = np.sum((y_true - y_pred) ** 2)
    
    # RMSE (Raiz do Erro Quadrático Médio)
    rmse = np.sqrt(sse / len(y_true))
    
    # R² (Coeficiente de Determinação)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_total == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - (sse / ss_total)
        
    return sse, rmse, r2


def plotar_resumo_parametros_metricas(N_RichardsParametres, metrics_history, TIMEFRAME_LABEL, DATA_TYPE):
    """
    Gera um painel com múltiplos subplots mostrando a evolução 
    dos 5 parâmetros da Curva de Richards e das métricas de qualidade (R², RMSE, SSE).
    """
    params_array = np.array(N_RichardsParametres) # Formato: (num_janelas, 5)
    janelas_ids = np.arange(len(params_array))
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    param_names = ['Parâmetro A (Assím.)', 'Parâmetro K (Cap. Suporte)', 'Parâmetro B (Taxa Cresc.)', 
                   'Parâmetro Q (Desloc. Y)', 'Parâmetro ν (Nu - Forma)']
    
    # 1. Plotar os 5 parâmetros da Curva Normal (N_RichardsParametres)
    for i in range(5):
        axes[i].plot(janelas_ids, params_array[:, i], marker='o', linestyle='-', color='royalblue')
        axes[i].set_title(param_names[i])
        axes[i].set_xlabel('Índice da Janela')
        axes[i].set_ylabel('Valor do Parâmetro')
        axes[i].grid(True, alpha=0.3)
        
    # Extrair métricas do histórico
    r2_list = [m['R2_Train'] for m in metrics_history]
    rmse_list = [m['RMSE_Train'] for m in metrics_history]
    sse_list = [m['SSE_Train'] for m in metrics_history]
    
    # 2. Plotar R²
    axes[5].plot(janelas_ids, r2_list, marker='s', linestyle='-', color='forestgreen')
    axes[5].set_title('Coeficiente de Determinação (R² - Treino)')
    axes[5].set_xlabel('Índice da Janela')
    axes[5].set_ylabel('R²')
    axes[5].grid(True, alpha=0.3)
    
    # 3. Plotar RMSE
    axes[6].plot(janelas_ids, rmse_list, marker='^', linestyle='-', color='crimson')
    axes[6].set_title('RMSE (Treino)')
    axes[6].set_xlabel('Índice da Janela')
    axes[6].set_ylabel('RMSE')
    axes[6].grid(True, alpha=0.3)
    
    # 4. Plotar SSE (com escala logarítmica caso haja muita variação)
    axes[7].plot(janelas_ids, sse_list, marker='d', linestyle='-', color='darkorange')
    axes[7].set_title('SSE - Sum of Squared Errors (Treino)')
    axes[7].set_xlabel('Índice da Janela')
    axes[7].set_ylabel('SSE')
    axes[7].set_yscale('log') # Escala log é útil para SSE devido a discrepâncias
    axes[7].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_summary_file = f'Resumo_Parametros_E_MetricasList_{TIMEFRAME_LABEL}_{DATA_TYPE}.png'
    plt.savefig(output_summary_file, dpi=150)
    plt.close(fig)
    print(f"Gráfico resumo de parâmetros e métricas salvo com sucesso em: {output_summary_file}")


def plotar_janela_forex(df_completo, start_idx, window_size, know_data, 
                          poptN, poptD, fit_success, file_name, 
                          TIMEFRAME_LABEL, DATA_TYPE, REF_SERIES, analysis_idx):
    """
    Plota a série temporal global completa e projeta a Curva de Richards 
    e sua derivada sobre TODO o histórico de dados disponível.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Extração de dados globais
    t_global = df_completo['t'].values
    y_global = df_completo['val'].values
    date_global = df_completo['Date']
    
    # Limites da janela e do treino
    end_window_idx = start_idx + window_size
    train_limit_idx = start_idx + know_data - 1
    train_date_limit = date_global.iloc[train_limit_idx]
    
    # -----------------------------------------
    # PAINEL 1: Série Global + Janela Investigada
    # -----------------------------------------
    ax1.plot(date_global, y_global, color='lightgray', linestyle='-', linewidth=1.5, label='Série Histórica Completa')
    
    # Destacar a janela ativa
    sub_date = date_global.iloc[start_idx:end_window_idx]
    sub_y = y_global[start_idx:end_window_idx]
    ax1.plot(sub_date, sub_y, color='royalblue', marker='o', markersize=4, linestyle='-', label=f'Janela Ativa (Tam: {window_size})')
    
    ax1.axvline(train_date_limit, color='orange', linestyle='--', linewidth=2, label=f'Limite Treino (Y = {know_data})')
    
    ax1.set_title(f'EUR/USD - TimeFrame: {TIMEFRAME_LABEL} | Série: {REF_SERIES.upper()} | Janela #{analysis_idx}')
    ax1.set_ylabel(f'Preço ({DATA_TYPE})')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # -----------------------------------------
    # PAINEL 2: Projeção para TODO O HISTÓRICO
    # -----------------------------------------
    ax2.plot(date_global, y_global, color='lightgray', linestyle='-', linewidth=1.5, label='Série Histórica Real')
    
    if fit_success:
        # IMPORTANTE: Usamos o eixo t global normalizado relativo à janela 
        # (ou o t_global completo) para prever em toda a série.
        # Aqui subtraímos t_global[start_idx] para manter o mesmo referencial temporal do treino (t=0 no início da janela).
        t_relative = t_global - t_global[start_idx]
        
        y_pred_total = richards_curve(t_relative, *poptN)
        dy_pred_total = richards_derivative(t_relative, *poptD)
        
        ax2.plot(date_global, y_pred_total, color='crimson', linewidth=2, label='Projeção Richards (Série Toda)')
        
        # Eixo Y secundário para a derivada ao longo de todo o histórico
        ax2_sec = ax2.twinx()
        ax2_sec.plot(date_global, dy_pred_total, color='purple', linewidth=1.5, linestyle='-.', label='1ª Derivada Global')
        ax2_sec.set_ylabel('Taxa de Variação (dy/dt)', color='purple')
        ax2_sec.tick_params(axis='y', labelcolor='purple')
        ax2_sec.grid(False)
        
        # Combinar legendas
        lines_1, labels_1 = ax2.get_legend_handles_labels()
        lines_2, labels_2 = ax2_sec.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    ax2.axvline(train_date_limit, color='orange', linestyle='--', linewidth=2, label='Corte Treino/Teste')
    ax2.set_xlabel('Data / Timestamp')
    ax2.set_ylabel('Preço Projetado')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    plt.close(fig)
    print(f'Gráfico global gerado e salvo em: {file_name}')




def plotar_resumo_parametros_comparados(N_RichardsParametres, D_RichardsParametres, metrics_history, TIMEFRAME_LABEL, DATA_TYPE):
    """
    Gera um painel 5x2 comparando lado a lado os parâmetros estimados 
    pela Curva de Richards Normal (Esquerda) e pela 1ª Derivada (Direita).
    """
    params_n = np.array(N_RichardsParametres) # Formato: (num_janelas, 5)
    params_d = np.array(D_RichardsParametres) # Formato: (num_janelas, 5)
    janelas_ids = np.arange(len(params_n))
    
    # 5 linhas (para os 5 parâmetros) e 2 colunas (Normal vs Derivada)
    fig, axes = plt.subplots(5, 2, figsize=(14, 16), sharex=True)
    
    param_names = [
        'Parâmetro A (Assím.)', 
        'Parâmetro K (Cap. Suporte)', 
        'Parâmetro B (Taxa Cresc.)', 
        'Parâmetro Q (Desloc. Y)', 
        'Parâmetro ν (Nu - Forma)'
    ]
    
    for i in range(5):
        # Coluna da Esquerda: Curva Normal
        axes[i, 0].plot(janelas_ids, params_n[:, i], marker='o', linestyle='-', color='royalblue')
        axes[i, 0].set_title(f'Normal - {param_names[i]}')
        axes[i, 0].set_ylabel('Valor')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Coluna da Direita: 1ª Derivada
        axes[i, 1].plot(janelas_ids, params_d[:, i], marker='s', linestyle='-', color='darkviolet')
        axes[i, 1].set_title(f'Derivada - {param_names[i]}')
        axes[i, 1].set_ylabel('Valor')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Adicionar rótulo do eixo X apenas na última linha
        if i == 4:
            axes[i, 0].set_xlabel('Índice da Janela')
            axes[i, 1].set_xlabel('Índice da Janela')

    plt.tight_layout()
    output_summary_file = f'Resumo_Comparativo_Parametros_{TIMEFRAME_LABEL}_{DATA_TYPE}.png'
    plt.savefig(output_summary_file, dpi=150)
    plt.close(fig)
    print(f"Gráfico resumo comparativo salvo com sucesso em: {output_summary_file}")

# Funções de suporte
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