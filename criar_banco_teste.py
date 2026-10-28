import duckdb
import os
import pandas as pd
import numpy as np

# Garante que a pasta data existe
os.makedirs('data', exist_ok=True)
DB_PATH = 'data/forex.duckdb'

# Remove o banco de dados antigo se ele já existir para evitar conflitos
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

print("Criando o banco de dados e a tabela forex_ticks...")
con = duckdb.connect(DB_PATH)

# Gera dados fictícios para o teste
np.random.seed(42)
n_linhas = 1500
timestamps = pd.date_range(start='2023-01-01', periods=n_linhas, freq='h')
precos_base = 1.1000 + np.cumsum(np.random.normal(0, 0.001, n_linhas))

df_simulado = pd.DataFrame({
    'timestamp': timestamps,
    'pair': 'EURUSD',
    'bid': precos_base - 0.0002,
    'ask': precos_base + 0.0002
})

# Cria explicitamente a tabela forex_ticks
con.execute("CREATE TABLE forex_ticks AS SELECT * FROM df_simulado;")

# Verifica se a tabela foi realmente criada
tabelas = con.execute("SHOW TABLES;").fetchall()
print("Tabelas existentes no banco:", tabelas)

con.close()
print("Banco de dados configurado com sucesso!")