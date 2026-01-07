# main.py
import pandas as pd
import sklearn
# Adicione estes imports
from src.utils.data_splitter import DataSplitter
from data_preprocessing import load_data, preprocess_data # Já deve estar aqui

print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("--- Versão da Main: Iniciando Execução Principal ---") # Manter essa linha da Tarefa 2

# Criar um DataFrame de exemplo ou usar o dummy_data.csv
df_raw = load_data() # Usar a função de carregamento existente

# Integrar o DataSplitter
print("\n--- Utilizando DataSplitter ---")
try:
    splitter = DataSplitter(df_raw)
    train_df, test_df = splitter.split(test_size=0.25, random_state=42)
    print(f"Dados originais: {df_raw.shape[0]} amostras")
    print(f"Dados de treino: {train_df.shape[0]} amostras")
    print(f"Dados de teste: {test_df.shape[0]} amostras")
except (TypeError, ValueError) as e:
    print(f"Erro ao usar DataSplitter: {e}")


print("\n--- Módulo de Pré-processamento Integrado ---")
data_frame = load_data()
processed_data_frame = preprocess_data(data_frame)
print("Processamento integrado concluído.")


if __name__ == "__main__":
    print("Ambiente configurado com sucesso!")