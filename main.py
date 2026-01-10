# main.py
import pandas as pd
import sklearn
# Adicione estes imports
import numpy as np
from src.utils.data_splitter import DataSplitter
from src.utils.data_processor import DataProcessor
from data_preprocessing import load_data, preprocess_data

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

# Criar um DataFrame de exemplo com NaN e categóricos para o processador
raw_data_for_processing = {
    'numerical_feat_1': [10, 20, np.nan, 30, 40, 50],
    'numerical_feat_2': [1.0, 2.5, 3.0, np.nan, 5.0, 6.5],
    'categorical_feat_A': ['Red', 'Blue', 'Red', 'Green', 'Blue', 'Red'],
    'categorical_feat_B': ['X', 'Y', 'Z', 'X', 'Y', np.nan],
    'target': [0, 1, 0, 1, 0, 1]
}
df_raw_processor = pd.DataFrame(raw_data_for_processing)


print("\n--- Utilizando DataProcessor ---")
try:
    processor = DataProcessor(df_raw_processor)

    # 1. Tratando valores ausentes
    df_processed_missing = processor.handle_missing_values(strategy='mean')
    print("\nApós tratativa de missing values (mean):\n", df_processed_missing.head())

    # 2. Normalizando features
    df_normalized = DataProcessor(df_processed_missing).normalize_features(columns=['numerical_feat_1', 'numerical_feat_2'])
    print("\nApós normalização:\n", df_normalized.head())

    # 3. Codificando categóricas (use o df_processed_missing antes de normalizar se preferir, ou normalize primeiro e depois encodifique)
    df_final = DataProcessor(df_normalized).encode_categorical(columns=['categorical_feat_A', 'categorical_feat_B'])
    print("\nApós One-Hot Encoding:\n", df_final.head())

except (TypeError, ValueError) as e:
    print(f"Erro ao usar DataProcessor: {e}")

print("\n--- Módulo de Pré-processamento Integrado Da Tarefa 2 ---")
data_frame_task2 = load_data()
# Não vamos preprocessar aqui para focar no DataProcessor, mas se quisesse, seria:
# processed_data_frame_task2 = preprocess_data(data_frame_task2)
print("Módulos da Tarefa 2 ainda presente.")


if __name__ == "__main__":
    print("Ambiente configurado com sucesso!")