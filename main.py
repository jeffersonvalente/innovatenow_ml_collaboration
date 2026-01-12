import os
import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression

from src.model_trainer import ModelTrainer
from src.utils.data_splitter import DataSplitter
from src.utils.data_processor import DataProcessor
from data_preprocessing import load_data, preprocess_data


print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("--- Versão da Main: Iniciando Execução Principal ---")


# ======================================================
# 1. Carregamento / Criação do DataFrame de exemplo
# ======================================================
print("\n--- Carregando dados ---")
df_raw = load_data()
print(f"Dataset carregado com shape: {df_raw.shape}")


# ======================================================
# 2. Pré-processamento dos dados com DataProcessor
# ======================================================
print("\n--- Pré-processando dados com DataProcessor ---")

processor = DataProcessor(df_raw)

# Exemplo de pipeline de pré-processamento
df_processed = processor.handle_missing_values(strategy="mean")

# Normalizar colunas numéricas (exemplo)
numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove("target")

df_processed = DataProcessor(df_processed).normalize_features(columns=numerical_cols)

# Codificar colunas categóricas
categorical_cols = df_processed.select_dtypes(include=["object"]).columns.tolist()
if categorical_cols:
    df_processed = DataProcessor(df_processed).encode_categorical(columns=categorical_cols)

print("Pré-processamento concluído.")
print(df_processed.head())


# ======================================================
# 3. Separação em treino e teste com DataSplitter
# ======================================================
print("\n--- Dividindo dados com DataSplitter ---")

splitter = DataSplitter(df_processed)
train_df, test_df = splitter.split(test_size=0.25, random_state=42)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")


# ======================================================
# 4. Treinamento do modelo com ModelTrainer
# ======================================================
print("\n--- Treinando modelo LogisticRegression ---")

log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
trainer = ModelTrainer(log_reg_model)

trainer.train(X_train, y_train)


# ======================================================
# 5. Avaliação do modelo
# ======================================================
print("\n--- Avaliando modelo ---")
accuracy = trainer.evaluate(X_test, y_test)
print(f"Acurácia do modelo: {accuracy:.4f}")


# ======================================================
# 6. Salvando o modelo treinado
# ======================================================
print("\n--- Salvando modelo ---")

model_dir = "models"
model_path = os.path.join(model_dir, "logistic_regression_model.joblib")

trainer.save_model(model_path)
print(f"Modelo salvo em: {model_path}")


# ======================================================
# 7. Carregando o modelo salvo
# ======================================================
print("\n--- Carregando modelo salvo ---")
loaded_model = ModelTrainer.load_model(model_path)

print(f"Modelo carregado: {type(loaded_model)}")


# ======================================================
# 8. Predição com o modelo carregado
# ======================================================
print("\n--- Fazendo predição com modelo carregado ---")

sample_X = X_test.iloc[:3]
predictions = loaded_model.predict(sample_X)

print("Amostras usadas para predição:")
print(sample_X)

print("Predições:")
print(predictions)


# ======================================================
# Execução principal
# ======================================================
if __name__ == "__main__":
    print("\nPipeline de Machine Learning executado com sucesso ✅")