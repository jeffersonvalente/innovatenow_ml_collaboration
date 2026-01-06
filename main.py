import pandas as pd
import sklearn

print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

if __name__ == "__main__":
    print("Ambiente configurado com sucesso!")
    from data_preprocessing import load_data, preprocess_data
    print("\n--- Módulo de Pré-processamento Integrado ---")
    data_frame = load_data()
    processed_data_frame = preprocess_data(data_frame)
    print("Processamento integrado concluído.")