# data_preprocessing.py
import pandas as pd

def load_data(filepath='dummy_data.csv'):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Criando {filepath} para teste...")
        df = pd.DataFrame({
            'feature1': [10, 20, 15, 25, 30],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(filepath, index=False)
        print("Dados dummy criados.")
        df = pd.read_csv(filepath)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Pré-processando dados...")
    df['feature1_scaled'] = (df['feature1'] - df['feature1'].mean()) / df['feature1'].std()
    print("Pré-processamento concluído.")
    return df

if __name__ == "__main__":
    print("Executando módulo de pré-processamento Standalone.")
    df = load_data()
    processed_df = preprocess_data(df)
    print(processed_df.head())