# src/utils/data_splitter.py
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataSplitter:
    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty.")
        self.dataframe = dataframe.copy() # Garante que o original não seja modificado

    def split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be between 0.0 and 1.0 (exclusive).")
        
        train_df, test_df = train_test_split(
            self.dataframe,
            test_size=test_size,
            random_state=random_state,
            shuffle=True # Boas práticas de ML
        )
        return train_df, test_df

if __name__ == "__main__":
    print("Executando módulo DataSplitter standalone para teste...")
    # Exemplo de criação de DataFrame dummy
    data = {
        'feature1': range(100),
        'feature2': [f'cat{i%3}' for i in range(100)],
        'target': [i%2 for i in range(100)]
    }
    df_example = pd.DataFrame(data)

    splitter = DataSplitter(df_example)
    train, test = splitter.split(test_size=0.3, random_state=1)

    print(f"Original DataFrame shape: {df_example.shape}")
    print(f"Train DataFrame shape: {train.shape}")
    print(f"Test DataFrame shape: {test.shape}")

    # Teste de erro
    try:
        DataSplitter(pd.DataFrame()).split()
    except ValueError as e:
        print(f"Erro esperado capturado: {e}")