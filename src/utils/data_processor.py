import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import List, Tuple, Union

class DataProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty")
        self.dataframe = dataframe.copy()

    def handle_missing_values(self, strategy: str = 'mean') -> pd.DataFrame:
        if strategy not in ['mean', 'median', 'drop']:
            raise ValueError(f"Invalid strategy: {strategy}. Choose from 'mean', 'median', or 'drop'.")
        
        processed_df = self.dataframe.copy()
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns

        if strategy == 'drop':
            processed_df = processed_df.dropna()
        elif strategy == 'mean':
            processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
        elif strategy == 'median':
            processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())

        return processed_df
    
    def normalize_features(self, columns: List[str]) -> pd.DataFrame:
        
        processed_df =self.dataframe.copy()

        for col in columns:
            if col not in processed_df.columns:
                raise ValueError(f"column '{col}' not found in DataFrame")
            if not pd.api.types.is_numeric_dtype(processed_df[col]):
                raise TypeError(f"Column '{col}' is not numeric and cannot be normalized")
            
        scaler = MinMaxScaler()
        processed_df[columns] = scaler.fit_transform(processed_df[columns])
        return processed_df
    
    def encode_categorical(self, columns: List[str]) -> pd.DataFrame:
        processed_df =  self.dataframe.copy()

        for col in columns:
            if col not in processed_df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = encoder.fit_transform(processed_df[columns])

        feature_names = encoder.get_feature_names_out(columns)
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=processed_df.index)

        processed_df = processed_df.drop(columns=columns)
        processed_df = pd.concat([processed_df, encoded_df], axis=1)
        return processed_df
    
if __name__ == "__main__":
    print("Testando DataProcessor standalone")
    data = {
        'feature_num_a': [10, 20, np.nan, 30, 40],
        'feature_num_b': [1.0, 2.5, 3.0, np.nan, 5.0],
        'feature_cat_a': ['A', 'B', 'A', 'C', 'B'],
        'feature_cat_b': ['X', 'Y', 'Z', 'X', 'Y'],
        'target': [0, 1, 0, 1, 0]
    }
    
    df_example = pd.DataFrame(data)
    print("Original DataFrame:\n", df_example)

    processor = DataProcessor(df_example)

    # Teste handle_missing_values

    df_no_nan_mean = processor.handle_missing_values(strategy = 'mean')
    print("\nApós preencher NaN com média:\n", df_no_nan_mean)

    df_no_nan_drop = processor.handle_missing_values(strategy='drop')
    print("\nApós dropar linhas com NaN:\n", df_no_nan_drop)

    # Teste normalize_features (no df sem NaN para simplificar)
    df_normalized = DataProcessor(df_no_nan_mean).normalize_features(columns=['feature_num_a', 'feature_num_b'])
    print("\nApós normalizar features númericas:\n", df_normalized)

    # Teste encode_categorical (no df sem NaN)
    df_encoded = DataProcessor(df_no_nan_mean).encode_categorical(columns=['feature_cat_a', 'feature_cat_b'])
    print("\nApós One-Hot Encode categoricas:\n", df_encoded)

