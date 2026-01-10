import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import List


class DataProcessor:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty.")
        # Sempre trabalhar em cópia para evitar side effects
        self.dataframe: pd.DataFrame = dataframe.copy()

    def handle_missing_values(self, strategy: str = "mean") -> pd.DataFrame:
        """
        Trata valores ausentes em colunas numéricas.
        - mean: preenche NaN numéricos com média da coluna
        - median: preenche NaN numéricos com mediana da coluna
        - drop: remove linhas com qualquer NaN
        Retorna um novo DataFrame (não altera o original).
        """
        if strategy not in {"mean", "median", "drop"}:
            raise ValueError(
                f"Invalid strategy: {strategy}. Choose from 'mean', 'median', or 'drop'."
            )

        processed_df = self.dataframe.copy()

        if strategy == "drop":
            return processed_df.dropna()

        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns available to fill missing values.")

        if strategy == "mean":
            fill_values = processed_df[numeric_cols].mean()
        else:  # strategy == "median"
            fill_values = processed_df[numeric_cols].median()

        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(fill_values)
        return processed_df

    def normalize_features(self, columns: List[str]) -> pd.DataFrame:
        """
        Aplica MinMaxScaler nas colunas numéricas especificadas.
        Retorna um novo DataFrame (não altera o original).
        """
        if not isinstance(columns, list) or len(columns) == 0:
            raise ValueError("columns must be a non-empty list of column names.")

        processed_df = self.dataframe.copy()

        for col in columns:
            if col not in processed_df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(processed_df[col]):
                raise TypeError(f"Column '{col}' is not numeric and cannot be normalized.")

        scaler = MinMaxScaler()
        processed_df[columns] = scaler.fit_transform(processed_df[columns])
        return processed_df

    def encode_categorical(self, columns: List[str]) -> pd.DataFrame:
        """
        Aplica OneHotEncoder nas colunas especificadas (categóricas),
        com handle_unknown='ignore'. Remove as colunas originais e adiciona as one-hot.
        Retorna um novo DataFrame (não altera o original).
        """
        if not isinstance(columns, list) or len(columns) == 0:
            raise ValueError("columns must be a non-empty list of column names.")

        processed_df = self.dataframe.copy()

        for col in columns:
            if col not in processed_df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Compatibilidade entre versões do sklearn:
        # - versões mais novas: sparse_output
        # - versões antigas: sparse
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        encoded_data = encoder.fit_transform(processed_df[columns])
        feature_names = encoder.get_feature_names_out(columns)

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=feature_names,
            index=processed_df.index,
        )

        processed_df = processed_df.drop(columns=columns)
        processed_df = pd.concat([processed_df, encoded_df], axis=1)
        return processed_df