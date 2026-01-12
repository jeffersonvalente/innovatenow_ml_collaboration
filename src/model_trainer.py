"""
model_trainer.py

Este módulo define a classe ModelTrainer, responsável por:
- Receber um modelo do scikit-learn
- Treinar o modelo
- Avaliar o desempenho utilizando acurácia
- Persistir o modelo treinado em disco
- Carregar modelos previamente salvos

O objetivo é encapsular o ciclo de vida básico de modelos de ML
de forma segura, tipada e bem documentada.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, Union

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


class SklearnModelProtocol(Protocol):
    """
    Protocolo que define o comportamento mínimo esperado
    de um modelo do scikit-learn.

    Qualquer modelo compatível deve implementar os métodos:
    - fit
    - predict
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict(self, X: pd.DataFrame) -> Any: ...


class ModelTrainer:
    """
    Classe responsável por treinar, avaliar e persistir
    modelos de machine learning do scikit-learn.
    """

    def __init__(self, model: SklearnModelProtocol) -> None:
        """
        Inicializa o ModelTrainer com um modelo sklearn.

        Parameters
        ----------
        model : SklearnModelProtocol
            Instância de um modelo do scikit-learn (ex: LogisticRegression).

        Raises
        ------
        TypeError
            Se o modelo não implementar os métodos fit e predict.
        """
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError(
                "O modelo fornecido deve possuir os métodos 'fit' e 'predict'."
            )

        self.model: SklearnModelProtocol = model
        self._is_trained: bool = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Treina o modelo com os dados fornecidos.

        Parameters
        ----------
        X : pd.DataFrame
            Features de treinamento.
        y : pd.Series
            Target de treinamento.

        Raises
        ------
        TypeError
            Se X não for DataFrame ou y não for Series.
        ValueError
            Se X ou y estiverem vazios.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve ser um pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y deve ser um pandas Series.")
        if X.empty or y.empty:
            raise ValueError("X e y não podem estar vazios.")

        self.model.fit(X, y)
        self._is_trained = True

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Avalia o modelo treinado utilizando acurácia.

        Parameters
        ----------
        X_test : pd.DataFrame
            Features de teste.
        y_test : pd.Series
            Target de teste.

        Returns
        -------
        float
            Score de acurácia do modelo.

        Raises
        ------
        RuntimeError
            Se o modelo ainda não foi treinado.
        TypeError
            Se X_test ou y_test não forem do tipo esperado.
        ValueError
            Se os dados de teste estiverem vazios.
        """
        if not self._is_trained:
            raise RuntimeError(
                "O modelo ainda não foi treinado. Execute o método train() primeiro."
            )

        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test deve ser um pandas DataFrame.")
        if not isinstance(y_test, pd.Series):
            raise TypeError("y_test deve ser um pandas Series.")
        if X_test.empty or y_test.empty:
            raise ValueError("X_test e y_test não podem estar vazios.")

        predictions = self.model.predict(X_test)
        accuracy: float = accuracy_score(y_test, predictions)

        return accuracy

    def save_model(self, path: Union[str, os.PathLike]) -> None:
        """
        Salva o modelo treinado em disco usando joblib.

        Parameters
        ----------
        path : str | os.PathLike
            Caminho onde o modelo será salvo.

        Raises
        ------
        RuntimeError
            Se o modelo ainda não foi treinado.
        ValueError
            Se o caminho for inválido.
        """
        if not self._is_trained:
            raise RuntimeError(
                "O modelo ainda não foi treinado e não pode ser salvo."
            )

        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("O path deve ser uma string ou path-like válido.")

        directory = os.path.dirname(os.fspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

        joblib.dump(self.model, path)

    @classmethod
    def load_model(cls, path: Union[str, os.PathLike]) -> Any:
        """
        Carrega um modelo previamente salvo em disco.

        Parameters
        ----------
        path : str | os.PathLike
            Caminho do arquivo do modelo.

        Returns
        -------
        Any
            Modelo carregado.

        Raises
        ------
        FileNotFoundError
            Se o arquivo não existir.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {path}")

        return joblib.load(path)