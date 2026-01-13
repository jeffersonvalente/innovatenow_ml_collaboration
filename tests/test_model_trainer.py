from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.model_trainer import ModelTrainer


@pytest.fixture
def sample_data():
    """
    Cria um dataset pequeno e determinístico para testes.

    Retorna:
        X_train, X_test, y_train, y_test (todos no formato esperado)
    """
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "target":   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )

    X = df[["feature1", "feature2"]]
    y = df["target"]

    # Split manual para evitar depender de train_test_split aqui
    X_train = X.iloc[:8].copy()
    y_train = y.iloc[:8].copy()
    X_test = X.iloc[8:].copy()
    y_test = y.iloc[8:].copy()

    return X_train, X_test, y_train, y_test


def test_train_runs_without_error(sample_data):
    """
    Verifica que o treino ocorre sem lançar exceções e que,
    após treinar, é possível usar o modelo para predizer.
    """
    X_train, X_test, y_train, _ = sample_data

    trainer = ModelTrainer(LogisticRegression(max_iter=1000, random_state=42))
    trainer.train(X_train, y_train)

    preds = trainer.model.predict(X_test)
    assert len(preds) == len(X_test)


def test_evaluate_returns_accuracy(sample_data):
    """
    Verifica que evaluate() retorna um float e bate com o accuracy_score
    calculado diretamente via sklearn.metrics.
    """
    X_train, X_test, y_train, y_test = sample_data

    trainer = ModelTrainer(LogisticRegression(max_iter=1000, random_state=42))
    trainer.train(X_train, y_train)

    score = trainer.evaluate(X_test, y_test)
    assert isinstance(score, float)

    direct_score = accuracy_score(y_test, trainer.model.predict(X_test))
    assert score == direct_score


def test_save_and_load_model(tmp_path: Path, sample_data):
    """
    Verifica:
    - salvar o modelo treinado em disco
    - carregar via load_model
    - tipo do modelo carregado
    - equivalência funcional: mesmas predições no mesmo input

    Observação:
    Comparar objetos sklearn por igualdade direta não é confiável.
    """
    X_train, X_test, y_train, _ = sample_data

    trainer = ModelTrainer(LogisticRegression(max_iter=1000, random_state=42))
    trainer.train(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    trainer.save_model(model_path)

    assert model_path.exists()

    loaded = ModelTrainer.load_model(model_path)
    assert isinstance(loaded, LogisticRegression)

    # Mais robusto: usa um slice do X_test do próprio fixture
    x_sample = X_test.iloc[:1]
    original_pred = trainer.model.predict(x_sample)
    loaded_pred = loaded.predict(x_sample)

    assert (original_pred == loaded_pred).all()


def test_evaluate_raises_if_not_trained(sample_data):
    """
    Avaliar sem treinar deve gerar RuntimeError.
    """
    _, X_test, _, y_test = sample_data

    trainer = ModelTrainer(LogisticRegression(max_iter=1000, random_state=42))
    with pytest.raises(RuntimeError):
        trainer.evaluate(X_test, y_test)


def test_save_raises_if_not_trained(tmp_path: Path):
    """
    Salvar sem treinar deve gerar RuntimeError.
    """
    trainer = ModelTrainer(LogisticRegression(max_iter=1000, random_state=42))
    with pytest.raises(RuntimeError):
        trainer.save_model(tmp_path / "model.joblib")


def test_train_raises_on_empty_data():
    """
    Treinar com dados vazios deve gerar ValueError.
    """
    trainer = ModelTrainer(LogisticRegression(max_iter=1000, random_state=42))

    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=int)

    with pytest.raises(ValueError):
        trainer.train(X_empty, y_empty)


def test_evaluate_raises_on_empty_data(sample_data):
    """
    Avaliar com dados vazios deve gerar ValueError (mesmo se treinado).
    """
    X_train, _, y_train, _ = sample_data

    trainer = ModelTrainer(LogisticRegression(max_iter=1000, random_state=42))
    trainer.train(X_train, y_train)

    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=int)

    with pytest.raises(ValueError):
        trainer.evaluate(X_empty, y_empty)


def test_train_type_validation():
    """
    Testes explícitos para TypeError no train():
    - X precisa ser DataFrame
    - y precisa ser Series
    """
    trainer = ModelTrainer(LogisticRegression(max_iter=1000, random_state=42))

    with pytest.raises(TypeError):
        trainer.train(np.array([[1, 2], [3, 4]]), pd.Series([0, 1]))  # X não é DataFrame

    with pytest.raises(TypeError):
        trainer.train(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"y": [0, 1]}))  # y não é Series


def test_evaluate_type_validation(sample_data):
    """
    Testes explícitos para TypeError no evaluate():
    - X_test precisa ser DataFrame
    - y_test precisa ser Series
    """
    X_train, X_test, y_train, y_test = sample_data

    trainer = ModelTrainer(LogisticRegression(max_iter=1000, random_state=42))
    trainer.train(X_train, y_train)

    with pytest.raises(TypeError):
        trainer.evaluate(X_test.values, y_test)  # numpy array ao invés de DataFrame

    with pytest.raises(TypeError):
        trainer.evaluate(X_test, y_test.to_frame())  # DataFrame ao invés de Series


def test_init_rejects_model_without_fit_predict():
    """
    Garante que o construtor rejeita objetos que não implementam fit/predict.
    """
    class BadModel:
        pass

    with pytest.raises(TypeError):
        ModelTrainer(BadModel())  # type: ignore[arg-type]


def test_load_model_raises_if_file_missing(tmp_path: Path):
    """
    Carregar um modelo que não existe deve gerar FileNotFoundError.
    """
    missing = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError):
        ModelTrainer.load_model(missing)