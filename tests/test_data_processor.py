import pytest
import pandas as pd
import numpy as np

from src.utils.data_processor import DataProcessor


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    # Tem NaN em numéricas + NaN em categórica (para validar 'drop')
    return pd.DataFrame(
        {
            "num_a": [10.0, 20.0, np.nan, 40.0],
            "num_b": [1.0, np.nan, 3.0, 4.0],
            "cat_a": ["A", "B", "A", "C"],
            "cat_b": ["X", "Y", np.nan, "X"],
        }
    )


@pytest.fixture
def df_clean() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "num_b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat_a": ["A", "B", "A", "C", "B"],
            "cat_b": ["X", "Y", "Z", "X", "Y"],
        }
    )


# ---------- Construtor / validações básicas ----------

def test_init_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        DataProcessor([1, 2, 3])  # type: ignore[arg-type]


def test_init_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        DataProcessor(pd.DataFrame())


def test_init_copies_dataframe(df_clean: pd.DataFrame) -> None:
    original = df_clean
    processor = DataProcessor(original)

    # modificar original não pode mudar o interno
    original.loc[0, "num_a"] = 999.0
    assert processor.dataframe.loc[0, "num_a"] != 999.0


# ---------- handle_missing_values ----------

def test_handle_missing_values_invalid_strategy_raises(df_with_missing: pd.DataFrame) -> None:
    processor = DataProcessor(df_with_missing)
    with pytest.raises(ValueError, match="Invalid strategy"):
        processor.handle_missing_values(strategy="mode")  # type: ignore[arg-type]


def test_handle_missing_values_mean_fills_numeric_nans(df_with_missing: pd.DataFrame) -> None:
    processor = DataProcessor(df_with_missing)
    original_copy = df_with_missing.copy(deep=True)

    result = processor.handle_missing_values(strategy="mean")

    # não altera o original do processor
    pd.testing.assert_frame_equal(processor.dataframe, original_copy)

    # numéricas sem NaN
    assert result["num_a"].isna().sum() == 0
    assert result["num_b"].isna().sum() == 0

    # valida valores preenchidos
    expected_mean_num_a = original_copy["num_a"].mean()  # mean ignora NaN
    expected_mean_num_b = original_copy["num_b"].mean()

    assert result.loc[2, "num_a"] == expected_mean_num_a
    assert result.loc[1, "num_b"] == expected_mean_num_b

    # categóricas continuam com NaN (DOD: só numéricas)
    assert pd.isna(result.loc[2, "cat_b"])


def test_handle_missing_values_median_fills_numeric_nans(df_with_missing: pd.DataFrame) -> None:
    processor = DataProcessor(df_with_missing)
    original_copy = df_with_missing.copy(deep=True)

    result = processor.handle_missing_values(strategy="median")

    assert result["num_a"].isna().sum() == 0
    assert result["num_b"].isna().sum() == 0

    expected_median_num_a = original_copy["num_a"].median()
    expected_median_num_b = original_copy["num_b"].median()

    assert result.loc[2, "num_a"] == expected_median_num_a
    assert result.loc[1, "num_b"] == expected_median_num_b


def test_handle_missing_values_drop_removes_any_nan_rows(df_with_missing: pd.DataFrame) -> None:
    processor = DataProcessor(df_with_missing)
    result = processor.handle_missing_values(strategy="drop")

    # no resultado não pode existir nenhum NaN
    assert result.isna().sum().sum() == 0

    # neste dataset, apenas a linha 0 não tem NaN em nenhuma coluna
    assert set(result.index) == {0, 3}


# ---------- normalize_features ----------

def test_normalize_features_missing_column_raises(df_clean: pd.DataFrame) -> None:
    processor = DataProcessor(df_clean)
    with pytest.raises(ValueError, match="not found in DataFrame"):
        processor.normalize_features(columns=["does_not_exist"])


def test_normalize_features_non_numeric_raises(df_clean: pd.DataFrame) -> None:
    processor = DataProcessor(df_clean)
    with pytest.raises(TypeError, match="is not numeric and cannot be normalized"):
        processor.normalize_features(columns=["cat_a"])


def test_normalize_features_range_and_consistency(df_clean: pd.DataFrame) -> None:
    processor = DataProcessor(df_clean)
    original_copy = df_clean.copy(deep=True)

    result = processor.normalize_features(columns=["num_a"])

    # não altera o original do processor
    pd.testing.assert_frame_equal(processor.dataframe, original_copy)

    # range [0,1]
    assert result["num_a"].min() == 0.0
    assert result["num_a"].max() == 1.0

    # consistência esperada para [10,20,30,40,50] -> [0,0.25,0.5,0.75,1]
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(result["num_a"].to_numpy(), expected, rtol=0, atol=1e-9)


def test_normalize_features_multiple_columns(df_clean: pd.DataFrame) -> None:
    processor = DataProcessor(df_clean)
    result = processor.normalize_features(columns=["num_a", "num_b"])

    assert result["num_a"].between(0.0, 1.0).all()
    assert result["num_b"].between(0.0, 1.0).all()


# ---------- encode_categorical ----------

def test_encode_categorical_missing_column_raises(df_clean: pd.DataFrame) -> None:
    processor = DataProcessor(df_clean)
    with pytest.raises(ValueError, match="not found in DataFrame"):
        processor.encode_categorical(columns=["nope"])


def test_encode_categorical_removes_original_and_adds_one_hot(df_clean: pd.DataFrame) -> None:
    processor = DataProcessor(df_clean)
    original_copy = df_clean.copy(deep=True)

    result = processor.encode_categorical(columns=["cat_a"])

    # não altera o original do processor
    pd.testing.assert_frame_equal(processor.dataframe, original_copy)

    # remove original
    assert "cat_a" not in result.columns

    # adiciona one-hot esperadas (A,B,C)
    assert "cat_a_A" in result.columns
    assert "cat_a_B" in result.columns
    assert "cat_a_C" in result.columns

    # número de colunas: remove 1, adiciona 3 -> +2
    assert result.shape[1] == df_clean.shape[1] + 2


def test_encode_categorical_multiple_columns_counts(df_clean: pd.DataFrame) -> None:
    processor = DataProcessor(df_clean)
    result = processor.encode_categorical(columns=["cat_a", "cat_b"])

    # remove originais
    assert "cat_a" not in result.columns
    assert "cat_b" not in result.columns

    # cat_a: {A,B,C} -> 3 colunas; cat_b: {X,Y,Z} -> 3 colunas
    # total novo = original - 2 + 6 = original + 4
    assert result.shape[1] == df_clean.shape[1] + 4


def test_encode_categorical_values_correctness(df_clean: pd.DataFrame) -> None:
    processor = DataProcessor(df_clean)
    result = processor.encode_categorical(columns=["cat_a"])

    # linha 0 tem cat_a = 'A'
    assert result.loc[0, "cat_a_A"] == 1.0
    assert result.loc[0, "cat_a_B"] == 0.0
    assert result.loc[0, "cat_a_C"] == 0.0