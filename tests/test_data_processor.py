# tests/test_data_processor.py
import pytest
import pandas as pd
import numpy as np
from src.utils.data_processor import DataProcessor

@pytest.fixture
def sample_dataframe():
    """DataFrame de exemplo com valores numéricos, categóricos e NaN"""
    return pd.DataFrame({
        'num_col1': [10.0, 20.0, np.nan, 40.0, 50.0],
        'num_col2': [1.0, 2.0, 3.0, np.nan, 5.0],
        'cat_col1': ['A', 'B', 'A', 'C', 'B'],
        'cat_col2': ['X', 'Y', 'Z', 'X', 'Y'],
        'target': [0, 1, 0, 1, 0]
    })

@pytest.fixture
def clean_dataframe():
    """DataFrame sem valores ausentes para testes de normalização e encoding"""
    return pd.DataFrame({
        'num_col1': [10.0, 20.0, 30.0, 40.0, 50.0],
        'num_col2': [1.0, 2.0, 3.0, 4.0, 5.0],
        'cat_col1': ['A', 'B', 'A', 'C', 'B'],
        'cat_col2': ['X', 'Y', 'Z', 'X', 'Y']
    })

# ========== Testes do Construtor ==========

def test_constructor_accepts_dataframe(sample_dataframe):
    processor = DataProcessor(sample_dataframe)
    assert isinstance(processor.dataframe, pd.DataFrame)

def test_empty_dataframe_raises_error():
    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        DataProcessor(pd.DataFrame())

def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        DataProcessor([1, 2, 3])

# ========== Testes handle_missing_values ==========

def test_handle_missing_values_mean_strategy(sample_dataframe):
    processor = DataProcessor(sample_dataframe)
    result = processor.handle_missing_values(strategy='mean')
    
    # Verifica que não há mais NaN nas colunas numéricas
    assert result['num_col1'].isna().sum() == 0
    assert result['num_col2'].isna().sum() == 0
    
    # Verifica que a média foi aplicada corretamente
    expected_mean_col1 = (10.0 + 20.0 + 40.0 + 50.0) / 4  # 30.0
    assert result['num_col1'].iloc[2] == expected_mean_col1
    
    expected_mean_col2 = (1.0 + 2.0 + 3.0 + 5.0) / 4  # 2.75
    assert result['num_col2'].iloc[3] == expected_mean_col2

def test_handle_missing_values_median_strategy(sample_dataframe):
    processor = DataProcessor(sample_dataframe)
    result = processor.handle_missing_values(strategy='median')
    
    # Verifica que não há mais NaN
    assert result['num_col1'].isna().sum() == 0
    assert result['num_col2'].isna().sum() == 0
    
    # Verifica que a mediana foi aplicada
    expected_median_col1 = 30.0  # mediana de [10, 20, 40, 50]
    assert result['num_col1'].iloc[2] == expected_median_col1

def test_handle_missing_values_drop_strategy(sample_dataframe):
    processor = DataProcessor(sample_dataframe)
    result = processor.handle_missing_values(strategy='drop')
    
    # Verifica que linhas com NaN foram removidas
    assert len(result) == 3  # Apenas 3 linhas sem NaN
    assert result.isna().sum().sum() == 0  # Nenhum NaN restante

def test_handle_missing_values_invalid_strategy(sample_dataframe):
    processor = DataProcessor(sample_dataframe)
    with pytest.raises(ValueError, match="Invalid strategy"):
        processor.handle_missing_values(strategy='invalid')

def test_handle_missing_values_returns_new_dataframe(sample_dataframe):
    processor = DataProcessor(sample_dataframe)
    result = processor.handle_missing_values(strategy='mean')
    
    # Verifica que o DataFrame original não foi modificado
    assert processor.dataframe['num_col1'].isna().sum() > 0
    assert result['num_col1'].isna().sum() == 0

# ========== Testes normalize_features ==========

def test_normalize_features_range(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    result = processor.normalize_features(columns=['num_col1', 'num_col2'])
    
    # Verifica que os valores estão no range [0, 1]
    assert result['num_col1'].min() == 0.0
    assert result['num_col1'].max() == 1.0
    assert result['num_col2'].min() == 0.0
    assert result['num_col2'].max() == 1.0

def test_normalize_features_consistency(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    result = processor.normalize_features(columns=['num_col1'])
    
    # Verifica valores específicos da normalização
    # num_col1 = [10, 20, 30, 40, 50] -> normalizado = [0.0, 0.25, 0.5, 0.75, 1.0]
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    np.testing.assert_array_almost_equal(result['num_col1'].values, expected)

def test_normalize_features_column_not_found(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    with pytest.raises(ValueError, match="not found in DataFrame"):
        processor.normalize_features(columns=['nonexistent_col'])

def test_normalize_features_non_numeric_column(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    with pytest.raises(TypeError, match="is not numeric and cannot be normalized"):
        processor.normalize_features(columns=['cat_col1'])

def test_normalize_features_returns_new_dataframe(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    result = processor.normalize_features(columns=['num_col1'])
    
    # Verifica que o DataFrame original não foi modificado
    assert processor.dataframe['num_col1'].iloc[0] == 10.0
    assert result['num_col1'].iloc[0] == 0.0

# ========== Testes encode_categorical ==========

def test_encode_categorical_creates_new_columns(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    result = processor.encode_categorical(columns=['cat_col1'])
    
    # Verifica que novas colunas one-hot foram criadas
    assert 'cat_col1_A' in result.columns
    assert 'cat_col1_B' in result.columns
    assert 'cat_col1_C' in result.columns

def test_encode_categorical_removes_original_columns(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    result = processor.encode_categorical(columns=['cat_col1'])
    
    # Verifica que a coluna original foi removida
    assert 'cat_col1' not in result.columns

def test_encode_categorical_correct_number_of_columns(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    original_cols = len(clean_dataframe.columns)
    result = processor.encode_categorical(columns=['cat_col1'])
    
    # cat_col1 tem 3 valores únicos (A, B, C)
    # Colunas esperadas: original - 1 (removida) + 3 (one-hot) = original + 2
    assert len(result.columns) == original_cols + 2

def test_encode_categorical_multiple_columns(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    result = processor.encode_categorical(columns=['cat_col1', 'cat_col2'])
    
    # Verifica que ambas foram codificadas
    assert 'cat_col1_A' in result.columns
    assert 'cat_col2_X' in result.columns
    assert 'cat_col1' not in result.columns
    assert 'cat_col2' not in result.columns

def test_encode_categorical_values_correctness(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    result = processor.encode_categorical(columns=['cat_col1'])
    
    # Primeira linha tem 'A', então cat_col1_A deve ser 1.0
    assert result['cat_col1_A'].iloc[0] == 1.0
    assert result['cat_col1_B'].iloc[0] == 0.0
    assert result['cat_col1_C'].iloc[0] == 0.0

def test_encode_categorical_column_not_found(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    with pytest.raises(ValueError, match="not found in DataFrame"):
        processor.encode_categorical(columns=['nonexistent_col'])

def test_encode_categorical_returns_new_dataframe(clean_dataframe):
    processor = DataProcessor(clean_dataframe)
    result = processor.encode_categorical(columns=['cat_col1'])
    
    # Verifica que o DataFrame original não foi modificado
    assert 'cat_col1' in processor.dataframe.columns
    assert 'cat_col1' not in result.columns