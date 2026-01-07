# tests/test_data_splitter.py
import pytest
import pandas as pd
from src.utils.data_splitter import DataSplitter # Caminho importante!

@pytest.fixture
def sample_dataframe():
    # DataFrame de exemplo com um número conhecido de linhas para testes
    return pd.DataFrame({
        'col1': range(100),
        'col2': [f'val{i%2}' for i in range(100)]
    })

def test_split_returns_dataframes(sample_dataframe):
    splitter = DataSplitter(sample_dataframe)
    train_df, test_df = splitter.split()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

def test_split_size_and_sum(sample_dataframe):
    splitter = DataSplitter(sample_dataframe)
    test_size = 0.2
    train_df, test_df = splitter.split(test_size=test_size)

    assert len(train_df) + len(test_df) == len(sample_dataframe)
    # Permita uma pequena variação devido ao arredondamento na divisão
    expected_test_len = int(len(sample_dataframe) * test_size)
    assert expected_test_len - 1 <= len(test_df) <= expected_test_len + 1

def test_random_state_reproducibility(sample_dataframe):
    splitter1 = DataSplitter(sample_dataframe)
    train1, test1 = splitter1.split(random_state=42)

    splitter2 = DataSplitter(sample_dataframe)
    train2, test2 = splitter2.split(random_state=42)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)

def test_empty_dataframe_raises_error():
    with pytest.raises(ValueError, match="DataFrame cannot be empty."):
        DataSplitter(pd.DataFrame())

def test_invalid_test_size_raises_error(sample_dataframe):
    splitter = DataSplitter(sample_dataframe)
    with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
        splitter.split(test_size=0.0)
    with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
        splitter.split(test_size=1.0)
    with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
        splitter.split(test_size=1.1)

def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame."):
        DataSplitter([1, 2, 3])