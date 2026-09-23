import pandas as pd
import pytest
from main import load_data, data_preprocess, train_model, income_share_by_education, DATA_PATH

# Unit Test 1
def test_load_data():
    # Ensure the data has sucessfuly been loaded without any errors
    df = load_data(DATA_PATH)
    assert not df.empty

# Unit Test 2
def test_data_preprocess():
    # Testing with raw data
    test_input = pd.DataFrame({
        "age": [25, 25, 40],
        "workclass": ["Private", "Private", "?"],
        "fnlwgt": [1, 1, 2],
        "education": ["Bachelors", "Bachelors", "High-School"],
        "income": ["<=50K", "<=50K", ">50K"],
    })
    cleaned = data_preprocess(test_input)

    # Testing to ensure .drop() columns have been sucessfuly removed
    assert "fnlwgt" not in cleaned.columns
    # Testing to see that .replace() NA values has worked
    assert not (cleaned == "?").values.any()

# Unit Test 3
def test_income_share_by_education():
    df = pd.DataFrame({
        "education": ["Bachelors", "Bachelors", "HS-grad", "HS-grad", "HS-grad", "Masters"],
        "income": [">50K", "<=50K", "<=50K", "<=50K", ">50K", ">50K"],
    })
    result = income_share_by_education(df)

    assert result["Bachelors"] == pytest.approx(0.5)
    assert result["HS-grad"] == pytest.approx(1 / 3)
    assert result["Masters"] == pytest.approx(1.0)

    # sort_values() should leave it in ascending order
    assert list(result) == sorted(result)

# System Test
def test_full_pipeline_runs():
    df = data_preprocess(load_data(DATA_PATH))
    model, accuracy = train_model(df)
    # Testing to ensure ML pipeline works
    assert model is not None
    # Ensure that ML model has provided a valid accuracy output 
    assert 0 < accuracy < 1.0
