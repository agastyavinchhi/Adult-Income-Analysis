import pandas as pd
import matplotlib.pyplot as plt

from main import load_data, data_preprocess, train_model, plot_income_by_education, DATA_PATH

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
def test_train_model():
    df_test = data_preprocess(load_data(DATA_PATH))
    # Select a small subsection of 200 samples for testing
    df_test = df_test.sample(200, random_state=1)
    # Testing to ensure ML pipeline works
    model, accuracy = train_model(df_test)
    assert model is not None
    # Ensure that ML model has provided a valid accuracy output 
    assert 0.0 <= accuracy <= 1.0

# System Test
def test_full_pipeline_runs():
    df = data_preprocess(load_data(DATA_PATH))
    model, accuracy = train_model(df)
    assert model is not None
    assert accuracy > 0.5