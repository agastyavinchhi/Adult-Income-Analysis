import pandas as pd
import pytest
from main import (
    load_data,
    clean_data,
    train_model,
    income_share_by_education,
    build_features,
    save_income_by_education_plot,
    DATA_PATH,
)


# Unit Test 1
def test_load_data():
    # Ensure the data has sucessfuly been loaded without any errors
    df = load_data(DATA_PATH)
    assert not df.empty


# Unit Test 2
def test_data_preprocess():
    # Testing with raw data
    test_input = pd.DataFrame(
        {
            "age": [25, 25, 40],
            "workclass": ["Private", "Private", "?"],
            "fnlwgt": [1, 1, 2],
            "education": ["Bachelors", "Bachelors", "High-School"],
            "income": ["<=50K", "<=50K", ">50K"],
        }
    )
    cleaned = clean_data(test_input)

    # Testing to ensure .drop() columns have been sucessfuly removed
    assert "fnlwgt" not in cleaned.columns
    # Testing to see that .replace() NA values has worked
    assert not (cleaned == "?").values.any()
    # Duplicate row collapses to one, and the "?" row is dropped entirely
    assert len(cleaned) == 1


# Unit Test 3
def test_income_share_by_education():
    df = pd.DataFrame(
        {
            "education": [
                "Bachelors",
                "Bachelors",
                "HS-grad",
                "HS-grad",
                "HS-grad",
                "Masters",
            ],
            "income": [">50K", "<=50K", "<=50K", "<=50K", ">50K", ">50K"],
        }
    )
    result = income_share_by_education(df)

    assert result["Bachelors"] == pytest.approx(0.5)
    assert result["HS-grad"] == pytest.approx(1 / 3)
    assert result["Masters"] == pytest.approx(1.0)

    # sort_values() should leave it in ascending order
    assert list(result) == sorted(result)


# System Test
def test_full_pipeline_runs():
    df = clean_data(load_data(DATA_PATH))
    model, accuracy = train_model(df)
    # Testing to ensure ML pipeline works
    assert model is not None
    # Ensure that ML model has provided a valid accuracy output
    assert 0 < accuracy < 1.0


# Edge Case: missing file should fail loudly, not return an empty frame
def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("does_not_exist.csv")


# Edge Case: if every row has a missing value, nothing should survive cleaning
def test_data_preprocess_all_rows_missing():
    test_input = pd.DataFrame(
        {
            "age": [30, 45],
            "workclass": ["?", "?"],
            "fnlwgt": [1, 2],
            "income": ["<=50K", ">50K"],
        }
    )
    cleaned = clean_data(test_input)
    assert cleaned.empty


# Edge Case: preprocessing should not modify the caller's original DataFrame
def test_data_preprocess_does_not_mutate_input():
    test_input = pd.DataFrame(
        {
            "age": [25],
            "workclass": ["?"],
            "fnlwgt": [1],
            "income": ["<=50K"],
        }
    )
    original = test_input.copy()
    clean_data(test_input)
    pd.testing.assert_frame_equal(test_input, original)


# Edge Case: an education level where nobody earns >50K should give 0.0, not be dropped
def test_income_share_by_education_zero_share():
    df = pd.DataFrame(
        {
            "education": ["Preschool", "Preschool", "Doctorate"],
            "income": ["<=50K", "<=50K", ">50K"],
        }
    )
    result = income_share_by_education(df)
    assert result["Preschool"] == 0.0
    assert result.index[0] == "Preschool"  # lowest share sorts first


# Unit Test: features and target are split correctly
def test_build_features():
    df = pd.DataFrame(
        {
            "age": [30, 50],
            "sex": ["Male", "Female"],
            "income": ["<=50K", ">50K"],
        }
    )
    X, y = build_features(df)

    # The target must never leak into the features
    assert "income" not in X.columns
    # Text categories are one-hot encoded into separate columns
    assert {"sex_Male", "sex_Female"} <= set(X.columns)
    assert list(y) == [0, 1]


# Edge Case: plot is saved even when the output folder does not exist yet
def test_save_plot_creates_folder(tmp_path):
    df = pd.DataFrame(
        {
            "education": ["Bachelors", "HS-grad"],
            "income": [">50K", "<=50K"],
        }
    )
    out = tmp_path / "new_folder" / "plot.png"
    save_income_by_education_plot(df, out)
    assert out.exists()
