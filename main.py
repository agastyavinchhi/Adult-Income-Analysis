from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = "adult.csv"
PLOT_PATH = "images/income_by_education.png"
TARGET = "income"
HIGH_INCOME = ">50K"


def load_data(path=DATA_PATH):
    """Read the raw census CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df):
    """Drop rows with missing values, duplicate rows, and the fnlwgt column."""
    # Missing values are stored as "?" in this dataset
    df = df.replace("?", pd.NA)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    # fnlwgt is a census sampling weight, not a property of the person
    return df.drop(columns=["fnlwgt"])


def is_high_income(df):
    """True for each row where the person earns more than $50K."""
    return df[TARGET] == HIGH_INCOME


def income_share_by_education(df):
    """Share of people earning >50K at each education level, lowest first."""
    return is_high_income(df).groupby(df["education"]).mean().sort_values()


def save_income_by_education_plot(df, path=PLOT_PATH):
    """Save a bar chart of the >50K share by education level to path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    income_share_by_education(df).plot(kind="barh", ax=ax)
    ax.set_xlabel("Share earning >50K")
    ax.set_title("Income by Education Level")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_features(df):
    """Split into one-hot encoded features X and a 0/1 target y."""
    X = pd.get_dummies(df.drop(columns=[TARGET]))
    y = is_high_income(df).astype(int)
    return X, y


def train_model(df):
    """Train on an 80/20 split; return the model and its test accuracy."""
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


def main():
    df = clean_data(load_data())
    save_income_by_education_plot(df)
    _, accuracy = train_model(df)
    print("Accuracy:", round(accuracy, 4))
    print("Plot saved to", PLOT_PATH)


if __name__ == "__main__":
    main()
