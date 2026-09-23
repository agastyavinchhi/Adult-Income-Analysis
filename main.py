import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

SHOW_PLOT = True
DATA_PATH = "adult.csv"

def load_data(path = DATA_PATH):
    return pd.read_csv(path)

def data_preprocess(df):
    # Missing values are stored as "?" in this dataset. Replace them, then drop those rows and any duplicates.
    df = df.replace("?", pd.NA)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    # fnlwgt is the census sampling weight, not a property of the person, so it is not a useful feature
    df = df.drop(columns=["fnlwgt"])
    return df

def income_share_by_education(df):
    return (df["income"] == ">50K").groupby(df["education"]).mean().sort_values()


def plot_income_by_education(df):
    # Plot: share earning >50K by education level
    income_share_by_education(df).plot(kind="barh", figsize=(8, 6))
    plt.xlabel("Share earning >50K")
    plt.title("Income by Education Level")
    plt.show()


def train_model(df):
    # GradientBoostingClassifier to predict high income. One-hot encode categorical variables.
    X = pd.get_dummies(df.drop(columns=["income"]))
    y = (df["income"] == ">50K").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


def main():
    df = data_preprocess(load_data(DATA_PATH))
    if SHOW_PLOT:
        plot_income_by_education(df)
    _, accuracy = train_model(df)
    print("Accuracy:", round(accuracy, 4))


if __name__ == "__main__":
    main()





