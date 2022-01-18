import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# import mlflow

def clean_data():
    data = pd.read_csv("../../data/01_raw/life-expectancy.csv")
    clean_data = data.dropna()
    clean_data.to_csv("../../data/02_intermediate/clean-data.csv")
    return clean_data

def make_transformer():
    data = pd.read_csv("../../data/02_intermediate/clean-data.csv")
    features_data = data.drop("Life expectancy ", 1)

    cat_columns = features_data.select_dtypes(include="object").columns
    num_columns = features_data.select_dtypes(exclude="object").columns

    transformer = ColumnTransformer([
        ("numerical", StandardScaler(), num_columns),
        ("categorical", OneHotEncoder(), cat_columns),
    ])

    transformer.fit(features_data)

    pickle.dump(transformer, "../../data/03_primary/transformer.pkl")

    return transformer

def split_data():
    data = pd.read_csv("../../data/02_intermediate/clean-data.csv")
    transformer = pickle.load("../../data/03_primary/transformer.pkl")

    y = data["Life expectancy "]
    x = data.drop(["Life expectancy "], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    x_train_transformed = transformer.transform(x_train)
    x_test_transformed = transformer.transform(x_test)

    d = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test, "x_train_transformed": x_train_transformed, "x_test_transformed": x_test_transformed}

    x_train.to_csv("../../data/04_features/x_train.csv")
    y_train.to_csv("../../data/04_features/y_train.csv")
    x_test.to_csv("../../data/04_features/x_test.csv")
    y_test.to_csv("../../data/04_features/y_test.csv")

    pickle.dump(x_train_transformed, "../../data/05_model_input/x_train_transformed.pkl")
    pickle.dump(x_test_transformed, "../../data/05_model_input/x_test_transformed.pkl")

    return d

def train_linear_regression_model():
    x_train_transformed = pickle.load("../../data/05_model_input/x_train_transformed.pkl")
    y_train = pd.read_csv("../../data/04_features/y_train.csv")
    regressor = LinearRegression()
    regressor.fit(x_train_transformed, y_train)
    pickle.dump(regressor, "../../data/06_models/regressor.pkl")
    return regressor

def predict():
    regressor = pickle.load("../../data/06_models/regressor.pkl")
    x_test_transformed = pickle.load("../../data/05_model_input/x_test_transformed.pkl")
    y_pred = regressor.predict(x_test_transformed)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv("../../data/07_model_output/y_predicted.csv")
    return y_pred_df

def evaluate_model():
    y_test = pd.read_csv("../../data/04_features/y_test.csv")
    y_pred = pd.read_csv("../../data/07_model_output/y_predicted.csv")
    r2 = r2_score(y_test, y_pred)
    with open("../../data/08_reporting/score.txt", "w") as f:
        f.write(f"r2 score: {r2}\n")
    return f"r2 score: {r2}\n"