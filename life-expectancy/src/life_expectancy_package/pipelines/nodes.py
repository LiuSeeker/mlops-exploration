import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import mlflow

def clean_data(data):
    return data.dropna()

def make_transformer(data):
    features_data = data.drop("Life expectancy ", 1)

    cat_columns = features_data.select_dtypes(include="object").columns
    num_columns = features_data.select_dtypes(exclude="object").columns

    transformer = ColumnTransformer([
        ("numerical", StandardScaler(), num_columns),
        ("categorical", OneHotEncoder(), cat_columns),
    ])

    transformer.fit(features_data)

    return transformer


def split_data(data, transformer):
    y = data["Life expectancy "]
    x = data.drop(["Life expectancy "], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    x_train_transformed = transformer.transform(x_train)
    x_test_transformed = transformer.transform(x_test)

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test, "x_train_transformed": x_train_transformed, "x_test_transformed": x_test_transformed}

def train_linear_regression_model(x_train_transformed, y_train):
    regressor = LinearRegression()
    regressor.fit(x_train_transformed, y_train)
    return regressor

def predict(regressor, x_test_transformed):
    y_pred = regressor.predict(x_test_transformed)
    return pd.DataFrame(y_pred)

def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    return f"r2 score: {r2}\n"