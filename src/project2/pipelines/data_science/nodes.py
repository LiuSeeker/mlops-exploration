import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mlflow
from mlflow import sklearn
from datetime import datetime

def train_model(x_train_transformed, y_train):
    regressor = LinearRegression()
    regressor.fit(x_train_transformed, y_train)
    sklearn.log_model(regressor, "logs/models")
    return regressor

def predict(regressor, x_test_transformed):
    y_pred = regressor.predict(x_test_transformed)
    return pd.DataFrame(y_pred)

def predict_zero(x_test):
    y_pred = [0]*len(x_test)
    return pd.DataFrame(y_pred)

def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mlflow.log_param("Time of prediction", datetime.now())
    mlflow.log_metric("R2 score", r2)
    return f"r2 score: {r2}"

def evaluate_model_zero(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mlflow.log_param("Time of prediction zero", datetime.now())
    mlflow.log_metric("R2 score zero", r2)
    return f"r2 score: {r2}"