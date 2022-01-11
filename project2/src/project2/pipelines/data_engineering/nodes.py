import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import mlflow


def concat_dataframes(northeast, southeast):
    concat_df = pd.DataFrame()
    dfs = [northeast, southeast]
    

    for df in dfs:
        concat_df = pd.concat([concat_df, df.sample(15000, random_state=0)])
    
    return concat_df

def clean_data(data):
    clean_data = data.drop(data[data[data.columns.values[3]] == -9999.0].index)

    return clean_data

def make_transformer(data):
    mlflow.log_artifact("data/02_intermediate/clean_df.csv", "logs/artifacts")
    features_data = data.drop(data.columns.values[3], 1)

    cat_columns = features_data.select_dtypes(include="object").columns
    num_columns = features_data.select_dtypes(exclude="object").columns

    transformer = ColumnTransformer([
        ("numerical", StandardScaler(), num_columns),
        ("categorical", OneHotEncoder(), cat_columns),
    ])

    transformer.fit(features_data)

    return transformer

def split_data(data, transformer):
    y = data[data.columns.values[3]]
    x = data.drop([data.columns.values[3]], axis=1)
    region = data["region"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=region)

    x_train_transformed = transformer.transform(x_train)
    x_test_transformed = transformer.transform(x_test)

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test, "x_train_transformed": x_train_transformed, "x_test_transformed": x_test_transformed}


