from kedro.pipeline import Pipeline, node

from .nodes import *

def create_pipeline():
    clean_data_node = node(
        func = clean_data,
        inputs = "data",
        outputs = "clean_data",
        name = "clean_data_node"
    )

    make_transformer_node = node(
        func = make_transformer,
        inputs = "clean_data",
        outputs = "transformer",
        name = "make_transformer_node"
    )

    split_data_node = node(
        func = split_data,
        inputs = ["clean_data", "transformer"],
        outputs = {"x_train": "x_train", "y_train": "y_train", "x_test": "x_test", "y_test": "y_test", "x_train_transformed": "x_train_transformed", "x_test_transformed": "x_test_transformed"},
        name = "split_data_node"
    )

    train_linear_regression_model_node = node(
        func = train_linear_regression_model,
        inputs = ["x_train_transformed", "y_train"],
        outputs = "regressor",
        name = "train_linear_regression_model_node"
    )

    predict_node = node(
        func = predict,
        inputs = ["regressor", "x_test_transformed"],
        outputs = "y_predicted",
        name = "predict_node"
    )

    evaluate_model_node = node(
        func = evaluate_model,
        inputs = ["y_test", "y_predicted"],
        outputs = "score",
        name = "evaluate_model_node"
    )

    return Pipeline([
        clean_data_node, make_transformer_node, split_data_node, train_linear_regression_model_node, predict_node, evaluate_model_node
    ])