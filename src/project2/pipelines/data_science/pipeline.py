from kedro.pipeline import Pipeline, node

from .nodes import train_model, predict, evaluate_model, predict_zero, evaluate_model_zero

def create_pipeline():

    train_model_node = node(
        func = train_model,
        inputs = ["x_train_transformed", "y_train"],
        outputs = "regressor",
        name = "train_model_node"
    )

    predict_node = node(
        func = predict,
        inputs = ["regressor", "x_test_transformed"],
        outputs = "y_predicted",
        name = "predict_node"
    )

    predict_zero_node = node(
        func = predict_zero,
        inputs = "x_test",
        outputs = "y_predicted_zero",
        name = "predict_zero_node"
    )

    evaluate_model_node = node(
        func = evaluate_model,
        inputs = ["y_test", "y_predicted"],
        outputs = "score",
        name = "evaluate_model_node"
    )

    evaluate_zero_node = node(
        func = evaluate_model_zero,
        inputs = ["y_test", "y_predicted_zero"],
        outputs = "score_zero",
        name = "evaluate_zero_node"
    )


    return Pipeline(
        [
            train_model_node, predict_node, evaluate_model_node, predict_zero_node, evaluate_zero_node
        ]
    )

