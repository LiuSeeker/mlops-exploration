from kedro.pipeline import Pipeline, node

from .nodes import concat_dataframes, split_data, make_transformer, clean_data

def create_pipeline():

    concat_dataframes_node = node(
        func = concat_dataframes,
        inputs = ["northeast", "southeast"],
        outputs = "concat_df",
        name = "concat_dataframes_node"
    )

    clean_data_node = node(
        func = clean_data,
        inputs = "concat_df",
        outputs = "clean_df",
        name = "clean_data_node"
    )

    make_transformer_node = node(
        func = make_transformer,
        inputs = "clean_df",
        outputs = "transformer",
        name = "make_transformer_node"
    )

    split_data_node = node(
        func = split_data,
        inputs = ["clean_df", "transformer"],
        outputs = {"x_train": "x_train", "y_train": "y_train", "x_test": "x_test", "y_test": "y_test", "x_train_transformed": "x_train_transformed", "x_test_transformed": "x_test_transformed"},
        name = "split_data_node"
    )


    return Pipeline(
        [
            concat_dataframes_node, clean_data_node, make_transformer_node, split_data_node
        ]
    )