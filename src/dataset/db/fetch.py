import pandas
import seml
import importlib
import sys
import tensorflow as tf



def get_model(model_file, model_index):
    models = importlib.import_module(model_file)
    model, name = models.get_model(model_index)
    return model


def get_data(dataset_name: str):
    results = seml.get_results(dataset_name, to_data_frame = True)
    results["model"] = results.apply(lambda x: tf.keras.models.model_from_json(x["result.model"]), axis = 1)
    return results





#print(get_data("dataset_experiment3").head())
