from numpy import number
from sacred import Experiment
from utils import energyEvaluation
import logging
import importlib
import seml

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(model_index: int, model_file: str, batch_size: int, number_forward_passes: int):
    logging.info("Received following config:")
    logging.info(f"Model: {model_index}, batch_size: {batch_size}, number_forward_passes: {number_forward_passes}")

    #Not sure if this is a good solution
    models = importlib.import_module(model_file)
    model, name = models.get_model(model_index)

    logging.info(f"Evaluate energy for model {name}")
    power = energyEvaluation.evaluate_energy_forward(model, model.input_shape, batch_size, number_forward_passes)
    logging.info(f"Measured power consumption of {power}kWh")
    result = {"name": name, "power": power}
    return result

print(run(7, "models.dense", 10, 10, False))
    

