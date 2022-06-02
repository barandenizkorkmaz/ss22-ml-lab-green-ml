from codecarbon import EmissionsTracker
import numpy as np

def evaluate_energy_forward(model, input_size, batch_size =1, repetitions = 1):
    """ Measures the energy consumption of a forward pass of the given model

    Parameters:
    -----------
        model:
            ML model that should be measured (has to be callable on input)
        input_size:
            size of the input to the model as int tuple
        batch_size: int
            size of a batch for the forward pass (default 1)
        repetitions: int
            number of repetitions for the measurement
        
    Returns:
    ----------
        float: measured average energy consumption of forward pass for the model
    """
    #Create random input:
    input = np.random.rand(batch_size, *input_size[1:])
    
    #Run repetitions forward passes and meassure the energy consumption
    tracker = EmissionsTracker(save_to_file= False)
    tracker.start()
    for i in range(repetitions):
        output = model(input)
    
    energy_consumption = tracker.stop()/repetitions
    return energy_consumption
    


