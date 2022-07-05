from codecarbon import EmissionsTracker
import numpy as np
import tensorflow as tf

def evaluate_energy_forward(model, input_size, batch_size =1, repetitions = 1, on_GPU = True):
    """ Measures the energy consumption of a forward pass of the given model using uniformly random input 

    Parameters:
    -----------
        model:
            ML model that should be measured (has to be callable on input)
        input_size:
            size of the input to the model as int tuple, can also be a list of int tuples if the model expects list of input tensors
        batch_size: int
            size of a batch for the forward pass (default 1)
        repetitions: int
            number of repetitions for the measurement
        on_GPU: bool
            bool indicating wheter the forward pass should be performed on GPU or CPU
        
    Returns:
    ----------
        float: measured average energy consumption of forward pass for the model
    """
    tf.debugging.set_log_device_placement(True)

    #Create random input:
    def measurement():
        if isinstance(input_size, list):
            input = []
            for s in input_size:
                input.append(tf.random.uniform((batch_size, *s[1:])))
        else:
            input = tf.random.uniform((batch_size, *input_size[1:]))
        
        #print(input.get_device()) #TODO: DELETE
        #Run repetitions forward passes and meassure the energy consumption
        tracker = EmissionsTracker(save_to_file= False)
        tracker.start()
        for i in range(repetitions):
            output = model(input)
        
        energy_consumption = tracker.stop()/repetitions
        return energy_consumption
    
    if on_GPU:
        with tf.device('/GPU:0'):
            return measurement()
    else:
        with tf.device('/CPU:0'):
            return measurement()



