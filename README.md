# Green ML - DAML

## Team Associates

In alphabetical order:

* Baran Deniz Korkmaz
* Barış Tura
* Niklas Kemper

Supervised by Bertrand Charpentier M.Sc.



## Introduction

This repository contains the project called **Green Machine Learning** for the practical course IN2106 - Large Scale Machine Learning offered by the Data Analytics and Machine Learning Group at Technical University of Munich.



## Program Interface

The project has been written in Python using the machine learning framework TensorFlow. To run the program, the user must ensure that the required packages have been installed. After navigating the **src** folder, the program can be executed by the following command:

```bash
python main.py
```



## Program Execution

The program can be executed via command-line interpreter. The user must only configure the **yaml** file which will be used for the program setup. The details about the yaml file will be provided in the **Program Input** section.



## Program Input

The program does not take any explicit command line argument. However, the user must set the path of the yaml file, which will be used in order to configure the program setup, in the main script manually. This section describes the details about how to configure the yaml file. One example can be found in the repository in the src folder as **run_config_layerwise.yaml**. 

The keys that you want to manually change are:

1. dataset_class: The name of the key of the dataset that you want to use.
2. model_class: The name of the key of the model that you want to use.

Each key represents either a dataset or a model. Each key has the following fields regardless of if it belongs to a dataset or a model:

1. module: The path to the module containing the class.
2. class: The name of the class.
3. params: The parameter set used in the corresponding class.

Finally for the evaluation, you need to specify the module that contain the metric functions and the metrics that you want to use in the evaluation. **Please keep them as fixed.**



## Program Output

The program outputs a csv file named **daml-green-ml-results.csv** which contains the training parameters and the results for the model used.
