from abc import ABC, abstractmethod


class DatasetPreparation(ABC):
    def __init__(self, path, dataset_name):
        self.path = path
        self.dataset_name = dataset_name
        self.raw_data = None
        self.prepared_data = None

    @abstractmethod
    def load(self):
        """
        Read the raw_data as pd.dataframe given that is located in the field 'path' and set the field self.raw_data.
        """

    @abstractmethod
    def prepare(self):
        """
        Prepare the dataset and set the field self.prepared_data.
            For a recurrent model that works model-wise fashion:

        e.g.
            For a recurrent model that works model-wise fashion:
                prepared_data = {
                    'model1': {
                        'layers': {
                            'layer1': {
                                'features' : ['feature_1_1', 'feature_1_2', ...,'feature1_N']
                                'x': [value1, value2, ..., valueN]
                            }
                            'layer2': {
                                'features' : ['feature_2_1', 'feature_2_2', ...,'feature2_N']
                                'x': [value1, value2, ..., valueN]
                            }
                            ...
                            'layerK': {
                                'features' : ['feature_K_1', 'feature_K_2', ...,'featureK_N']
                                'x': [value1, value2, ..., valueN]
                            }
                        }
                        'y': target
                    }
                    ...
                    'modelZ': {
                        'layers': {
                            'layer1': {
                                'features' : ['feature_1_1', 'feature_1_2', ...,'feature1_N']
                                'x': [value1, value2, ..., valueN]
                            }
                            'layer2': {
                                'features' : ['feature_2_1', 'feature_2_2', ...,'feature2_N']
                                'x': [value1, value2, ..., valueN]
                            }
                            ...
                            'layerK': {
                                'features' : ['feature_K_1', 'feature_K_2', ...,'featureK_N']
                                'x': [value1, value2, ..., valueN]
                            }
                        }
                        'y': target
                    }

                }

        e.g.
            For regression model that works layer-wise fashion:
                prepared_data = {
                    'layer_type1': {
                        'features' : ['feature1_1', 'feature2_1', ... ,'featureN_1']
                        'layer_instance1': {
                            'x': [value1, value2, ... ,valueN]
                            'y': target
                        }
                        'layer_instance2': {
                            'x': [value1, value2, ... ,valueN]
                            'y': target
                        }
                        ...
                        'layer_instanceM': {
                            'x': [value1, value2, ... ,valueN]
                            'y': target
                        }
                    }
                    ...
                    'layer_typeK': {
                        'features' : ['feature1_K', 'feature2_K', ... ,'featureN_K']
                        'layer_instance1': {
                            'x': [value1, value2, ... ,valueN]
                            'y': target
                        }
                        'layer_instance2': {
                            'x': [value1, value2, ... ,valueN]
                            'y': target
                        }
                        ...
                        'layer_instanceM': {
                            'x': [value1, value2, ... ,valueN]
                            'y': target
                        }
                    }
                }
        """

    @abstractmethod
    def save(self):
        """
        Save self.prepared_data into the folder self.dataset_name.

        e.g.
            For a recurrent model that works model-wise fashion:
            ..
            |
            |__dataset_name
                |
                |__ model1
                    |
                    |_layer1.txt
                        Contains the dictionary:
                            {
                                'features' : ['feature_1_1', 'feature_1_2', ...,'feature1_N']
                                'x': [value1, value2, ..., valueN]
                            }
                    |
                    |_layer2.txt
                    |
                    |
                    ..
                    |
                    |_layerK.txt
                    |
                    |_y.txt
                ...
                |__ modelZ
                    |
                    |_layer1.txt
                    |
                    |_layer2.txt
                    |
                    |
                    ..
                    |
                    |_layerK.txt
                     |
                    |_y.txt

        """