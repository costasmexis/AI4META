import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from dataloader import DataLoader

class MachineLearningModel(DataLoader):
    def __init__(self, model, param_grid, label, csv_dir):
        ''' Class to hold the machine learning model and related data 
            Inherits from DataLoader class
            - model (sklearn model): model to be used
            - param_grid (dict): hyperparameters to be tuned
            - label (str): name of the target column
            - csv_dir (str): path to the csv file
        '''
        super().__init__(label, csv_dir)
        self.model = model
        self.name = model.__class__.__name__
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None
        self.best_estimator = None

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        # Make predictions using the trained model
        pass

    def evaluate(self, X, y):
        # Evaluate the performance of the model
        pass

class MLPipelines(MachineLearningModel):
    def __init__(self, model, param_grid, label, csv_dir):
        ''' Class to perform machine learning pipelines 
            Inherits from MachineLearningModel class
            - model (sklearn model): model to be used
            - param_grid (dict): hyperparameters to be tuned
            - label (str): name of the target column
            - csv_dir (str): path to the csv file
        '''
        super().__init__(model, param_grid, label, csv_dir)

    def cross_validation(self, X, y, scoring='accuracy', cv=5):
        ''' Function to perform a simple cross validation
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
        '''
        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        print(f'Average {scoring}: {np.mean(scores)}')
        print(f'Standard deviation {scoring}: {np.std(scores)}')

    def boosrap(self):
        pass

