import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, \
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier
from .Bayesian_Grid import bayesian_grid
from dataloader import DataLoader
from optuna.integration import OptunaSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, mean_squared_error

class MachineLearningEstimator(DataLoader):
    def __init__(self, estimator, param_grid, label, csv_dir):
        ''' Class to hold the machine learning estimator and related data 
            Inherits from DataLoader class
            - estimator (sklearn estimator): estimator to be used
            - param_grid (dict): hyperparameters to be tuned
            - label (str): name of the target column
            - csv_dir (str): path to the csv file
        '''
        super().__init__(label, csv_dir)
        self.estimator = estimator
        self.name = estimator.__class__.__name__
        self.param_grid = param_grid
        self.bayesian_grid = bayesian_grid
        self.best_params = None
        self.best_score = None
        self.best_estimator = None
        
        self.available_clfs = {
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'LogisticRegression': LogisticRegression(),
            'XGBClassifier': XGBClassifier(),
            'GaussianNB': GaussianNB(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'SVC': SVC(),
            'PLSRegression': PLSRegression()
        }
        
        # Check if the estimator is valid 
        if self.name not in self.available_clfs.keys():
            raise ValueError(f'Invalid estimator: {self.name}. Select one of the following: {list(self.available_clfs.keys())}')
   
    def encode_labels(self):
        self.label_encoder = LabelEncoder()
        
        self.y = self.label_encoder.fit_transform(self.y)
        
        label_mapping = {class_label: index for index, class_label in enumerate(self.label_encoder.classes_)}
        print("Label mapping:", label_mapping)
    
    # def PLS_performance(self, model, X_test, y_test, scoring='accuracy'):
    #     y_pred = model.predict(X_test)

    #     if isinstance(model, PLSRegression):
    #        y_pred = (y_pred[:, 0] > 0.5).astype('uint8')
          
    # #     if self.inner_scoring == 'roc_auc':
    # #        score = roc_auc_score(y_test, y_pred)
    # #     elif self.inner_scoring == 'matthews_corrcoef':
    # #        score = matthews_corrcoef(y_test, y_pred)
    # #   # elif self.inner_scoring == 'specificity':
    # #   #   score = mean_squared_error(y_test, y_pred)
    # #     elif self.inner_scoring == 'accuracy':
            
    #     score = accuracy_score(y_test, y_pred)
    #     return score

    def grid_search(self, X=None, y=None, scoring='accuracy', cv=5, verbose=True):
        ''' Function to perform a grid search
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
            - verbose (bool): whether to print the results
        '''
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        
        if X is None and y is None:
            X = self.X 
            y = self.y
        
        grid_search = GridSearchCV(self.estimator, self.param_grid, scoring=scoring, cv=cv)
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_estimator = grid_search.best_estimator_
        if verbose:
            print(f'Best parameters: {self.best_params}')
            print(f'Best {scoring}: {self.best_score}')

    def random_search(self, X=None, y=None, scoring='accuracy', cv=5, n_iter=100, verbose=True):
        ''' Function to perform a random search
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
            - n_iter (int): number of iterations
            - verbose (bool): whether to print the results
        '''
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')

        if X is None and y is None:
            X = self.X
            y = self.y

        random_search = RandomizedSearchCV(self.estimator, self.param_grid, scoring=scoring, cv=cv, n_iter=n_iter)
        random_search.fit(X, y)
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_estimator = random_search.best_estimator_
        if verbose:
            print(f'Best parameters: {self.best_params}')
            print(f'Best {scoring}: {self.best_score}')

   
    def bayesian_search(self, X=None, y=None, scoring='accuracy', cv=5, direction='maximize', n_trials=100, verbose=True, box=False):
        grid = self.bayesian_grid['ManualSearch']
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
            
        if X is None and y is None:
            X = self.X#.values
            y = self.y
                              
        def objective(trial):
            clf = grid[self.name](trial)
            # if clf.__class__ == PLSRegression:
            #     cv_splitter = KFold(n_splits=cv, shuffle=True)
            #     model_scores = []
            #     # print(type(X), type(y),X)
            #     for train_idx, test_idx in cv_splitter.split(X, y):
            #         # X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            #         # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            #         X_train, X_test = X[train_idx], X[test_idx]
            #         y_train, y_test = y[train_idx], y[test_idx]
            #         clf.fit(X_train, y_train)
            #         score = self.PLS_performance(clf, X_test, y_test)
            #         model_scores.append(score) 
            #     final_score = np.mean(model_scores)  
            # else:
            final_score = cross_val_score(clf, X, y, scoring=scoring, cv=cv).mean()
            return final_score

        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_estimator = grid[self.name](study.best_trial)

        print(f'Best parameters: {self.best_params}')
        print(f'Best {scoring}: {self.best_score}')
        
        # if box and self.estimator.__class__ == PLSRegression:
        #     print("Boxplot is not available for PLSRegression")
        
        if box:
            best_scores = [trial.value for trial in study.trials if trial.value is not None]
            plt.style.use('seaborn-whitegrid')
            plt.boxplot(best_scores, widths=0.75, whis=2)
            plt.ylim(0, 1)
            plt.title(f"Cross-Validation Scores Across Trials for {self.name}")
            plt.ylabel('Scores')
            plt.xlabel(f'{cv}-Fold Cross-Validation')
            plt.show()

    
    # def bayesian_search(self, X=None, y=None, scoring='accuracy', cv=5, n_trials=100, verbose=True):
    #     if scoring not in sklearn.metrics.get_scorer_names():
    #         raise ValueError(
    #             f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')

    #     if X is None and y is None:
    #         X = self.X 
    #         y = self.y

    #     # Set up OptunaSearchCV with the Bayesian grid 
    #     optuna_search = OptunaSearchCV(
    #         estimator=self.available_clfs[self.name],  # Ensure this matches the classifier names in bayesian_grid
    #         param_distributions=self.bayesian_grid[self.name],
    #         n_trials=n_trials,
    #         scoring=scoring,
    #         cv=cv,
    #         verbose=verbose
    #     )

    #     # Perform the search
    #     optuna_search.fit(X, y)

    #     # Store the results
    #     self.best_params = optuna_search.best_params_
    #     self.best_score = optuna_search.best_score_
    #     self.best_estimator = optuna_search.best_estimator_
        if verbose:
            print(f'Best parameters: {self.best_params}')
            print(f'Best {scoring}: {self.best_score}')
        
        
