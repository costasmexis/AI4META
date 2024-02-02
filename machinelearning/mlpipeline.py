import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from dataloader import DataLoader

from .mlestimator import MachineLearningEstimator
from .optuna_grid import optuna_grid


class MLPipelines(MachineLearningEstimator):
    """ Class to perform machine learning pipelines

    :param estimator: Estimator to be used
    :type estimator: sklearn estimator
    :param param_grid: Hyperparameters grid to be searched
    :type param_grid: dict
    :param label: me of target column
    :type label: str
    :param csv_dir: Path to the csv file
    :type csv_dir: str
    """       
    def __init__(self, label: str, csv_dir: str, estimator: object = None, param_grid: dict = None, ):
         super().__init__(estimator, param_grid, label, csv_dir)

    def cross_validation(self, scoring: str = "matthews_corrcoef", cv: int = 5) -> list:
        """ Performs cross validation on a given estimator

        :param scoring: Scoring metric, defaults to ``matthews_corrcoef``
        :type scoring: str, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :return: List of scores for each fold
        :rtype: list
        """        
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        scores = cross_val_score(self.estimator, self.X, self.y, cv=cv, scoring=scoring)
        print(f"Average {scoring}: {np.mean(scores)}")
        print(f"Standard deviation {scoring}: {np.std(scores)}")
        return scores

    def bootsrap(
        self,
        n_iter=100,
        test_size=0.2,
        optimizer="grid_search",
        random_iter=25,
        n_trials=100,
        cv=5,
        scoring="matthews_corrcoef",
        verbose=False
    ):
        """Performs boostrap validation on a given estimator.

        :param n_iter: Number of iterations to perform bootstrap validation, defaults to 100
        :type n_iter: int, optional
        :param test_size: Test size for each iteration, defaults to 0.2
        :type test_size: float, optional
        :param optimizer: Method to use for hyperparameter optimization, defaults to ``grid_search``
        :type optimizer: str, optional
        :param random_iter: Number of iterations for ``RandomSearchCV``, defaults to 25
        :type random_iter: int, optional
        :param n_trials: Number of trials for ``Optuna``, defaults to 100
        :type n_trials: int, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :param scoring: Scoring metric, defaults to ``matthews_corrcoef``
        :type scoring: str, optional
        :return: List of evaluation metrics for each iteration
        :rtype: list
        """        
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        eval_metrics = []
        for i in range(n_iter):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=i
            )
            if self.param_grid is None or self.param_grid == {}:
                self.estimator.fit(X_train, y_train)
            else:
                if optimizer == "grid_search":
                    self.grid_search(
                        X_train, y_train, scoring=scoring, cv=cv, verbose=False
                    )
                elif optimizer == "random_search":
                    self.random_search(
                        X_train,
                        y_train,
                        scoring=scoring,
                        cv=cv,
                        n_iter=random_iter,
                        verbose=verbose,
                    )
                elif optimizer == "bayesian_search":
                    self.bayesian_search(
                        X_train,
                        y_train,
                        scoring=scoring,
                        direction="maximize",
                        cv=cv,
                        n_trials=n_trials,
                        verbose=verbose,
                    )
                    self.best_estimator.fit(X_train, y_train)
                else:
                    raise ValueError(
                        f"Invalid optimizer: {optimizer}. Select one of the following: grid_search, bayesian_search"
                    )

            y_pred = self.best_estimator.predict(X_test)
            eval_metrics.append(get_scorer(scoring)._score_func(y_test, y_pred))

        print(f"Average {scoring}: {np.mean(eval_metrics)}")
        print(f"Standard deviation {scoring}: {np.std(eval_metrics)}")
        return eval_metrics

    def nested_cross_validation(
        self,
        inner_scoring="matthews_corrcoef",
        outer_scoring="matthews_corrcoef",
        inner_splits=3,
        outer_splits=5,
        optimizer="grid_search",
        n_trials=100,
        n_iter=25,
        n_runs=10,
        n_jobs=-1,
        verbose=0,
    ):
        """Performs nested cross-validation for a given model and dataset in order to perform model selection

        :param inner_scoring: Inner loop scoring metric, defaults to ``matthews_corrcoef``
        :type inner_scoring: str, optional
        :param outer_scoring: Outer loop scoring metric, defaults to ``matthews_corrcoef``
        :type outer_scoring: str, optional
        :param inner_splits: Number of folds for inner loop cross validation, defaults to 3
        :type inner_splits: int, optional
        :param outer_splits: Number of folds for outer loop cross validation, defaults to 5
        :type outer_splits: int, optional
        :param optimizer: Method to be used for hyperparameter optimization, defaults to ``grid_search``
        :type optimizer: str, optional
        :param n_trials: Number of trials for ``Optuna``, defaults to 100
        :type n_trials: int, optional
        :param n_iter: Number of iterations for ``RandomizedSearchCV``, defaults to 25
        :type n_iter: int, optional
        :param n_runs: Number of runs for the Nested Cross Validation, defaults to 10
        :type n_runs: int, optional
        :param n_jobs: Number of workers to be used, defaults to -1
        :type n_jobs: int, optional
        :param verbose: Verbose, defaults to 0
        :type verbose: int, optional
        :return: Nested scores for each run 
        :rtype: list
        """        
        
        # Check if both inner and outer scoring metrics are valid
        if inner_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )
        if outer_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        print(
            f"Performing nested cross-validation for {self.estimator.__class__.__name__}..."
        )

        nested_scores = []
        for i in tqdm(range(n_runs)):
            inner_cv = StratifiedKFold(
                n_splits=inner_splits, shuffle=True, random_state=i
            )
            outer_cv = StratifiedKFold(
                n_splits=outer_splits, shuffle=True, random_state=i
            )

            if optimizer == "grid_search":
                clf = GridSearchCV(
                    estimator=self.estimator,
                    scoring=inner_scoring,
                    param_grid=self.param_grid,
                    cv=inner_cv,
                    n_jobs=n_jobs,
                    verbose=verbose,
                )
            elif optimizer == "random_search":
                clf = RandomizedSearchCV(
                    estimator=self.estimator,
                    scoring=inner_scoring,
                    param_distributions=self.param_grid,
                    cv=inner_cv,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    n_iter=n_iter,
                )
            elif optimizer == "bayesian_search":
                clf = optuna.integration.OptunaSearchCV(
                    estimator=self.estimator,
                    scoring=inner_scoring,
                    param_distributions=optuna_grid["NestedCV"][self.estimator.__class__.__name__],
                    cv=inner_cv,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    n_trials=n_trials,
                )
            else:
                raise Exception("Unsupported optimizer.")

            nested_score = cross_val_score(
                clf,
                X=self.X,
                y=self.y,
                cv=outer_cv,
                scoring=outer_scoring,
                n_jobs=n_jobs,
            )
            nested_scores.append(nested_score)

        nested_scores = [item for sublist in nested_scores for item in sublist]

        return nested_scores

    def model_selection(
        self,
        optimizer="grid_search",
        n_trials=100,
        n_iter=25,
        n_runs=10,
        score="matthews_corrcoef",
        exclude=None,
        result=False,
        box=True,
        train_best=True,
        verbose=True
    ):
        """Performs model selection for a given dataset

        :param optimizer: _description_, defaults to "grid_search"
        :type optimizer: str, optional
        :param n_trials: _description_, defaults to 100
        :type n_trials: int, optional
        :param n_iter: _description_, defaults to 25
        :type n_iter: int, optional
        :param n_runs: _description_, defaults to 10
        :type n_runs: int, optional
        :param score: _description_, defaults to "matthews_corrcoef"
        :type score: str, optional
        :param exclude: _description_, defaults to None
        :type exclude: _type_, optional
        :param result: _description_, defaults to False
        :type result: bool, optional
        :param box: _description_, defaults to True
        :type box: bool, optional
        :param train_best: _description_, defaults to None
        :type train_best: bool, optional
        :return: Scores for each estimator
        :rtype: pd.DataFrame
        """        
        all_scores = []
        results = []

        if exclude is not None:
            exclude_classes = [classifier.__class__ for classifier in exclude]
        else:
            exclude_classes = []

        clfs = [
            clf
            for clf in self.available_clfs.keys()
            if self.available_clfs[clf].__class__ not in exclude_classes
        ]

        for estimator in tqdm(clfs):
            print(f"Performing nested cross-validation for {estimator}...")
            # self.name = estimator
            self.estimator = self.available_clfs[estimator]
            scores_est = self.nested_cross_validation(
                optimizer=optimizer,
                n_runs=n_runs,
                n_iter=n_iter,
                n_trials=n_trials,
                inner_scoring=score,
                outer_scoring=score,
            )
            scores_array = np.array([round(num, 4) for num in scores_est])
            all_scores.append(scores_array)
            results.append(
                {
                    "Estimator": estimator,
                    "Scores": scores_array,
                    "Mean Score": np.mean(scores_array),
                    "Max Score": np.max(scores_array),
                }
            )

        self.estimator = self.available_clfs[max(results, key=lambda x: x["Mean Score"])["Estimator"]]

        if train_best == "bayesian_search":
            self.bayesian_search(cv=5, n_trials=n_iter, verbose=verbose)
        elif train_best == "grid_search":
            self.grid_search(cv=5, verbose=verbose)
        elif train_best == "random_search":
            self.random_search(cv=5, n_iter=n_iter, verbose=verbose)
        elif train_best is None:
            print(f"Best estimator: {self.best_estimator}")
        else:
            raise ValueError(
                f'Invalid type of best estimator train. Choose between "bayesian_search", "grid_search", "random_search" or None.'
            )
            
        if box:
            plt.boxplot(all_scores, labels=clfs)
            plt.title("Model Selection Results")
            plt.ylabel("Score")
            plt.xticks(rotation=90)
            plt.show()

        if result:
            return pd.DataFrame(results)
