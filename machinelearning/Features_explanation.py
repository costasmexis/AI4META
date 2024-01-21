import pandas as pd
import numpy as np
import shap
# from .mlestimator import MachineLearningEstimator
# from .mlpipeline import MLPipelines

class Features_explanation():
    def __init__(self, best_estimator, X, y):
        self.best_estimator = best_estimator
        self.X = X
        self.y = y
    
    # def f(self, feature):
    #     set_res = set(self.y)
    #     unique_targets = list(set_res)
    #     if feature >= self.X.shape[1]:
    #         raise ValueError(
    #             f'Too high feature number. The number of features is {self.X.shape[1]}. Select a feature between 0 and {self.X.shape[1]-1}') 
    #     else:
    #         print(f'best estimator = {self.best_estimator}, {self.name}')
    #         return self.best_estimator.predict_proba(self.X)[:, feature]

    def explainer2use(self, explainer_type='general'):
        if explainer_type == 'general':
            return shap.Explainer(self.best_estimator, self.X)
        elif explainer_type == 'tree':
            if self.name not in ['DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']:
                raise ValueError("Only DecisionTreeClassifier, RandomForestClassifier, XGBClassifier are supported for tree explainer")
            else:
                return shap.TreeExplainer(self.best_estimator, self.X)
        elif explainer_type == 'linear':
            if self.name not in ['LogisticRegression', 'LinearDiscriminantAnalysis']:
                raise ValueError("Only LogisticRegression and LinearDiscriminantAnalysis are supported for linear explainer")
            else:
                return shap.LinearExplainer(self.best_estimator, self.X)
        else:
            raise ValueError("Unsupported explainer. Select one of 'general', 'tree', 'linear'")

    def calculate_shap_values(self, explainer_type='general'):
        # num_features = self.X.shape[1]
        # min_evals = 2 * num_features + 1
        explainer = self.explainer2use(explainer_type)
        return explainer.shap_values(self.X)#, max_evals=min_evals)

    def plot_shap_values(self, explainer_type='general'):
        shap_values = self.calculate_shap_values(explainer_type)
        shap.summary_plot(shap_values, self.X)

    # def beeswarn_plt(self, explainer_type='general'):
    #     shap_values = self.calculate_shap_values(explainer_type)
    #     shap.plots.beeswarm(shap_values)