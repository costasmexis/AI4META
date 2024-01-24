import pandas as pd
import numpy as np
import shap
shap.initjs()
# from .mlestimator import MachineLearningEstimator
# from .mlpipeline import MLPipelines

class Features_explanation():
    def __init__(self, best_estimator, X, y):
        self.best_estimator = best_estimator
        self.X = X
        self.y = y
    
    # def f(self, feature=0):
    #     set_res = set(self.y)
    #     unique_targets = list(set_res)
    #     if feature >= self.X.shape[1]:
    #         raise ValueError(
    #             f'Too high feature number. The number of features is {self.X.shape[1]}. Select a feature between 0 and {self.X.shape[1]-1}') 
    #     else:
    #         return self.best_estimator.predict_proba(self.X)[:, feature]

    def explainer2use(self, explainer_type='general'):
        if explainer_type == 'general':
            return shap.Explainer(self.best_estimator.predict, self.X)
        elif explainer_type == 'tree':
            if self.name not in ['DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']:
                raise ValueError("Only DecisionTreeClassifier, RandomForestClassifier, XGBClassifier are supported for tree explainer")
            else:
                return shap.TreeExplainer(self.best_estimator.predict, self.X)
        elif explainer_type == 'linear':
            if self.name not in ['LogisticRegression', 'LinearDiscriminantAnalysis']:
                raise ValueError("Only LogisticRegression and LinearDiscriminantAnalysis are supported for linear explainer")
            else:
                return shap.LinearExplainer(self.best_estimator.predict, self.X)
        else:
            raise ValueError("Unsupported explainer. Select one of 'general', 'tree', 'linear'")

    def calculate_shap_values(self, explainer_type='general'):
        num_features = self.X.shape[1]
        max_evals = 2 * num_features + 1
        explainer = self.explainer2use(explainer_type)
        return explainer(self.X, max_evals=max_evals)

    def plot_shap_values(self, explainer_type='general',max_display=10,plot_type=None):
        shap_values = self.calculate_shap_values(explainer_type)
        # shap_obj = shap.Explanation(values=shap_values,data=self.X,feature_names=self.X.columns)
        shap.summary_plot(shap_values=shap_values, features=self.X,feature_names=self.X.columns,max_display=max_display,sort=True,plot_type=plot_type)

    def beeswarn_plt(self, explainer_type='general',n_features=10):
        explainer = shap.Explainer(self.best_estimator)#, self.X)
        shap_values = explainer(self.X)
        shap_obj = shap.Explanation(values=shap_values,data=self.X,feature_names=self.X.columns, base_values=explainer.expected_value)
        shap.plots.beeswarm(shap_obj[:,:,0], max_display=n_features)



        
