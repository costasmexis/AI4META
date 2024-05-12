import pandas as pd
import numpy as np
import math
import shap
shap.initjs()
from .mlestimator import MachineLearningEstimator
# from .mlpipeline import MLPipelines

class FeaturesExplanation(MachineLearningEstimator):
    def __init__(self, best_estimator, X, y, label_mapping,shap_values = None):
        self.best_estimator = best_estimator
        self.explainer = None
        self.name = best_estimator.__class__.__name__
        self.X = X
        self.y = y
        self.shap_values = shap_values
        self.label_mapping = label_mapping
        
    
    def calculate_shap_values(self,explainer_type='general'):
        '''
        Calculate SHAP values
        input:
            explainer_type: 'general', 'tree', 'linear' 
        '''
        if explainer_type == 'general':
            try :
                self.explainer = shap.Explainer(self.best_estimator, self.X)
            except TypeError as e:
                if "The passed model is not callable and cannot be analyzed directly with the given masker!" in str(e):
                    print("Switching to predict_proba due to compatibility issue with the model.")
                    self.explainer = shap.Explainer(lambda X: self.best_estimator.predict_proba(X), self.X)
                else:
                    raise TypeError(e)

        elif explainer_type == 'tree':
            if self.name not in ['DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier','CatBoostClassifier','LightGBMClassifier']:
                raise ValueError("Only DecisionTreeClassifier, RandomForestClassifier, XGBClassifier are supported for tree explainer")
            elif self.name == 'XGBClassifier' and self.best_estimator.booster != 'gbtree':
                raise ValueError("XGBClassifier requires 'booster' to be 'gbtree'")
            else:
                self.explainer = shap.TreeExplainer(self.best_estimator, data=self.X, model_output='probability')
        elif explainer_type == 'linear':
            if self.name not in ['LogisticRegression', 'LinearDiscriminantAnalysis']:
                raise ValueError("Only LogisticRegression and LinearDiscriminantAnalysis are supported for linear explainer")
            else:
                self.explainer = shap.LinearExplainer(self.best_estimator, self.X)

        else:
            raise ValueError("Unsupported explainer. Select one of 'general', 'tree', 'linear'")
        
        if self.explainer is not None:
            try:
                self.shap_values = self.explainer(self.X)
            except ValueError:
                num_features = self.X.shape[1]
                max_evals = 2 * num_features + 1
                self.shap_values = self.explainer(self.X, max_evals=max_evals)
        else: 
            raise ValueError("Explainer is not defined")

    def plot_shap_values(self,max_display=10,plot_type='summary',label=1):
        if plot_type == 'summary':
            try:
                shap.summary_plot(shap_values=self.shap_values[:,:,label], features=self.X,feature_names=self.X.columns,max_display=max_display,sort=True)
                print(f'The plot is for label {label}, corresponding to {self.label_mapping[label]}')
            except IndexError:
                print(f'The shap values do not exist for the label {label}. The following is the summary plot for all the labels.')
                shap.summary_plot(shap_values=self.shap_values, features=self.X,feature_names=self.X.columns,max_display=max_display,sort=True)
        elif plot_type == 'beeswarm':
            try:
                shap.plots.beeswarm(self.shap_values[:,:,label], max_display=max_display)
                print(f'The plot is for label {label}, corresponding to {self.label_mapping[label]}')
            except IndexError:
                print(f'The shap values do not exist for the label {label}. The following is the beeswarm plot for all the labels.')
                shap.plots.beeswarm(self.shap_values, max_display=max_display)    
        # elif plot_type == 'decision':
        #     print(self.shap_values[:,:,label][0])
        #     if isinstance(self.shap_values, np.ndarray):
        #         if self.shap_values.ndim == 3:  # Multi-class case
        #             shap_values_for_label = self.shap_values[:, :, label]
        #         else:
        #             shap_values_for_label = self.shap_values

        #         expected_value = self.explainer.expected_value
        #         if isinstance(expected_value, list):
        #             expected_value = expected_value[label]
        #         elif isinstance(expected_value, np.ndarray):
        #             expected_value = expected_value[label] if expected_value.ndim > 0 else expected_value
                
        #         shap.decision_plot(expected_value, shap_values_for_label, feature_names=self.X.columns, link='logit', show=True)
        #         print(f'The decision plot is for label {label}, corresponding to {self.label_mapping[label]}')
        elif plot_type == 'bar':
            if len(self.shap_values.shape)==3:
                shap.plots.bar(self.shap_values[:, :, label], max_display=max_display)
            elif len(self.shap_values.shape)==2:
                shap.plots.bar(self.shap_values, max_display=max_display)
            else:pass
        else: 
            raise ValueError("Unsupported plot type. Select one of 'summary', 'beeswarm'")




        
