import pandas as pd
import numpy as np
import math
import shap

shap.initjs()
from .mlestimator import MachineLearningEstimator


class FeaturesExplanation:
    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.explainer = None
        self.X = X
        self.y = y
        self.shap_values = None

    def calculate_shap_values(self, explainer_type="general"):
        """
        Calculate SHAP values
        input:
            explainer_type: 'general', 'tree', 'linear'
        """
        if explainer_type == "general":
            self.explainer = shap.Explainer(self.estimator, self.X)
        elif explainer_type == "tree":
            if self.estimator.__class__.__name__ not in [
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "XGBClassifier",
            ]:
                raise ValueError(
                    "Only DecisionTreeClassifier, RandomForestClassifier, XGBClassifier are supported for tree explainer"
                )
            elif (
                self.estimator.__class__.__name__ == "XGBClassifier" and self.estimator.booster != "gbtree"
            ):
                raise ValueError("XGBClassifier requires 'booster' to be 'gbtree'")
            else:
                self.explainer = shap.TreeExplainer(
                    self.estimator, data=self.X, model_output="probability"
                )
        elif explainer_type == "linear":
            if self.estimator.__class__.__name__ not in ["LogisticRegression", "LinearDiscriminantAnalysis"]:
                raise ValueError(
                    "Only LogisticRegression and LinearDiscriminantAnalysis are supported for linear explainer"
                )
            else:
                self.explainer = shap.LinearExplainer(self.estimator, self.X)
        else:
            raise ValueError(
                "Unsupported explainer. Select one of 'general', 'tree', 'linear'"
            )

        if self.explainer is not None:
            try:
                self.shap_values = self.explainer(self.X)
            except ValueError:
                num_features = self.X.shape[1]
                max_evals = 2 * num_features + 1
                self.shap_values = self.explainer(self.X, max_evals=max_evals)
        else:
            raise ValueError("Explainer is not defined")

    def plot_shap_values(self, max_display=10, plot_type="summary", label=0):
        if plot_type == "summary":
            try:
                shap.summary_plot(
                    shap_values=self.shap_values[:, :, label],
                    features=self.X,
                    feature_names=self.X.columns,
                    max_display=max_display,
                    sort=True,
                )
                print(
                    f"The plot is for label {label}, corresponding to {self.label_mapping[label]}"
                )
            except IndexError:
                print(
                    f"The shap values do not exist for the label {label}. The following is the summary plot for all the labels."
                )
                shap.summary_plot(
                    shap_values=self.shap_values,
                    features=self.X,
                    feature_names=self.X.columns,
                    max_display=max_display,
                    sort=True,
                )
        elif plot_type == "beeswarm":
            try:
                shap.plots.beeswarm(
                    self.shap_values[:, :, label], max_display=max_display
                )
                print(
                    f"The plot is for label {label}, corresponding to {self.label_mapping[label]}"
                )
            except IndexError:
                print(
                    f"The shap values do not exist for the label {label}. The following is the beeswarm plot for all the labels."
                )
                shap.plots.beeswarm(self.shap_values, max_display=max_display)

        else:
            raise ValueError(
                "Unsupported plot type. Select one of 'summary', 'beeswarm'"
            )
