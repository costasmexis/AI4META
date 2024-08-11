import shap
shap.initjs()

class MLExplainer:
    def __init__(self, estimator, X, y, label_mapping, shap_values=None):
        self.estimator = estimator
        self.explainer = None
        self.name = estimator.__class__.__name__
        self.X = X
        self.y = y
        self.shap_values = shap_values
        self.label_mapping = label_mapping

    def calculate_shap_values(self, explainer_type="general"):
        """
        Calculate SHAP values
        Note: explainer_type: 'general', 'tree', 'linear'
        """
        if explainer_type == "general":
            try:
                self.explainer = shap.Explainer(self.estimator, self.X)
            except TypeError as e:
                if "The passed model is not callable and cannot be analyzed directly with the given masker!" in str(e):
                    print("Switching to predict_proba due to compatibility issue with the model.")
                    self.explainer = shap.Explainer(lambda X: self.estimator.predict_proba(X), self.X)
                else:
                    raise TypeError(e)

        elif explainer_type == "tree":
            if self.name not in [
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "XGBClassifier",
                "CatBoostClassifier",
                "LightGBMClassifier",
            ]:
                raise ValueError(
                    "Only DecisionTreeClassifier, RandomForestClassifier, XGBClassifier are supported for tree explainer"
                )
            elif self.name == "XGBClassifier" and self.estimator.booster != "gbtree":
                raise ValueError("XGBClassifier requires 'booster' to be 'gbtree'")
            else:
                self.explainer = shap.TreeExplainer(
                    self.estimator, data=self.X, model_output="probability"
                )
        elif explainer_type == "linear":
            if self.name not in ["LogisticRegression", "LinearDiscriminantAnalysis"]:
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

    def plot_shap_values(self, max_display=10, plot_type="summary", label=1):
        # Convert shap_values to Explanation object if not already one
        if not isinstance(self.shap_values, shap.Explanation):
            self.shap_values = shap.Explanation(values=self.shap_values, 
                                                feature_names=self.X.columns,
                                                data=self.X)

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
        elif plot_type == "bar":
            if len(self.shap_values.shape) == 3:
                shap.plots.bar(self.shap_values[:, :, label], max_display=max_display)
            elif len(self.shap_values.shape) == 2:
                shap.plots.bar(self.shap_values, max_display=max_display)
            else:
                pass
        else:
            raise ValueError(
                "Unsupported plot type. Select one of 'summary', 'beeswarm'"
            )
