import shap
from src.models.mlestimator import MachineLearningEstimator

shap.initjs()

class MLExplainer(MachineLearningEstimator):
    """
    Class for calculating and visualizing SHAP values for machine learning models.

    This class supports general, tree-based, and linear explainers for various types of models.

    Attributes:
    -----------
    estimator : object
        The trained machine learning model to explain.
    X : pandas.DataFrame
        Input feature data.
    y : pandas.Series or numpy.ndarray
        Target labels.
    label_mapping : dict
        Mapping of label indices to label names.
    shap_values : shap.Explanation or None
        Calculated SHAP values.
    explainer : shap.Explainer or None
        The SHAP explainer object.
    """

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
        Calculate SHAP values for the given model and dataset.

        Parameters:
        -----------
        explainer_type : str, optional
            Type of SHAP explainer to use. Options are 'general', 'tree', 'linear'.
            Defaults to 'general'.

        Raises:
        -------
        ValueError
            If the explainer type is unsupported.
        TypeError
            If the model is incompatible with the general explainer.
        """
        if explainer_type == "general":
            try:
                self.explainer = shap.Explainer(self.estimator, self.X)
            except TypeError as e:
                if "The passed model is not callable" in str(e):
                    print("Switching to predict_proba due to compatibility issue with the model.")
                    self.explainer = shap.Explainer(lambda X: self.estimator.predict_proba(X), self.X)
                else:
                    raise TypeError(e)

        elif explainer_type == "tree":
            if self.name not in [
                "RandomForestClassifier", "XGBClassifier", "CatBoostClassifier", "LightGBMClassifier"]:
                raise ValueError("Tree explainer supports tree-based models only.")
            elif self.name == "XGBClassifier" and self.estimator.booster != "gbtree":
                raise ValueError("XGBClassifier requires 'booster' to be 'gbtree'.")
            self.explainer = shap.TreeExplainer(self.estimator, data=self.X, model_output="probability")

        elif explainer_type == "linear":
            if self.name not in ["LogisticRegression", "LinearDiscriminantAnalysis"]:
                raise ValueError("Linear explainer supports linear models only.")
            self.explainer = shap.LinearExplainer(self.estimator, self.X)

        else:
            raise ValueError("Unsupported explainer. Use 'general', 'tree', or 'linear'.")

        if self.explainer is not None:
            try:
                self.shap_values = self.explainer(self.X)
            except ValueError:
                num_features = self.X.shape[1]
                max_evals = 2 * num_features + 1
                self.shap_values = self.explainer(self.X, max_evals=max_evals)
        else:
            raise ValueError("Explainer is not defined.")

    def plot_shap_values(self, max_display=10, plot_type="summary", label=1):
        """
        Plot SHAP values using various visualization types.

        Parameters:
        -----------
        max_display : int, optional
            Maximum number of features to display, by default 10.
        plot_type : str, optional
            Type of SHAP plot ('summary', 'beeswarm', 'bar'). Defaults to 'summary'.
        label : int, optional
            Label index for multi-class models. Defaults to 1.

        Raises:
        -------
        ValueError
            If an unsupported plot type is provided.
        """
        # Ensure shap_values is in the proper format
        if not isinstance(self.shap_values, shap.Explanation):
            self.shap_values = shap.Explanation(
                values=self.shap_values,
                feature_names=self.X.columns,
                data=self.X
            )

        if plot_type == "summary":
            try:
                shap.summary_plot(
                    shap_values=self.shap_values[:, :, label],
                    features=self.X,
                    feature_names=self.X.columns,
                    max_display=max_display,
                    sort=True,
                )
                print(f"The plot is for label {label}, corresponding to {self.label_mapping[label]}.")
            except IndexError:
                print(f"The SHAP values do not exist for the label {label}. Showing summary plot for all labels.")
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
                print(f"The plot is for label {label}, corresponding to {self.label_mapping[label]}.")
            except IndexError:
                print(f"The SHAP values do not exist for the label {label}. Showing beeswarm plot for all labels.")
                shap.plots.beeswarm(self.shap_values, max_display=max_display)

        elif plot_type == "bar":
            if len(self.shap_values.shape) == 3:
                shap.plots.bar(self.shap_values[:, :, label], max_display=max_display)
            elif len(self.shap_values.shape) == 2:
                shap.plots.bar(self.shap_values, max_display=max_display)
            else:
                print("Unexpected SHAP values format for bar plot.")

        else:
            raise ValueError("Unsupported plot type. Use 'summary', 'beeswarm', or 'bar'.")
