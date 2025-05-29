import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.models.mlestimator import MachineLearningEstimator
from typing import Optional
import os
import pickle

shap.initjs()

class MLExplainer(MachineLearningEstimator):
    """
    Class for calculating and visualizing SHAP values for machine learning models.

    This class extends MachineLearningEstimator to add SHAP analysis functionality.
    """

    def __init__(
            self, 
            label: str,
            csv_dir: str,
            index_col: Optional[str] = None,
            normalization: str = "minmax",
            mv_method: str = "median",
            fs_method: str = "mrmr",
            inner_fs_method: str = "chi2",
            class_balance_method: Optional[str] = None,
            database_name: Optional[str] = None,
            preprocess_mode: str = "general",
            best_model: Optional[object] = None,
            estimator_name: Optional[str] = None,
            shap_values: Optional[np.ndarray] = None
        ) -> None:
        """
        Initialize the MLExplainer with a trained model and dataset.
        
        Parameters
        ----------
        label : str
            Target column name
        csv_dir : str
            Path to the CSV file
        index_col : str, optional
            Column to use as index
        normalization : str, default="minmax"
            Normalization method to use
        mv_method : str, default="median"
            Missing values handling method
        fs_method : str, default="mrmr"
            Feature selection method
        inner_fs_method : str, default="chi2"
            Inner feature selection method
        class_balance_method : str, optional
            Class balancing method
        database_name : str, optional
            Database name for storing results
        preprocess_mode : str, default="general"
            Preprocessing mode
        best_model : object, optional
            Trained model to explain
        estimator_name : str, optional
            Name of the estimator
        shap_values : ndarray, optional
            Pre-calculated SHAP values
        """
        # Initialize the parent class with basic parameters
        super().__init__(
            label=label,
            csv_dir=csv_dir,
            index_col=index_col,
            normalization=normalization,
            mv_method=mv_method,
            fs_method=fs_method,
            inner_fs_method=inner_fs_method,
            class_balance_method=class_balance_method,
            database_name=database_name,
            preprocess_mode=preprocess_mode
        )
        
        # Add SHAP-specific attributes
        self.best_model = best_model
        self.estimator_name = estimator_name
        self.explainer = None
        
        # Set SHAP values if provided and valid
        if self.shap_values is None or np.all(shap_values == 0):
            self.shap_values = None
        else:
            self.shap_values = shap_values
            
        # Create label mapping from existing data
        self.label_mapping = {i: class_name for i, class_name in enumerate(np.unique(self.y))}
        
        self.logger.info("MLExplainer initialized with model: %s", estimator_name)

    def calculate_shap_values(
            self, 
            model_path: Optional[str] = None,
            explainer_type:str = "general"
        ) -> np.ndarray:
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
            If the explainer type is unsupported or model is not set.
        TypeError
            If the model is incompatible with the general explainer.
        """

        if model_path is not None:
            if os.path.exists(model_path):
                with open(model_path, "rb") as model_file:
                    self.best_model = pickle.load(model_file)
                self.logger.info("Loaded model from path: %s", model_path)
            else:
                self.logger.warning("Model path does not exist: %s", model_path)

        if self.best_model is None:
            raise ValueError("No model is set. Train or load a model before calculating SHAP values.")
            
        if explainer_type == "general":
            try:
                self.explainer = shap.Explainer(self.best_model, self.X)
            except TypeError as e:
                if "The passed model is not callable" in str(e):
                    self.logger.info("Switching to predict_proba due to compatibility issue with the model.")
                    self.explainer = shap.Explainer(lambda X: self.best_model.predict_proba(X), self.X)
                else:
                    raise TypeError(e)

        elif explainer_type == "tree":
            if self.estimator_name not in [
                "RandomForestClassifier", "XGBClassifier", "CatBoostClassifier", "LGBMClassifier"
            ]:
                raise ValueError("Tree explainer supports tree-based models only.")
            self.explainer = shap.TreeExplainer(self.best_model, data=self.X, model_output="probability")

        elif explainer_type == "linear":
            if self.estimator_name not in ["LogisticRegression", "LinearDiscriminantAnalysis", "ElasticNet"]:
                raise ValueError("Linear explainer supports linear models only.")
            self.explainer = shap.LinearExplainer(self.best_model, self.X)

        else:
            raise ValueError("Unsupported explainer. Use 'general', 'tree', or 'linear'.")

        if self.explainer is not None:
            try:
                self.shap_values = self.explainer(self.X)
                self.logger.info("SHAP values calculated successfully")
            except ValueError:
                num_features = self.X.shape[1]
                max_evals = 2 * num_features + 1
                self.shap_values = self.explainer(self.X, max_evals=max_evals)
                self.logger.info(f"SHAP values calculated with max_evals={max_evals}")
        else:
            raise ValueError("Explainer is not defined.")
        
        return self.shap_values

    def plot_shap_values(
            self, 
            max_display: int = 10, 
            features_name_list: list = None, 
            plot_type: str = "summary", 
            label: int = 1
        ) -> None:
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
            If an unsupported plot type is provided or no SHAP values are calculated.
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values available. Run calculate_shap_values() first.")
            
        if features_name_list is not None:
            X = self.X[features_name_list]
        else:
            X = self.X
        
        # Ensure shap_values is in the proper format
        if not isinstance(self.shap_values, shap.Explanation):
            self.logger.info("Converting SHAP values to Explanation object")
            self.shap_values = shap.Explanation(
                values=self.shap_values,
                feature_names=X.columns,
                data=X
            )
        
        if plot_type == "summary":
            try:
                # Try to plot for specific class label
                shap.summary_plot(
                    shap_values=self.shap_values[:, :, label] if len(self.shap_values.shape) == 3 else self.shap_values,
                    features=X,
                    feature_names=X.columns,
                    max_display=max_display,
                    sort=True,
                )
                self.logger.info(f"Created summary plot for label {label}, corresponding to {self.label_mapping.get(label, 'unknown')}")
            except (IndexError, AttributeError):
                self.logger.info(f"Could not create specific label plot. Showing summary plot for all data.")
                shap.summary_plot(
                    shap_values=self.shap_values.values if hasattr(self.shap_values, 'values') else self.shap_values,
                    features=X,
                    feature_names=X.columns,
                    max_display=max_display,
                    sort=True,
                )

        elif plot_type == "beeswarm":
            try:
                shap.plots.beeswarm(
                    self.shap_values[:, :, label] if len(self.shap_values.shape) == 3 else self.shap_values, 
                    max_display=max_display
                )
                self.logger.info(f"Created beeswarm plot for label {label}")
            except (IndexError, AttributeError):
                self.logger.info("Could not create specific label plot. Showing beeswarm plot for all data.")
                shap.plots.beeswarm(self.shap_values, max_display=max_display)

        elif plot_type == "bar":
            try:
                if len(self.shap_values.shape) == 3:
                    shap.plots.bar(self.shap_values[:, :, label], max_display=max_display)
                else:
                    shap.plots.bar(self.shap_values, max_display=max_display)
                self.logger.info(f"Created bar plot for SHAP values")
            except Exception as e:
                self.logger.error(f"Error creating bar plot: {str(e)}")
                print("Unexpected SHAP values format for bar plot.")

        else:
            raise ValueError("Unsupported plot type. Use 'summary', 'beeswarm', or 'bar'.")

    def plot_shap_pca(
            self, 
            label: int = 1
        ) -> None:
        """
        Generate a PCA plot of SHAP values, colored by the target labels.

        Parameters:
        -----------
        label : int, optional
            The class label to visualize. If None, uses all data.  For multiclass,
            if label is None, it will plot using the first principal component
            of the SHAP values for each class. If label is specified, it uses
            the SHAP values for that specific class.
        """
        if not isinstance(self.shap_values, shap.Explanation):
             self.shap_values = shap.Explanation(
                values=self.shap_values,
                feature_names=self.X.columns,
                data=self.X
            )
        # Handle multiclass SHAP values
        if len(self.shap_values.shape) == 3:
            if label is None:
                # Use the first principal component of SHAP values for each class.
                shap_values_2d = np.array([PCA(n_components=1).fit_transform(self.shap_values[:, :, i].values).flatten() for i in range(self.shap_values.shape[2])]).T
            else:
                shap_values_2d = self.shap_values[:, :, label].values
        elif len(self.shap_values.shape) == 2:
            shap_values_2d = self.shap_values.values
        else:
            raise ValueError("Unexpected SHAP values format.")

        # Apply PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(shap_values_2d)
        principal_df = pd.DataFrame(
            data=principal_components, columns=["principal component 1", "principal component 2"]
        )

        # Prepare data for plotting
        if isinstance(self.y, pd.Series):
            target = self.y.values
        else:
            target = self.y
        if len(target.shape) > 1:
            target = np.argmax(target, axis=1)

        final_df = pd.concat([principal_df, pd.Series(target, name="label")], axis=1)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        targets = np.unique(target)
        labels = [self.label_mapping[t] for t in targets]
        colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "purple", "brown"]  # Extend as needed

        for i, target_label in enumerate(targets):
            indices_to_keep = final_df["label"] == target_label
            ax.scatter(
                final_df.loc[indices_to_keep, "principal component 1"],
                final_df.loc[indices_to_keep, "principal component 2"],
                color=colors[i % len(colors)],  # Use modulo to cycle through colors
                label=labels[i],
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel("PC1", fontsize=15)
        ax.set_ylabel("PC2", fontsize=15)
        ax.set_title("PCA of SHAP Values", fontsize=20)
        ax.legend(loc="best")
        plt.show()
