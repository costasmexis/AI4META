from sklearn.metrics import get_scorer_names
import re
from typing import Dict, List, Optional, Union
import pandas as pd
import logging

class ConfigValidator:
    """
    A class to validate and normalize configuration settings for machine learning pipelines.
    
    This class handles validation of configuration parameters, normalizes values, and ensures
    all required settings are present with appropriate values.
    """
    
    def __init__(self, available_clfs: Dict):
        """
        Initialize the ConfigValidator with available classifiers.
        
        Args:
            available_clfs (Dict): Dictionary of available classifier objects
        """
        self.available_clfs = available_clfs
        self.valid_scorers = list(get_scorer_names()) + ["specificity"]
        self.valid_class_balance = ["smote", "borderline_smote", "tomek", None]
        self.valid_missing_values = ["mean", "median", "0", "drop"]
        self.valid_normalizations = ["minmax", "standard", None]
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)
        
        # Set to track logged messages
        self.logged_messages = set()
        
    def _log_once(self, message: str, level: str = 'warning') -> None:
        """Log a message only if it hasn't been logged before."""
        if message not in self.logged_messages:
            if level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'info':
                self.logger.info(message)
            self.logged_messages.add(message)
    
    def validate_config(self, 
                       config: Dict,
                       main_type: str,
                       X: pd.DataFrame,
                       csv_dir: str,
                       label: str) -> Dict:
        """
        Validate and normalize the complete configuration dictionary.
        
        Args:
            config (Dict): Configuration dictionary to validate
            main_type (str): Type of validation process ('rncv', 'rcv', etc.)
            X (pd.DataFrame): Input dataset
            csv_dir (str): Path to the CSV file
            label (str): Target label name
            
        Returns:
            Dict: Validated and normalized configuration dictionary
        """
        # Check for required keys and provide defaults only when needed
        self._validate_required_keys(config, main_type)
        
        # Validate basic settings
        self._validate_missing_values(config, X)
        self._validate_features(config, X, main_type)
        self._validate_normalization(config)
        self._validate_class_balance(config)
        
        # Validate scoring metrics
        if main_type == "rncv":
            self._validate_nested_cv_scoring(config)
        else:
            self._validate_basic_scoring(config)

        # Add the main_type in the configuration
        config["model_selection_type"] = main_type
            
        # Configure classifiers
        self._configure_classifiers(config)
        
        # Add dataset information
        self._add_dataset_info(config, csv_dir, label, X)
        
        return config
    
    def _get_required_keys(self, main_type: str) -> Dict[str, any]:
        """Get the required configuration keys and their default values for each validation type."""
        base_required = {
            "missing_values_method": "median",
            "rounds": 10,
            "normalization": "minmax",
            "class_balance": None,
            "feature_selection_type": "mrmr",
            "feature_selection_method": "chi2",
            "extra_metrics": ['roc_auc', 'matthews_corrcoef', 'recall', 'accuracy', 
                            'precision', 'f1', 'average_precision', 'balanced_accuracy',
                            'specificity']
        }
        
        if main_type == "rncv":
            base_required.update({
                "n_trials": 100,
                "inner_scoring": "mathews_corrcoef",
                "outer_scoring": "roc_auc",
                "inner_splits": 5,
                "outer_splits": 5,
                'inner_selection': ["validation_score", "one_sem", "gso_1", "gso_2", "one_sem_grd"]
            })
        else:
            base_required.update({
                "scoring": "roc_auc",
                "splits": 5
            })
            
        return base_required
        
    def _validate_required_keys(self, config: Dict, main_type: str) -> None:
        """
        Validate that all required keys are present in the config.
        If a required key is missing, use the default value.
        """
        required_keys = self._get_required_keys(main_type)
        
        missing_keys = []
        for key, default_value in required_keys.items():
            if key not in config:
                if default_value is None:
                    missing_keys.append(key)
                else:
                    config[key] = default_value
                    self._log_once(f"Missing required key '{key}'. Using default value: {default_value}")
                    
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
            
    def _validate_missing_values(self, config: Dict, X: pd.DataFrame) -> None:
        """Validate missing values handling method."""
        method = config["missing_values_method"]
        if method not in self.valid_missing_values:
            config["missing_values_method"] = "median"
            self._log_once("Invalid missing values method. Using default: median")
            
        if method == "drop" and X.isnull().values.any():
            config["missing_values_method"] = "median"
            self._log_once("Cannot drop missing values. Using median instead.")
            
    def _validate_features(self, config: Dict, X: pd.DataFrame, main_type: str) -> None:
        """Validate feature selection settings."""
        num_features = config.get("num_features")
        
        if main_type in ['rncv', 'rcv']:
            if isinstance(num_features, int):
                config["num_features"] = [num_features]
            elif num_features is None:
                config["num_features"] = [X.shape[1]]
            if isinstance(num_features, list):
                if None in num_features:
                    # Replace None with the number of features
                    num_features = [X.shape[1] if n is None else n for n in num_features]
                    config["num_features"] = num_features
            elif not isinstance(num_features, list):
                raise ValueError("num_features must be an integer, list, or None")
        else:
            if num_features is None or num_features > X.shape[1]:
                config["num_features"] = X.shape[1]
            elif not isinstance(num_features, int):
                raise ValueError("num_features must be an integer or None")
                
    def _validate_normalization(self, config: Dict) -> None:
        """Validate normalization method."""
        if config["normalization"] not in self.valid_normalizations:
            config["normalization"] = "minmax"
            self._log_once("Invalid normalization method. Using default: minmax")
            
    def _validate_class_balance(self, config: Dict) -> None:
        """Validate class balancing method."""
        if config["class_balance"] not in self.valid_class_balance:
            config["class_balance"] = None
            self._log_once("Invalid class balance method. Using default: None")

    def _validate_scorer(self, scoring: str) -> None:
        """Validate if a scoring metric is valid."""
        if scoring not in self.valid_scorers:
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Valid options: {self.valid_scorers}"
            )
        
    def _validate_nested_cv_scoring(self, config: Dict) -> None:
        """Validate scoring for nested cross-validation."""
        self._validate_scorer(config["outer_scoring"])
        self._validate_scorer(config["inner_scoring"])
        
        if config["outer_scoring"] not in config["extra_metrics"]:
            config["extra_metrics"].insert(0, config["outer_scoring"])
            
    def _validate_basic_scoring(self, config: Dict) -> None:
        """Validate scoring for basic cross-validation."""
        self._validate_scorer(config["scoring"])
        
        if config["scoring"] not in config["extra_metrics"]:
            config["extra_metrics"].insert(0, config["scoring"])
            
    def _configure_classifiers(self, config: Dict) -> None:
        """Configure classifier selection based on include/exclude lists."""
        include_classes = config.get("search_on")
        exclude_classes = config.get("exclude")
        if exclude_classes is None:
            exclude_classes = []

        if include_classes:
            config["clfs"] = [clf for clf in include_classes 
                            if clf in self.available_clfs]
        else:
            config["clfs"] = [clf for clf in self.available_clfs 
                            if clf not in exclude_classes]
            
    def _add_dataset_info(self, config: Dict, csv_dir: str, label: str, X: pd.DataFrame) -> None:
        """Add dataset-related information to the configuration."""
        config["csv_dir"] = csv_dir
        config["label"] = label
        
        # Extract dataset name from path
        pattern = r"[^/]+(?=\.csv)"
        match = re.search(pattern, csv_dir)
        config["dataset_plot_name"] = match.group() if match else ''
        
        # Set features information
        config["features_name"] = (None if config["num_features"] == [X.shape[1]] 
                                 else config["num_features"])
        config["all_features"] = X.shape[1]
        config["model_selection_type"] = config.get("model_selection_type", "cv")
