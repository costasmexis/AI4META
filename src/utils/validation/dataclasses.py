# In src/utils/validation/config_types.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
import re
import os
import logging
import pandas as pd
from datetime import datetime
from sklearn.metrics import get_scorer_names
from src.constants.translators import DEFAULT_CONFIG_EVAL, DEFAULT_CONFIG_MS
from src.constants.translators import AVAILABLE_CLFS
from src.constants.parameters_grid import optuna_grid


@dataclass
class ModelSelectionConfig:
    """Data class for model selection configuration with built-in validation"""
    
    # Model selection parameters with default values
    model_selection_type: str = 'both'
    normalization: str = 'minmax'
    class_balance: str = None
    feature_selection_method: str = 'chi2'
    feature_selection_type: str = 'mrmr'
    missing_values_method: str = 'median'
    rounds: int = 10
    exclude: Optional[List[str]] = None
    search_on: Optional[List[str]] = None
    num_features: Optional[Union[int, List[int]]] = None
    return_csv: bool = True
    plot: Optional[str] = None
    scoring: str = "roc_auc"
    inner_scoring: str = "matthews_corrcoef"
    splits: int = 5
    inner_splits: int = 5
    freq_feat: Optional[int] = None
    inner_selection: List[str] = field(default_factory=lambda: ["validation_score", "one_sem", "gso_1", "gso_2", "one_sem_grd"])
    sfm: bool = False
    extra_metrics: List[str] = field(default_factory=lambda: ['roc_auc','recall', 'matthews_corrcoef', 'accuracy', 'balanced_accuracy',  
                           'precision', 'f1', 'average_precision', 'specificity'])
    info_to_db: bool = False
    filter_csv: Optional[Dict] = None
    parallel: str = "thread_per_round"
    n_trials: int = 100
    
    # Derived attributes (will be set during validation)
    clfs: Optional[List[str]] = None
    dataset_name: Optional[str] = None
    all_features: Optional[int] = None
    dataset_csv_name: Optional[str] = None
    dataset_histogram_name: Optional[str] = None
    dataset_plot_name: Optional[str] = None
    dataset_json_name: Optional[str] = None
    
    # Validation helper variables
    _logger: Optional[logging.Logger] = field(default=None, repr=False)
    _logged_messages: set = field(default_factory=set, repr=False)
    
    def validate(self, X: pd.DataFrame, csv_dir: str) -> 'ModelSelectionConfig':
        """
        Validate this configuration instance against the dataset and return self.
        
        Parameters
        ----------
        X : DataFrame
            Input features for validation
        csv_dir : str
            Path to the CSV file
            
        Returns
        -------
        ModelSelectionConfig
            Self, after validation (for method chaining)
        """
        # Setup logging if not already done
        if self._logger is None:
            self._logger = logging.getLogger(__name__)
        
        # Validate model selection type
        self._validate_model_selection_type()
        
        # Validate number of features
        self._validate_num_features(X)
        
        # Validate scoring metrics
        self._validate_scoring_metrics()
        
        # Validate parallel option
        self._validate_parallel()
        
        # Configure classifiers
        self._configure_classifiers()

        # Validate SFM (Sequential Feature Selection) setting
        self._validate_sfm()
        
        # Extract dataset info
        self._extract_dataset_info(X, csv_dir)

        # Set result names
        self._set_result_names()
        
        return self
    
    def _log_once(self, message: str, level: str = 'warning') -> None:
        """Log a message only if it hasn't been logged before."""
        if message not in self._logged_messages:
            if level == 'warning':
                self._logger.warning(message)
            elif level == 'error':
                self._logger.error(message)
            elif level == 'info':
                self._logger.info(message)
            self._logged_messages.add(message)
    
    def _validate_model_selection_type(self) -> None:
        """Validate model selection type"""
        valid_model_types = ["rcv_accel", "rnested_cv", "both"]
        if self.model_selection_type not in valid_model_types:
            self._log_once(f"Invalid model selection type: {self.model_selection_type}. Using default: both")
            self.model_selection_type = 'both'
    
    def _validate_num_features(self, X: pd.DataFrame) -> None:
        """Validate number of features"""
        if isinstance(self.num_features, int):
            self.num_features = [self.num_features]
        elif self.num_features is None:
            self.num_features = [X.shape[1]]
        elif isinstance(self.num_features, list):
            if not all(isinstance(n, (int, type(None))) for n in self.num_features):
                self._log_once("num_features list must contain only integers or None. Using all features.")
                raise ValueError("num_features list must contain only integers or None.")
            self.num_features = [X.shape[1] if n is None else n for n in self.num_features]
        else:
            self._log_once("num_features must be an int, list of ints, or None. Using all features.")
            raise ValueError("num_features must be an int, list of ints, or None.")
    
    def _validate_scoring_metrics(self) -> None:
        """Validate scoring metrics"""
        valid_scorers = list(get_scorer_names()) + ["specificity"]
        
        if self.scoring not in valid_scorers:
            raise ValueError(f"Invalid scoring metric: {self.scoring}. "
                           f"Valid options: {valid_scorers}")
        
        if self.inner_scoring not in valid_scorers:
            raise ValueError(f"Invalid inner scoring metric: {self.inner_scoring}. "
                            f"Valid options: {valid_scorers}")
            
        # Ensure outer_scoring is in extra_metrics
        if self.scoring not in self.extra_metrics:
            self.extra_metrics.insert(0, self.scoring)
    
    def _validate_parallel(self) -> None:
        """Validate parallel option"""
        valid_parallel_options = ["thread_per_round", "freely_parallel"]
        if self.parallel not in valid_parallel_options:
            self._log_once(f"Invalid parallel value: {self.parallel}. Using default: thread_per_round")
            self.parallel = "thread_per_round"

    def _configure_classifiers(self) -> None:
        """Configure classifier selection based on include/exclude lists"""
        if self.exclude is None:
            self.exclude = []
            
        if self.search_on:
            self.clfs = [clf for clf in self.search_on if clf in AVAILABLE_CLFS]
        else:
            self.clfs = [clf for clf in AVAILABLE_CLFS if clf not in self.exclude]
    
    def _validate_sfm(self) -> None:
        """Validate SFM (Sequential Feature Selection) setting"""
        if self.sfm not in [True, False]:
            self._log_once(f"Invalid SFM value: {self.sfm}. Using default: False")
            self.sfm = False
    
    def _extract_dataset_info(self, X: pd.DataFrame, csv_dir: str) -> None:
        """Extract and store dataset-related information"""
        # Extract dataset name from path
        pattern = r"[^/]+(?=\.csv)"
        match = re.search(pattern, csv_dir)
        self.dataset_name = match.group() if match else ''
        
        # Set features information
        self.all_features = X.shape[1]

    def _extract_final_name(self) -> str:
        """
        Extract the final name for plots, CSVs, etc., including non-default parameter values.
        
        Returns
        -------
        str
            Formatted name including dataset name and non-default parameters
        """
        # Start with dataset name
        final_name = self.dataset_name
        
        # Get default values from constants
        defaults = DEFAULT_CONFIG_MS

        # Add non-default configurations
        for param, value in defaults.items():
            if param in vars(self) and getattr(self, param) != value:
                final_name += f"_{param}_{getattr(self, param)}"

        # Add timestamp to ensure uniqueness
        final_name += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # Add the type of model selection
        if self.model_selection_type == "both":
            final_name += "_MS_BOTH"
        elif self.model_selection_type == "rcv_accel":
            final_name += "_MS_RCV"
        elif self.model_selection_type == "rnested_cv":
            final_name += "_MS_RCV_NESTED"

        return final_name

    def _set_result_names(self) -> None:
        # Get the base name
        base_name = self._extract_final_name()
        
        # CSV results directory
        results_csv_dir = "results/csv/"
        os.makedirs(results_csv_dir, exist_ok=True)
        self.dataset_csv_name = os.path.join(results_csv_dir, f"{base_name}.csv")
        
        # Image results directory
        results_image_dir = "results/images/"
        os.makedirs(results_image_dir, exist_ok=True)
        self.dataset_plot_name = os.path.join(results_image_dir, f"{base_name}_plot.png")
        
        # Histogram results directory
        results_hist_dir = "results/histograms/"
        os.makedirs(results_hist_dir, exist_ok=True)
        self.dataset_histogram_name = os.path.join(results_hist_dir, f"{base_name}_histogram")

        # Json for feature importance
        results_json_dir = "results/json/"
        os.makedirs(results_json_dir, exist_ok=True)
        self.dataset_json_name = os.path.join(results_json_dir, f"{base_name}_frfs.json")

@dataclass
class ModelTuning:
    """Data class for model tuning configuration"""
    estimator_name: str = None
    search_type: str = 'bayesian'
    normalization: str = 'minmax'
    class_balance: str = None
    missing_values_method: str = 'median'
    # feature_selection_method: str = 'chi2'
    # feature_selection_type: str = 'mrmr'
    scoring: str = "matthews_corrcoef"
    splits: int = 5
    inner_selection: str = "validation_score"
    info_to_db: bool = False
    n_trials: int = 100
    processors: int = -1
    # save_model: bool = True
    param_grid: Optional[Dict] = None
    features_name_list: Optional[List[str]] = None

    # Derived attributes (will be set during validation)
    dataset_name: Optional[str] = None
    model_path: Optional[str] = None
    params_path: Optional[str] = None
    metadata_path: Optional[str] = None

   # Validation helper variables
    _logger: Optional[logging.Logger] = field(default=None, repr=False)
    _logged_messages: set = field(default_factory=set, repr=False)

    def validate(self, X: pd.DataFrame, csv_dir: str) -> 'ModelEvaluationConfig':
        """
        Validate this configuration instance against the dataset and return self.
        
        Parameters
        ----------
        X : DataFrame
            Input features for validation
        csv_dir : str
            Path to the CSV file
            
        Returns
        -------
        ModelEvaluationConfig
            Self, after validation (for method chaining)
        """
        # Setup logging if not already done
        if self._logger is None:
            self._logger = logging.getLogger(__name__)
        
        # Validate search type
        self._validate_search_type()
        
        # Validate estimator name
        self._validate_estimator_name()
        
        # Validate inner selection method
        self._validate_inner_selection()
        
        # Validate processors
        self._validate_processors()

        # Validate parameter grid
        self._validate_param_grid()

        # Extract dataset info
        self._extract_dataset_info(X, csv_dir)

        # Set result names
        self._set_result_names()

        # Validate feature names list
        self._validate_features_name_list(X)
        
        return self
    
    def _log_once(self, message: str, level: str = 'warning') -> None:
        """Log a message only if it hasn't been logged before."""
        if message not in self._logged_messages:
            if level == 'warning':
                self._logger.warning(message)
            elif level == 'error':
                self._logger.error(message)
            elif level == 'info':
                self._logger.info(message)
            self._logged_messages.add(message)
    
    def _validate_search_type(self) -> None:
        """Validate model evaluation type"""
        valid_evaluation_types = ["bayesian", "random", "grid"]
        if self.search_type not in valid_evaluation_types:
            self._log_once(f"Invalid model evaluation type: {self.search_type}. Using default: bayesian")
            self.search_type = 'bayesian'
    
    def _validate_estimator_name(self) -> None:
        """Validate estimator name"""
        if self.estimator_name is None:
            self._log_once("No estimator specified. You must provide an estimator name.", level='error')
            raise ValueError("No estimator specified. You must provide an estimator name.")
        
        if self.estimator_name not in AVAILABLE_CLFS:
            self._log_once(f"Invalid estimator name: {self.estimator_name}. "
                            f"Valid options: {list(AVAILABLE_CLFS.keys())}", level='error')
            raise ValueError(f"Invalid estimator name: {self.estimator_name}. "
                            f"Valid options: {list(AVAILABLE_CLFS.keys())}")
        else: 
            self.estimator = AVAILABLE_CLFS[self.estimator_name]
    
    def _validate_inner_selection(self) -> None:
        """Validate inner selection method"""
        valid_inner_selection_methods = ["validation_score", "one_sem", "gso_1", "gso_2", "one_sem_grd"]
        if self.inner_selection not in valid_inner_selection_methods:
            self._log_once(f"Invalid inner selection method: {self.inner_selection}. Using default: validation_score")
            self.inner_selection = "validation_score"
    
    def _validate_processors(self) -> None:
        """Validate processors setting"""
        import multiprocessing
        max_processors = multiprocessing.cpu_count()
        
        if self.processors == -1:
            self.processors = max_processors
        elif self.processors < 1:
            self._log_once(f"Invalid processors value: {self.processors}. Must be >= 1 or -1. Using 1.")
            self.processors = 1
        elif self.processors > max_processors:
            self._log_once(f"Requested {self.processors} processors, but only {max_processors} are available. "
                            f"Using {max_processors}.")
            self.processors = max_processors
    
    def _validate_param_grid(self) -> None:
        """Validate parameter grid"""
        if self.param_grid is not None:
            if not isinstance(self.param_grid, dict):
                self._log_once("param_grid must be a dictionary. Using default: None")
                raise ValueError("param_grid must be a dictionary.")
            else:
                self.param_grid = self.param_grid
        else:
            if self.search_type == 'bayesian':
                self.param_grid = optuna_grid["NestedCV"][self.estimator_name]
            else:
                self.param_grid = optuna_grid["SklearnParameterGrid"][self.estimator_name]

    def _validate_features_name_list(self, X: pd.DataFrame) -> None:
        """Validate features names list"""
        if self.features_name_list is not None:
            # Check if all feature names exist in the dataset
            invalid_features = [f for f in self.features_name_list if f not in X.columns]
            if invalid_features:
                self._log_once(f"Invalid feature names: {invalid_features}. These features do not exist in the dataset.")
                # Remove invalid features
                self.features_name_list = [f for f in self.features_name_list if f in X.columns]
                
            # Check if the list is empty after removing invalid features
            if not self.features_name_list:
                self._log_once("No valid features in features_name_list. Using all features.")
                self.features_name_list = None

    def _extract_dataset_info(self, X: pd.DataFrame, csv_dir: str) -> None:
        """Extract and store dataset-related information"""
        # Extract dataset name from path
        pattern = r"[^/]+(?=\.csv)"
        match = re.search(pattern, csv_dir)
        self.dataset_name = match.group() if match else ''
        
        # Set features information
        self.all_features = X.shape[1]

    def _extract_final_name(self) -> str:
        """
        Extract the final name for plots, CSVs, etc., including non-default parameter values.
        
        Returns
        -------
        str
            Formatted name including dataset name and non-default parameters
        """
        # Start with dataset name and estimator
        final_name = f"{self.dataset_name}_{self.estimator_name}"
        
        # Get default values from constants (assuming DEFAULT_CONFIG_EVAL exists)
        defaults = DEFAULT_CONFIG_EVAL

        # Add non-default configurations
        for param, value in defaults.items():
            if param in vars(self) and getattr(self, param) != value:
                param_value = getattr(self, param)
                # Handle special cases like lists, dicts
                if isinstance(param_value, list):
                    param_value = "_".join(str(x) for x in param_value[:3])  # Limit to first 3 elements
                    if len(getattr(self, param)) > 3:
                        param_value += "_etc"
                elif isinstance(param_value, dict):
                    param_value = "custom_dict"
                
                final_name += f"_{param}_{param_value}"

        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_name += f'_{timestamp}'

        # Add tuning type
        final_name += f"_TUNE_{self.search_type.upper()}"
        
        return final_name

    def _set_result_names(self) -> None:
        """Set paths for output files"""
        # Get the base name
        base_name = self._extract_final_name()
        
        # Model directory for saving trained models
        # if self.save_model:
        model_dir = "results/models/"
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, f"{base_name}_model.pkl")
        self.params_path = os.path.join(model_dir, f"{base_name}_params.json")
        self.metadata_path = os.path.join(model_dir, f"{base_name}_metadata.json")

@dataclass
class ModelEvaluationConfig:
    """Data class for model evaluation configuration with built-in validation"""
    
    # Model selection parameters with default values
    estimator_name: str = None
    model_path: str = None
    evaluation: str = 'cv_rounds'
    normalization: str = 'minmax'
    class_balance: str = None
    feature_selection_method: str = 'chi2'
    missing_values_method: str = 'median'
    rounds: int = 20
    splits: int = 5
    # scoring: str = "matthews_corrcoef"
    extra_metrics: List[str] = field(default_factory=lambda: ['roc_auc','recall', 'matthews_corrcoef', 'accuracy', 'balanced_accuracy',  
                           'precision', 'f1', 'average_precision', 'specificity'])
    info_to_db: bool = False
    calculate_shap: bool = False
    boxplot: bool = False
    features_name_list: Optional[List[str]] = None

    # Derived attributes (will be set during validation)
    dataset_name: Optional[str] = None
    dataset_csv_name: Optional[str] = None
    dataset_plot_name: Optional[str] = None

   # Validation helper variables
    _logger: Optional[logging.Logger] = field(default=None, repr=False)
    _logged_messages: set = field(default_factory=set, repr=False)
   
    def validate(self, X: pd.DataFrame, csv_dir: str) -> 'ModelEvaluationConfig':
        """
        Validate this configuration instance against the dataset and return self.
        
        Parameters
        ----------
        X : DataFrame
            Input features for validation
        csv_dir : str
            Path to the CSV file
            
        Returns
        -------
        ModelEvaluationConfig
            Self, after validation (for method chaining)
        """
        # Setup logging if not already done
        if self._logger is None:
            self._logger = logging.getLogger(__name__)

        # Validate model path
        self._validate_model_path()
        
        # # Validate scoring metrics
        # self._validate_scoring_metrics()
        
        # Validate evaluation method
        self._validate_evaluation_method()
        
        # Extract dataset info
        self._extract_dataset_info(X, csv_dir)

        # Set result names
        self._set_result_names()

        # Validate feature names list
        self._validate_features_name_list(X)
        
        return self
    
    def _log_once(self, message: str, level: str = 'warning') -> None:
        """Log a message only if it hasn't been logged before."""
        if message not in self._logged_messages:
            if level == 'warning':
                self._logger.warning(message)
            elif level == 'error':
                self._logger.error(message)
            elif level == 'info':
                self._logger.info(message)
            self._logged_messages.add(message)

    def _validate_model_path(self) -> None:
        """Validate model path"""
        if self.model_path is None:
            self._log_once("No model path specified. You must provide a model path.", level='error')
            raise ValueError("No model path specified. You must provide a model path.")
        
        if not os.path.exists(self.model_path):
            self._log_once(f"Model file does not exist: {self.model_path}.", level='error')
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}.")
    
    # def _validate_scoring_metrics(self) -> None:
    #     """Validate scoring metrics"""
    #     valid_scorers = list(get_scorer_names()) + ["specificity"]
        
    #     if self.scoring not in valid_scorers:
    #         raise ValueError(f"Invalid scoring metric: {self.scoring}. "
    #                         f"Valid options: {valid_scorers}")
            
    #     # Ensure scoring is in extra_metrics
    #     if self.scoring not in self.extra_metrics:
    #         self.extra_metrics.insert(0, self.scoring)
            
    #     # Validate all metrics in extra_metrics
    #     for metric in self.extra_metrics:
    #         if metric not in valid_scorers:
    #             self._log_once(f"Invalid metric in extra_metrics: {metric}. Removing it.")
    #             self.extra_metrics.remove(metric)
    
    def _validate_evaluation_method(self) -> None:
        """Validate evaluation method"""
        valid_evaluation_methods = ["cv_rounds", "bootstrap", "oob", "train_test"]
        if self.evaluation not in valid_evaluation_methods:
            self._log_once(f"Invalid evaluation method: {self.evaluation}. Using default: cv_rounds")
            self.evaluation = "cv_rounds"

    def _validate_features_name_list(self, X: pd.DataFrame) -> None:
        """Validate features names list"""
        if self.features_name_list is not None:
            # Check if all feature names exist in the dataset
            invalid_features = [f for f in self.features_name_list if f not in X.columns]
            if invalid_features:
                self._log_once(f"Invalid feature names: {invalid_features}. These features do not exist in the dataset.")
                # Remove invalid features
                self.features_name_list = [f for f in self.features_name_list if f in X.columns]
                
            # Check if the list is empty after removing invalid features
            if not self.features_name_list:
                self._log_once("No valid features in features_name_list. Using all features.")
                self.features_name_list = None

    def _extract_dataset_info(self, X: pd.DataFrame, csv_dir: str) -> None:
        """Extract and store dataset-related information"""
        # Extract dataset name from path
        pattern = r"[^/]+(?=\.csv)"
        match = re.search(pattern, csv_dir)
        self.dataset_name = match.group() if match else ''
        
        # Set features information
        self.all_features = X.shape[1]

    def _extract_final_name(self) -> str:
        """
        Extract the final name for plots, CSVs, etc., including non-default parameter values.
        
        Returns
        -------
        str
            Formatted name including dataset name and non-default parameters
        """

        # Try to extract estimator_name from model_path using AVAILABLE_CLFS and regex
        estimator_name = None
        if self.model_path:
            for clf_name in AVAILABLE_CLFS:
                # Use word boundaries to avoid partial matches
                pattern = rf"\b{re.escape(clf_name)}\b"
                if re.search(pattern, self.model_path):
                    estimator_name = clf_name
                    break

        # Start with dataset name and estimator
        final_name = f"{self.dataset_name}_{estimator_name}"
        
        # Get default values from constants (assuming DEFAULT_CONFIG_EVAL exists)
        defaults = DEFAULT_CONFIG_EVAL

        # Add non-default configurations
        for param, value in defaults.items():
            if param in vars(self) and getattr(self, param) != value:
                param_value = getattr(self, param)
                # Handle special cases like lists, dicts
                if isinstance(param_value, list):
                    param_value = "_".join(str(x) for x in param_value[:3])  # Limit to first 3 elements
                    if len(getattr(self, param)) > 3:
                        param_value += "_etc"
                elif isinstance(param_value, dict):
                    param_value = "custom_dict"
                
                final_name += f"_{param}_{param_value}"

        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_name += f'_{timestamp}'

        # Add evaluation type
        final_name += f"_EVAL_{self.evaluation.upper()}"
        
        return final_name

    def _set_result_names(self) -> None:
        """Set paths for output files"""
        # Get the base name
        base_name = self._extract_final_name()
        
        # CSV results directory
        results_csv_dir = "results/csv/model_evaluation/"
        os.makedirs(results_csv_dir, exist_ok=True)
        self.dataset_csv_name = os.path.join(results_csv_dir, f"{base_name}.csv")
        
        # Image results directory
        results_image_dir = "results/images/model_evaluation/"
        os.makedirs(results_image_dir, exist_ok=True)
        self.dataset_plot_name = os.path.join(results_image_dir, f"{base_name}_plot.png")
