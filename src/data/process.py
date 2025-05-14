# process.py
"""
Data processing module that inherits from DataLoader and implements
various data preprocessing procedures.
"""
from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
import logging

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    SelectPercentile,
)
from mrmr import mrmr_classif
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks

from src.constants.translators import SFM_COMPATIBLE_ESTIMATORS, AVAILABLE_CLFS

# Import the DataLoader class
from src.data.dataloader import DataLoader

class DataProcessor(DataLoader):
    """
    A class for processing data, including handling missing values, normalization,
    feature selection, and class balancing.

    This class inherits from DataLoader and extends its functionality with
    various data processing procedures.
    """

    # Class-level logging configuration
    _logged_operations = {
        'missing_values': {},
        'normalization': {},
        'feature_selection': {},
        'class_balance': {}
    }

    def __init__(
        self,
        label: str,
        csv_dir: str,
        index_col: Optional[str] = None,
        normalization: Optional[str] = 'minmax',
        fs_method: Optional[str] = 'mrmr',
        inner_fs_method: Optional[str] = 'chi2',
        mv_method: Optional[str] = 'median',
        class_balance_method: Optional[str] = None,
        preprocess_mode: Optional[str] = 'general'
    ) -> None:
        """
        Initialize DataProcessor instance.

        Parameters
        ----------
        label : str
            The column name of the target variable.
        csv_dir : str
            Path to the CSV file.
        index_col : str, optional
            Name of the index column, by default None.
        normalization : str, optional
            Normalization method ('minmax', 'standard', None), by default 'minmax'.
        fs_method : str, optional
            Feature selection method ('mrmr', 'kbest', 'percentile'), by default 'mrmr'.
        inner_fs_method : str, optional
            Scoring function for feature selection, by default 'chi2'.
        mv_method : str, optional
            Missing values handling method ('mean', 'median', '0', 'drop'), by default 'median'.
        class_balance_method : str, optional
            Class balancing method ('smote', 'borderline_smote', 'tomek', None), by default None.
        preprocess_mode : str, optional
            Mode of preprocessing ('general' or 'ms' for model selection), by default 'general'.
        """
        # Call the parent constructor
        super().__init__(label, csv_dir, index_col)
        
        # Set processing parameters
        self.normalization = normalization
        self.fs_method = fs_method
        self.inner_fs_method = inner_fs_method
        self.mv_method = mv_method
        self.class_balance_method = class_balance_method
        self.preprocess_mode = preprocess_mode
        self.selected_features = None

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)

        # Validate all parameters
        self._validate_parameters()

    def _log_once(self, operation: str, method: str, message: str) -> None:
        """
        Log a message only once for a specific operation and method.

        Parameters
        ----------
        operation : str
            The operation type ('missing_values', 'normalization', etc.).
        method : str
            The specific method being used.
        message : str
            The message to log.
        """
        if method not in self._logged_operations[operation]:
            self._logged_operations[operation][method] = True
            self.logger.info(message)

    def missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Parameters
        ----------
        X : DataFrame
            Input data to process.

        Returns
        -------
        DataFrame
            Data with missing values handled according to the specified method.

        Raises
        ------
        Exception
            If an unsupported method is specified.
        """
        if X is None:
            raise ValueError("Input data X cannot be None")
            
        X_result = X.copy()  # Create a copy to avoid modifying the original
        
        total_missing = X_result.isnull().sum().sum()
        if total_missing > 0:
            self._log_once('missing_values', 'count', f"Found {total_missing} missing values")

        # Apply the specified method
        if self.mv_method == "drop":
            initial_rows = len(X_result)
            X_result = X_result.dropna()
            self._log_once('missing_values', self.mv_method, f"✓ Dropped {initial_rows - len(X_result)} rows with missing values")
        elif self.mv_method in ["mean", "median", "0"]:
            if self.mv_method == "0":
                fill_value = 0
            else:
                fill_value = getattr(X_result, self.mv_method)()
            X_result = X_result.fillna(fill_value)
            self._log_once('missing_values', self.mv_method, f"✓ Filled missing values using {self.mv_method}")
        elif self.mv_method is None:
            self._log_once('missing_values', 'none', "✓ No missing value handling applied")
        else:
            raise Exception(f"Unsupported method: {self.mv_method}. Use 'drop', 'mean', 'median', '0', or None")
        
        return X_result
    
    def normalize_ms(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize both training and test data using the same transformer.

        Parameters
        ----------
        X_train : DataFrame
            Training data to normalize.
        X_test : DataFrame
            Test data to normalize using training data parameters.

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            Normalized training and test data.

        Raises
        ------
        Exception
            If an unsupported normalization method is specified.
        """
        if X_train is None or X_test is None:
            raise ValueError("Input data X_train and X_test cannot be None")
            
        # Apply the specified normalization method
        if self.normalization in ["minmax", "standard"]:
            scaler = MinMaxScaler() if self.normalization == "minmax" else StandardScaler()
            
            X_train_result = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns,
                index=X_train.index
            )
            
            # Transform test data using training parameters
            X_test_result = pd.DataFrame(
                scaler.transform(X_test), 
                columns=X_test.columns,
                index=X_test.index
            )
            
            self._log_once('normalization', f"{self.normalization}_train_test", 
                         f"✓ Applied {self.normalization} normalization to train and test sets")
        elif self.normalization is None:
            X_train_result = X_train.copy()
            X_test_result = X_test.copy()
            self._log_once('normalization', 'none', "✓ No normalization applied")
        else:
            raise Exception(f"Unsupported method: {self.normalization}. Use 'minmax', 'standard', or None")
            
        return X_train_result, X_test_result

    def normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the input data using specified method.

        Parameters
        ----------
        X : DataFrame
            Input data to normalize.

        Returns
        -------
        DataFrame
            Normalized data.

        Raises
        ------
        Exception
            If an unsupported method is specified.
        """
        if X is None:
            raise ValueError("Input data X cannot be None")
            
        # Apply the specified normalization method
        if self.normalization in ["minmax", "standard"]:
            scaler = MinMaxScaler() if self.normalization == "minmax" else StandardScaler()
            
            X_normalized = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns,
                index=X.index
            )
            
            self._log_once('normalization', self.normalization, 
                         f"✓ Applied {self.normalization} normalization")
        elif self.normalization is None:
            X_normalized = X.copy()
            self._log_once('normalization', 'none', "✓ No normalization applied")
        else:
            raise Exception(f"Unsupported method: {self.normalization}. Use 'minmax', 'standard', or None")

        return X_normalized

    def feature_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        num_features: int
    ) -> List[str]:
        """
        Perform feature selection on the dataset.

        Parameters
        ----------
        X : DataFrame
            Input features.
        y : ndarray
            Target labels.
        num_features : int
            Number of features to select.

        Returns
        -------
        List[str]
            List of selected feature names.

        Raises
        ------
        Exception
            If an unsupported method is specified or if invalid parameters are provided.
        """
        if X is None or y is None:
            raise ValueError("Input data X and y cannot be None")
            
        if num_features is None:
            raise ValueError("num_features cannot be None for feature selection")
            
        if num_features > X.shape[1]:
            self._log_once('feature_selection', 'warning', 
                         f"Requested {num_features} features but only {X.shape[1]} available. Using all features.")
            return list(X.columns)
            
        # Map scoring method names to functions
        method_mapping = {
            "chi2": chi2,
            "f_classif": f_classif,
            "mutual_info_classif": mutual_info_classif,
        }
        
        # Apply the specified feature selection method
        if self.fs_method == "mrmr":
            selected_features = mrmr_classif(X, y, K=num_features, show_progress=False)
            self._log_once('feature_selection', f"{self.fs_method}_{num_features}", 
                         f"✓ Selected {num_features} features using {self.fs_method}")
        elif self.fs_method in ["kbest", "percentile"]:
            # Check for proper normalization if using chi2
            if self.inner_fs_method == "chi2":
                # Chi2 requires non-negative features
                X_min = X.min().min()
                if X_min < 0:
                    self._log_once('feature_selection', 'warning', 
                                 "Warning: chi2 requires non-negative features. Consider using MinMaxScaler.")
            
            # Create and apply the selector
            if self.fs_method == "kbest":
                selector = SelectKBest(method_mapping[self.inner_fs_method], k=num_features)
            else:  # percentile
                selector = SelectPercentile(method_mapping[self.inner_fs_method], percentile=num_features)
            
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self._log_once('feature_selection', f"{self.fs_method}_{num_features}", 
                         f"✓ Selected {len(selected_features)} features using {self.fs_method}")
        else:
            raise Exception(f"Unsupported method: {self.fs_method}. Use 'mrmr', 'kbest', or 'percentile'")

        self.selected_features = selected_features
        return selected_features

    def class_balance_fnc(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply class balancing strategies to address imbalanced datasets.

        Parameters
        ----------
        X : DataFrame
            Feature dataset.
        y : ndarray
            Target labels.
        random_state : int, optional
            Random state for reproducibility.

        Returns
        -------
        Tuple[DataFrame, ndarray]
            Balanced feature dataset and balanced target labels.

        Raises
        ------
        ValueError
            If an unsupported balancing method is specified.
        Exception
            If an error occurs during class balancing.
        """
        if X is None or y is None:
            raise ValueError("Input data X and y cannot be None")
            
        # No class balancing if method is None
        if self.class_balance_method is None:
            self._log_once('class_balance', 'none', "✓ No class balancing requested, returning original data")
            return X, y
        
        try:
            # Select and apply the appropriate balancing method
            if self.class_balance_method == 'smote':
                balancer = SMOTE(random_state=random_state)
                self._log_once('class_balance', 'smote', "✓ Applying SMOTE oversampling...")
            elif self.class_balance_method == 'borderline_smote':
                balancer = BorderlineSMOTE(random_state=random_state)
                self._log_once('class_balance', 'borderline_smote', "✓ Applying Borderline SMOTE oversampling...")
            elif self.class_balance_method == 'tomek':
                balancer = TomekLinks()
                self._log_once('class_balance', 'tomek', "✓ Applying Tomek links undersampling...")
            else:
                raise ValueError(
                    f"Unsupported balancing method: {self.class_balance_method}. "
                    "Choose from ['smote', 'borderline_smote', 'tomek']"
                )

            # Perform resampling
            X_balanced, y_balanced = balancer.fit_resample(X, y)
            
            # Log results
            original_class_counts = np.bincount(y) if isinstance(y, np.ndarray) else y.value_counts()
            balanced_class_counts = np.bincount(y_balanced) if isinstance(y_balanced, np.ndarray) else y_balanced.value_counts()
            self._log_once('class_balance', f"{self.class_balance_method}_results", 
                         f"✓ Class balance results: Before {dict(enumerate(original_class_counts))}, After {dict(enumerate(balanced_class_counts))}")

            return X_balanced, y_balanced

        except Exception as e:
            self.logger.error(f"Error during class balancing: {str(e)}")
            raise
    
    def process_data(
        self,
        X: pd.DataFrame = None,
        y: np.ndarray = None,
        X_test: pd.DataFrame = None,
        num_features: Optional[int] = None,
        features_name_list: Optional[List[str]] = None,
        random_state: int = 42,
        sfm: Optional[bool] = False,
        estimator_name: Optional[str] = None
    ) -> Union[Tuple[pd.DataFrame, np.ndarray, Optional[str]], 
               Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Optional[str]]]:
        """
        Process data by applying preprocessing steps based on the selected mode.
        
        Parameters
        ----------
        X : DataFrame, optional
            Input features (or training features in 'ms' mode).
        y : ndarray, optional
            Target labels.
        X_test : DataFrame, optional
            Test features (used only in 'ms' mode).
        num_features : int, optional
            Number of features to select. If None, no feature selection is applied.
        features_name_list : List[str], optional
            List of feature names to use. If provided, feature selection is skipped.
        random_state : int, optional
            Random state for reproducibility.
            
        Returns
        -------
        Union[Tuple[DataFrame, ndarray, str], Tuple[DataFrame, ndarray, DataFrame, str]]
            In 'general' mode: (processed_X, processed_y, feature_indicator)
            In 'ms' mode: (processed_X_train, processed_y_train, processed_X_test, feature_indicator)
            
        Raises
        ------
        ValueError
            If required inputs are missing.
        """
        # Use instance data if none provided
        if X is None:
            if self.X is None:
                raise ValueError("Input data X cannot be None and no instance data available")
            X = self.X.copy()
            
        if y is None:
            if self.y is None:
                raise ValueError("Input data y cannot be None and no instance data available")
            y = self.y.copy()
        
        # Choose processing mode
        if self.preprocess_mode == 'general':
            return self.process_general(
                X=X, 
                y=y, 
                num_features=num_features, 
                features_name_list=features_name_list,
                random_state=random_state
            )
        else:
            if X_test is None:
                raise ValueError("X_test cannot be None when using 'ms' mode")
                
            return self.process_ms(
                X_train=X, 
                y_train=y, 
                X_test=X_test, 
                random_state=random_state, 
                num_features=num_features,
                sfm=sfm,
                estimator_name=estimator_name
            )
        
    def sfm_fs(
        self,
        estimator_name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        num_features: Optional[int],
        X_test: Optional[pd.DataFrame] = None,
        ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Perform feature selection using SelectFromModel.

        Parameters
        ----------
        estimator_name : str
            Name of the estimator to use for feature selection.
        X_train : DataFrame
            Training feature set.
        X_test : DataFrame
            Testing feature set.
        y_train : Series or ndarray
            Training target variable.
        num_features : int or None, optional
            Number of features to select. If None, threshold is used.

        Returns
        -------
        X_train_selected : DataFrame
            Training set with selected features.
        X_test_selected : DataFrame
            Testing set with selected features.
        num_features : int
            Number of features selected.
        """
        # Get the estimator
        estimator = AVAILABLE_CLFS[estimator_name]

        # Create SelectFromModel
        sfm = SelectFromModel(estimator, max_features=num_features, threshold=-np.inf)

        # Fit SelectFromModel
        sfm.fit(X_train, y_train)

        # Get selected features
        selected_features = sfm.get_support(indices=True)
        selected_columns = X_train.columns[selected_features].to_list()

        # Select features
        X_train_selected = X_train[selected_columns]
        if X_test is not None:
            X_test_selected = X_test[selected_columns]
        else:
            X_test_selected = None

        return X_train_selected, X_test_selected, len(selected_columns)

    def process_general(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        num_features: Optional[int] = None,
        features_name_list: Optional[List[str]] = None,
        random_state: int = 42,
        sfm: Optional[bool] = False,
        estimator_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[str]]:
        """
        Process data in general mode (single dataset).
        
        Parameters
        ----------
        X : DataFrame
            Input features.
        y : ndarray
            Target labels.
        num_features : int, optional
            Number of features to select.
        features_name_list : List[str], optional
            List of feature names to use instead of feature selection.
        random_state : int, optional
            Random state for reproducibility.
            
        Returns
        -------
        Tuple[DataFrame, ndarray, str]
            Processed features, processed labels, and feature selection indicator.
        """
        # Feature pre-selection by list if provided
        if features_name_list is not None:
            X_selected = X[features_name_list]
            self._log_once('feature_selection', 'list', 
                         f"✓ Selected {len(features_name_list)} features from provided list")
        else: 
            X_selected = X

        # Normalize the data
        X_normalized = self.normalize(X_selected)
        
        # Handle missing values
        X_cleaned = self.missing_values(X_normalized)
        
        # Feature selection
        if features_name_list is not None:
            feature_indicator = len(features_name_list)
        elif (num_features == X.shape[1]) or \
              ((self.fs_method == 'percentile') and (num_features == 100)):
            self._log_once('feature_selection', 'none', 
                         "✓ No feature selection needed - using all features")
            feature_indicator = 'none'
        elif sfm and (num_features != X_cleaned.shape[1]) and (estimator_name in SFM_COMPATIBLE_ESTIMATORS):
            # Use SelectFromModel for feature selection
            X_cleaned, _, feature_indicator = self.sfm_fs(
                estimator_name=estimator_name,
                X_train=X_cleaned,
                X_test=None,
                y_train=y,
                num_features=num_features
            )
            self._log_once('feature_selection', 'sfm', 
                         f"✓ Selected {num_features} features using SFM with {estimator_name}")
        else:
            selected_features = self.feature_selection(
                X=X_cleaned,
                y=y,
                num_features=num_features
            )
            X_cleaned = X_cleaned[selected_features]
            feature_indicator = num_features
        
        # Apply class balancing
        if self.class_balance_method is not None:
            X_cleaned, y = self.class_balance_fnc(
                X=X_cleaned, y=y, random_state=random_state
            )

        return X_cleaned, y, feature_indicator

    def process_ms(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        random_state: int,
        num_features: Optional[int] = None,
        sfm: Optional[bool] = False,
        estimator_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Optional[str]]:
        """
        Process data in model selection mode (train/test split).
        
        Parameters
        ----------
        X_train : DataFrame
            Training features.
        y_train : ndarray
            Training labels.
        X_test : DataFrame
            Test features.
        random_state : int
            Random state for reproducibility.
        num_features : int, optional
            Number of features to select.
        sfm : bool, optional
            Whether to use SelectFromModel for feature selection.
        estimator : str, optional
            The estimator to use for feature selection if sfm is True.
            
        Returns
        -------
        Tuple[DataFrame, ndarray, DataFrame, str]
            Processed training features, processed training labels, 
            processed test features, and feature selection indicator.
        """
        # Normalize both training and test data
        X_train_normalized, X_test_normalized = self.normalize_ms(X_train, X_test)
        
        # Handle missing values
        X_train_cleaned = self.missing_values(X_train_normalized)
        X_test_cleaned = self.missing_values(X_test_normalized)

        # Feature selection
        if (num_features == X_train.shape[1]) or \
            ((self.fs_method == 'percentile') and (num_features == 100)):
            self._log_once('feature_selection', 'none',
                         "✓ No feature selection needed - using all features")
            feature_indicator = 'none'
        elif sfm and (num_features != X_train.shape[1]) and (estimator_name in SFM_COMPATIBLE_ESTIMATORS):
            # Use SelectFromModel for feature selection
            X_train_cleaned, X_test_cleaned, feature_indicator = self.sfm_fs(
                estimator_name=estimator_name,
                X_train=X_train_cleaned,
                X_test=X_test_cleaned,
                y_train=y_train,
                num_features=num_features
            )
            self._log_once('feature_selection', 'sfm', 
                         f"✓ Selected {num_features} features using SFM with {estimator_name}")
        else:
            selected_features = self.feature_selection(
                X=X_train_cleaned,
                y=y_train,
                num_features=num_features
            )
            X_train_cleaned = X_train_cleaned[selected_features]
            X_test_cleaned = X_test_cleaned[selected_features]
            feature_indicator = num_features

        # Apply class balancing to training data only
        if self.class_balance_method is not None:
            X_train_cleaned, y_train = self.class_balance_fnc(
                X=X_train_cleaned, 
                y=y_train, 
                random_state=random_state
            )     
            
        return X_train_cleaned, y_train, X_test_cleaned, feature_indicator
    
    def _validate_parameters(self):
        """
        Validate all parameters passed to the DataProcessor.
        
        This method checks that all configuration parameters have valid values
        and raises informative error messages if any invalid values are found.
        
        Raises
        ------
        ValueError
            If any parameter has an invalid value.
        """
        self._validate_normalization(self.normalization)
        self._validate_fs_method(self.fs_method)
        self._validate_inner_fs_method(self.inner_fs_method)
        self._validate_mv_method(self.mv_method)
        self._validate_class_balance(self.class_balance_method)
        self._validate_preprocess_mode(self.preprocess_mode)

    def _validate_normalization(self, method):
        """
        Validate the normalization method.
        
        Parameters
        ----------
        method : str or None
            The normalization method to validate.
            
        Raises
        ------
        ValueError
            If the method is not one of the valid options.
        """
        valid_methods = ["minmax", "standard", None]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid normalization method: '{method}'. "
                f"Valid options are: {', '.join([str(m) for m in valid_methods])}"
            )

    def _validate_fs_method(self, method):
        """
        Validate the feature selection method.
        
        Parameters
        ----------
        method : str or None
            The feature selection method to validate.
            
        Raises
        ------
        ValueError
            If the method is not one of the valid options.
        """
        valid_methods = ["mrmr", "kbest", "percentile", None]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid feature selection method: '{method}'. "
                f"Valid options are: {', '.join([str(m) for m in valid_methods])}"
            )

    def _validate_inner_fs_method(self, method):
        """
        Validate the inner feature selection scoring method.
        
        Parameters
        ----------
        method : str or None
            The inner feature selection scoring method to validate.
            
        Raises
        ------
        ValueError
            If the method is not one of the valid options.
        """
        valid_methods = ["chi2", "f_classif", "mutual_info_classif", None]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid inner feature selection method: '{method}'. "
                f"Valid options are: {', '.join([str(m) for m in valid_methods])}"
            )

    def _validate_mv_method(self, method):
        """
        Validate the missing values handling method.
        
        Parameters
        ----------
        method : str or None
            The missing values handling method to validate.
            
        Raises
        ------
        ValueError
            If the method is not one of the valid options.
        """
        valid_methods = ["mean", "median", "0", "drop", None]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid missing values method: '{method}'. "
                f"Valid options are: {', '.join([str(m) for m in valid_methods])}"
            )

    def _validate_class_balance(self, method):
        """
        Validate the class balancing method.
        
        Parameters
        ----------
        method : str or None
            The class balancing method to validate.
            
        Raises
        ------
        ValueError
            If the method is not one of the valid options.
        """
        valid_methods = ["smote", "borderline_smote", "tomek", None]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid class balancing method: '{method}'. "
                f"Valid options are: {', '.join([str(m) for m in valid_methods])}"
            )

    def _validate_preprocess_mode(self, mode):
        """
        Validate the preprocessing mode.
        
        Parameters
        ----------
        mode : str
            The preprocessing mode to validate.
            
        Raises
        ------
        ValueError
            If the mode is not one of the valid options.
        """
        valid_modes = ["general", "ms"]
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid preprocessing mode: '{mode}'. "
                f"Valid options are: {', '.join(valid_modes)}"
            )