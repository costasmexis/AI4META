from typing import Optional, List, Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    SelectPercentile,
)
from mrmr import mrmr_classif
import logging

class DataLoader:
    """A class for loading and preprocessing data for machine learning tasks."""

    # Class-level logging flags
    _logged_operations = {
        'missing_values': {},  # Will store method-specific logs
        'normalization': {},   # Will store method-specific logs
        'feature_selection': {},  # Will store method-specific logs
        'class_balance': {},   # Will store method-specific logs
    }

    def __init__(self, label: str, csv_dir: str, index_col: Optional[str] = None):
        """Initialize DataLoader instance."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.csv_dir = csv_dir
        self.index_col = index_col
        self.label = label
        self.data = None
        self.X = None
        self.y = None
        self.label_mapping = None
        self.selected_features = None
        self.supported_extensions = ["csv", "tsv", "txt"]
        self.scaler = None
        
        self.__load_data()
        self.__encode_labels()

    def _log_once(self, operation: str, method: str, message: str) -> None:
        """Log a message only once for a specific operation and method."""
        if method not in self._logged_operations[operation]:
            self._logged_operations[operation][method] = True
            self.logger.info(message)

    def __load_data(self) -> None:
        """Load data from file into pandas DataFrame."""        
        file_extension = self.csv_dir.split(".")[-1]
        if file_extension in self.supported_extensions:
            sep = "," if file_extension == "csv" else "\t"
            self.data = pd.read_csv(self.csv_dir, sep=sep, index_col=self.index_col)
        else:
            raise Exception(f"Unsupported file type. Supported types: {self.supported_extensions}")

        self.X = self.data.drop(self.label, axis=1)
        self.y = self.data[self.label].copy()
        
        # self.logger.info(f"✓ Data loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")

    def __encode_labels(self) -> None:
        """Encode target variable from string to numeric values."""        
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.y)
        self.label_mapping = {
            index: class_label
            for index, class_label in enumerate(label_encoder.classes_)
        }
        # self.logger.info(f"✓ Labels encoded: {len(self.label_mapping)} classes")

    def missing_values(self, data: Optional[pd.DataFrame] = None, method: str = "median") -> Optional[pd.DataFrame]:
        """Handle missing values in the dataset."""        
        initial_data = data is None
        if initial_data:
            data = self.X
            total_missing = data.isnull().sum().sum()
            if total_missing > 0:
                self._log_once('missing_values', 'count', f"Found {total_missing} missing values")

        if method == "drop":
            initial_rows = len(data)
            data.dropna(inplace=True)
            self._log_once('missing_values', method, f"✓ Dropped {initial_rows - len(data)} rows with missing values")
        elif method in ["mean", "median", "0"]:
            fill_value = 0 if method == "0" else getattr(data, method)()
            data.fillna(fill_value, inplace=True)
            self._log_once('missing_values', method, f"✓ Filled missing values using {method}")
        elif method is None:
            self._log_once('missing_values', 'none', "✓ No missing value handling applied")
        else:
            raise Exception(f"Unsupported method: {method}. Use 'drop', 'mean', 'median', '0', or None")

        if initial_data:
            self.X = data
            return None
        return data

    def normalize(
        self,
        X: Optional[pd.DataFrame] = None,
        method: str = "minmax",
        X_test: Optional[pd.DataFrame] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Normalize the input data using specified method."""        
        initial_data = X is None
        if initial_data:
            X = self.X

        if method in ["minmax", "standard"]:
            self.scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            if X_test is not None:
                X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
                self._log_once('normalization', f"{method}_train_test", f"✓ Applied {method} normalization to train and test sets")
            else:
                self._log_once('normalization', method, f"✓ Applied {method} normalization")
        elif method is None:
            self._log_once('normalization', 'none', "✓ No normalization applied")
        else:
            raise Exception(f"Unsupported method: {method}. Use 'minmax', 'standard', or None")

        if initial_data:
            self.X = X
            return None
        elif X_test is not None:
            return X, X_test
        return X

    def feature_selection(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[np.ndarray] = None,
        method: str = "mrmr",
        num_features: int = 10,
        inner_method: str = "chi2",
    ) -> Optional[List[str]]:
        """Perform feature selection on the dataset."""        
        datasetXy = X is None and y is None
        if datasetXy:
            X = self.X
            y = self.y

        initial_features = X.shape[1]

        method_mapping = {
                "chi2": chi2,
                "f_classif": f_classif,
                "mutual_info_classif": mutual_info_classif,
            }
        
        if method == "mrmr":
            self.selected_features = mrmr_classif(X, y, K=num_features, show_progress=False)
            X = X[self.selected_features]
            self._log_once('feature_selection', f"{method}_{num_features}", f"✓ Selected {num_features} features using {method}")
        else:
            # if not isinstance(self.scaler, MinMaxScaler):
            #     raise Exception("Feature selection methods require MinMaxScaler normalization")
            
            if method == "kbest":
                selector = SelectKBest(method_mapping[inner_method], k=num_features)
            elif method == "percentile":
                selector = SelectPercentile(method_mapping[inner_method], percentile=num_features)
            
            selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.get_support()].tolist()
            X = X[self.selected_features]
            self._log_once('feature_selection', f"{method}_{num_features}", f"✓ Selected {len(self.selected_features)} features using {method}")

        if datasetXy:
            self.X = X
            return None
        return self.selected_features

    def create_test_data(self, output_file: str = "test_data.csv") -> None:
        """Generate a test data template CSV file."""
        import os
        if not os.path.exists("./results"):
            os.makedirs("./results")
        test_data = pd.DataFrame(columns=self.X.columns.values)
        test_data.to_csv(f"./results/{output_file}")
        self.logger.info(f"✓ Created test data template: {output_file}")

    def __str__(self) -> str:
        """Return string representation of dataset information."""
        return f"Number of rows: {self.data.shape[0]}\nNumber of columns: {self.data.shape[1]}"

    def __getitem__(self, idx: int) -> pd.Series:
        """Get a sample from the dataset by index."""
        return self.data.iloc[idx]