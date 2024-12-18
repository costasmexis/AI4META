import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    SelectPercentile,
)
from mrmr import mrmr_classif


class DataLoader:
    def __init__(self, label, csv_dir, index_col=None):
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

    def __load_data(self):
        """ Function to load data from a file to a pandas dataframe."""
        file_extension = self.csv_dir.split(".")[-1]
        if file_extension in self.supported_extensions:
            sep = "," if file_extension == "csv" else "\t"
            self.data = pd.read_csv(self.csv_dir, sep=sep, index_col=self.index_col)
        else:
            raise Exception("Unsupported file type.")

        self.X = self.data.drop(self.label, axis=1)
        self.y = self.data[self.label].copy()

    def __encode_labels(self):
        """ Function to encode the target varable. From str to 1/0. """
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.y)
        self.label_mapping = {
            index: class_label
            for index, class_label in enumerate(label_encoder.classes_)
        }

    def missing_values(self, data=None, method="median"):
        """ Function to handle missing values in the dataset 

        :param data: Dataset to handle missing values in. Defaults to None.
        :type data: pandas.DataFrame, optional
        :param method: The method to use for handling missing values. Defaults to "median".
                   Supported methods are "drop", "mean", "median", and "0".
        :type method: str, optional
        :return: The dataset with missing values handled.
        :rtype: pandas.DataFrame
        """
        initial_data = False
        if data is None:
            data = self.X
            initial_data = True
            total_missing = data.isnull().sum().sum()
            print(f"Number of missing values: {total_missing}")
        if method == "drop":
            data.dropna(inplace=True)
        elif method in ["mean", "median", "0"]:
            fill_value = 0 if method == "0" else getattr(data, method)()
            data.fillna(fill_value, inplace=True)
        elif method == None:
            pass
        else:
            raise Exception("Unsupported missing values method.")
        
        if initial_data:
            self.X = data
        else:
            return data

    def normalize(self, X=None, method="minmax", train_test_set=False, X_test=None):
        """
        Normalize the input data using the specified method.

        :param X: Input data to be normalized. If None, the data stored in self.X will be used. Defaults to None.
        :type X: pandas.DataFrame, optional
        :param method: The normalization method to be used. Supported methods are "minmax" and "standard". Defaults to "minmax".
        :type method: str, optional
        :param train_test_set: Indicates whether the normalization is applied to a train-test set. Defaults to False.
        :type train_test_set: bool, optional
        :param X_test: The test data to be normalized. Only used when train_test_set is True. Defaults to None.
        :type X_test: pandas.DataFrame, optional
        :return: The normalized data. If train_test_set is True, returns a tuple containing the normalized train and test data.
        :rtype: pandas.DataFrame or tuple
        """
        initial_data = False
        if X is None:
            X = self.X
            initial_data = True
            print(f"Converting the raw data with {method} normalization method....")
        if method in ["minmax", "standard"]:
            self.scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            if train_test_set:
                X_test = pd.DataFrame(
                    self.scaler.transform(X_test), columns=X_test.columns
                )
        elif method == None:
            pass
        else:
            raise Exception("Unsupported normalization method.")
        if train_test_set:
            pass
        else:
            print("Normalization completed.")
        if initial_data:
            self.X = X
        elif train_test_set:
            return X, X_test
        else:
            return X

    def feature_selection(
            self, X=None, y=None, method="mrmr", num_features=10, inner_method="chi2"
        ):
            """
            Perform feature selection on the dataset.

            :param X: Input features, defaults to None
            :type X: array-like, optional
            :param y: Target variable, defaults to None
            :type y: array-like, optional
            :param method: The feature selection method to use. Supported methods are 'mrmr', 'kbest', and 'percentile', defaults to 'mrmr'
            :type method: str, optional
            :param num_features: The number of features to select, defaults to 10
            :type num_features: int, optional
            :param inner_method: The scoring function to use for feature selection, defaults to 'chi2'
            :type inner_method: str, optional
            :return: The selected features if X and y are not provided, otherwise None
            :rtype: array-like or None
            """
            if method not in ["mrmr", "kbest", "percentile"]:
                raise Exception(
                    "Unsupported feature selection method. Select one of 'mrmr', 'kbest', 'percentile'"
                )
            if inner_method not in ["chi2", "f_classif", "mutual_info_classif"]:
                raise Exception(
                    "Unsupported inner method. Select one of 'chi2', 'f_classif', 'mutual_info_classif'"
                )
            if (
                method == "percentile"
                and num_features is not None
                and (num_features >= 100 or num_features <= 0)
            ):
                raise Exception(
                    "num_features for percentile option must be between 0 and 100."
                )
            datasetXy = False
            if X is None and y is None:
                X = self.X
                y = self.y
                datasetXy = True

            method_mapping = {
                "chi2": chi2,
                "f_classif": f_classif,
                "mutual_info_classif": mutual_info_classif,
            }
            scoring_function = method_mapping[inner_method]

            if method == "mrmr":
                self.selected_features = mrmr_classif(
                    X, y, K=num_features, show_progress=False
                )
                X = X[self.selected_features]

            else:
                if not isinstance(self.scaler, MinMaxScaler):
                    raise Exception("Feature selection method requires MinMaxScaler.")
                if method == "kbest":
                    X = SelectKBest(scoring_function, k=num_features).fit_transform(X, y)

                elif method == "percentile":
                    X = SelectPercentile(
                        scoring_function, percentile=num_features
                    ).fit_transform(X, y)
                else:
                    raise Exception("Unsupported feature selection method.")

            if datasetXy:
                self.X = X
                self.y = y
            else:
                return self.selected_features

    def create_test_data(self, output_file="test_data.csv"):
        """ Function to generate a test data template csv file """
        if not os.path.exists("./results"):
            os.makedirs("./results")
        test_data = pd.DataFrame(columns=self.X.columns.values)
        test_data.to_csv(f"./results/{output_file}")

    def __str__(self):
        """ Function to print the dataset information """
        return f"Number of rows: {self.data.shape[0]} \nNumber of columns: {self.data.shape[1]}"

    def __getitem__(self, idx):
        """ Function to get a sample from the dataset """
        return self.data.iloc[idx]
