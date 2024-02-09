import pandas as pd
from mrmr import mrmr_classif
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class DataLoader:
    """Load a metabolomics dataset from a file.

    :param label: Name of the target column
    :type label: str
    :param csv_dir: Path to the csv file
    :type csv_dir: str
    """

    def __init__(self, label: str, csv_dir: str):
        self.csv_dir = csv_dir
        self.label = label
        self.data = None
        self.X = None
        self.y = None
        self.label_mapping = None
        self.selected_features = None
        self.scaler = None
        self.__load_data()
        self.__encode_labels()

    def __load_data(self, index_col=0):
        """Load data from file

        :param index_col: Index column of dataset, defaults to 0
        :type index_col: int, optional
        """
        supported_extensions = ["csv", "tsv", "txt"]
        file_extension = self.csv_dir.split(".")[-1]
        if file_extension not in supported_extensions:
            raise Exception("Unsupported file type.")
        sep = "," if file_extension == "csv" else "\t"
        self.data = pd.read_csv(self.csv_dir, sep=sep, index_col=index_col)
        self.X = self.data.drop(self.label, axis=1)
        self.y = self.data[self.label].copy()

    def __encode_labels(self):
        """Function to encode the target variable. From str to 1/0."""
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.data[self.label])
        self.label_mapping = dict(
            zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        )
        print("Label mapping:", self.label_mapping)

    def missing_values(self, method="drop"):
        """Handle missing values in the dataset

        :param method: Method to use, defaults to ``drop``
        :type method: str, optional
        """
        total_missing = self.data.isnull().sum().sum()
        print(f"Number of missing values: {total_missing}")
        if method == "drop":
            self.data.dropna(inplace=True)
        elif method in ["mean", "median"]:
            fill_value = self.data.mean() if method == "mean" else self.data.median()
            self.data.fillna(fill_value, inplace=True)
        else:
            raise Exception("Unsupported missing values method.")

    def normalize(self, method="minmax"):
        """Normalize the dataset

        :param method: Method to use for normalization, defaults to ``minmax``
        :type method: str, optional
        """
        if method in ["minmax", "standard"]:
            self.scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
            self.X = pd.DataFrame(
                self.scaler.fit_transform(self.X), columns=self.X.columns
            )
        else:
            raise Exception("Unsupported normalization method.")

    def feature_selection(
        self, method="mrmr", n_features=10, inner_method="chi2", percentile=10
    ):
        """Feature selection method

        :param method: Feature selection method, defaults to ``mrmr``
        :type method: str, optional
        :param n_features: Number of features to be selected, defaults to 10
        :type n_features: int, optional
        :param inner_method: Inner method for *SelectKBest*, defaults to ``chi2``
        :type inner_method: str, optional
        :param percentile: Percentile for *SelectPercentile*, defaults to 10
        :type percentile: int, optional
        """
        if method not in ["mrmr", "kbest", "percentile"]:
            raise Exception(
                "Unsupported feature selection method. Select one of 'mrmr', 'kbest', 'percentile'"
            )
        if inner_method not in ["chi2", "f_classif", "mutual_info_classif"]:
            raise Exception(
                "Unsupported inner method. Select one of 'chi2', 'f_classif', 'mutual_info_classif'"
            )
        if percentile > 100 or percentile < 0:
            raise Exception("Percentile must be between 0 and 100.")

        method_mapping = {
            "chi2": chi2,
            "f_classif": f_classif,
            "mutual_info_classif": mutual_info_classif,
        }
        scoring_function = method_mapping[inner_method]

        if method == "mrmr":
            self.selected_features = mrmr_classif(self.X, self.y.values, K=n_features)
            self.X = self.X[self.selected_features]
        else:
            if not isinstance(self.scaler, MinMaxScaler):
                raise Exception("Feature selection method requires MinMaxScaler.")
            if method == "kbest":
                selector = SelectKBest(scoring_function, k=n_features)
            elif method == "percentile":
                selector = SelectPercentile(scoring_function, percentile=percentile)
            self.X = selector.fit_transform(self.X, self.y)

    def create_test_data(self, output_dir="./test_data.csv"):
        """Create a .csv file for test data

        :param output_dir: Path to file, defaults to ``./test_data.csv``
        :type output_dir: str, optional
        """
        test_data = pd.DataFrame(columns=self.X.columns.values)
        test_data.to_csv(output_dir)

    def __str__(self):
        """Function to print the dataset information."""
        return f"Number of rows: {self.data.shape[0]} \nNumber of columns: {self.data.shape[1]}"

    def __getitem__(self, idx):
        """Function to get a sample from the dataset."""
        return self.data.iloc[idx]
