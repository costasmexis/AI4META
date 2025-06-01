from typing import Optional, List, Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

class DataLoader:
    """
    DataLoader is a utility class for loading and preprocessing tabular data from CSV, TSV, or TXT files.
    Attributes:
        csv_dir (str): Path to the data file.
        index_col (Optional[str]): Column to use as the row labels of the DataFrame.
        label (str): Name of the target column.
    Methods:
        __init__(label, csv_dir, index_col=None):
            Initializes the DataLoader, loads the data, and encodes the labels.
        __load_data():
            Loads data from the specified file into a pandas DataFrame, splitting into features (X) and target (y).
        __encode_labels():
            Encodes the target variable (y) from string labels to numeric values using LabelEncoder, and stores the mapping.
    """

    def __init__(self, 
                 label: str, 
                 csv_dir: str, 
                 index_col: Optional[str] = None
            ) -> None:
        
        self.csv_dir = csv_dir
        self.index_col = index_col
        self.label = label
        self.X = None
        self.y = None
        self.label_mapping = None
        
        self.__load_data()
        self.__encode_labels()

    def __load_data(
            self
        ) -> None:
        """Load data from file into pandas DataFrame."""        
        file_extension = self.csv_dir.split(".")[-1]
        if file_extension in ["csv", "tsv", "txt"]:
            sep = "," if file_extension == "csv" else "\t"
            data = pd.read_csv(self.csv_dir, sep=sep, index_col=self.index_col)
        else:
            raise Exception('Unsupported file type. Supported types: ["csv", "tsv", "txt"]')

        self.X = data.drop(self.label, axis=1)
        self.y = data[self.label].copy()
        
    def __encode_labels(
            self
        ) -> None:
        """Encode target variable from string to numeric values."""        
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.y)
        self.label_mapping = {
            index: class_label
            for index, class_label in enumerate(label_encoder.classes_)
        }
        logging.info(f"Label mapping: {self.label_mapping}")
