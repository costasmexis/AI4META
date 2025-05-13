from typing import Optional, List, Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    """A class for loading data"""

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

    def __load_data(self) -> None:
        """Load data from file into pandas DataFrame."""        
        file_extension = self.csv_dir.split(".")[-1]
        if file_extension in ["csv", "tsv", "txt"]:
            sep = "," if file_extension == "csv" else "\t"
            data = pd.read_csv(self.csv_dir, sep=sep, index_col=self.index_col)
        else:
            raise Exception('Unsupported file type. Supported types: ["csv", "tsv", "txt"]')

        self.X = data.drop(self.label, axis=1)
        self.y = data[self.label].copy()
        
    def __encode_labels(self) -> None:
        """Encode target variable from string to numeric values."""        
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.y)
        self.label_mapping = {
            index: class_label
            for index, class_label in enumerate(label_encoder.classes_)
        }
        print(f"Label mapping: {self.label_mapping}")
