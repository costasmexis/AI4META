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
    def __init__(self, label: str, csv_dir: str):
        """
        Constructor for the DataLoader class.

        :param label: Target variable column name
        :type label: str
        :param csv_dir: Path to the csv file
        :type csv_dir: str
        """
        self.csv_dir = csv_dir
        self.label = label
        self.data = None
        self.X = None
        self.y = None
        self.label_mapping = None
        self.selected_features = None
        self.supported_extensions = ['csv', 'tsv', 'txt']
        self.scaler = None 
        self._load_data() 
        self._encode_labels()
        
    def _load_data(self, index_col=0):
        """
        Function to load the dataset from the csv file.

        :param index_col: Index column, defaults to 0
        :type index_col: int, optional
        """

        file_extension = self.csv_dir.split('.')[-1]
        if file_extension in self.supported_extensions:
            sep = ',' if file_extension == 'csv' else '\t'
            self.data = pd.read_csv(self.csv_dir, sep=sep, index_col=index_col)
        else:
            raise Exception("Unsupported file type.")
                
        self.X = self.data.drop(self.label, axis=1)
        self.y = self.data[self.label].copy()
    
    def _encode_labels(self):
        """
        Function to encode the target labels.
        """
        
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.y)
        self.label_mapping = {index: class_label for index, class_label in enumerate(label_encoder.classes_)}
        print("Label mapping:", self.label_mapping)
        
    def missing_values(self, method='drop'):
        """
        Function to handle missing values in the dataset.

        :param method: Method to be used, defaults to 'drop'
        :type method: str, optional
        """
        total_missing = self.X.isnull().sum().sum()
        print(f'Number of missing values: {total_missing}')

        if method == 'drop':
            self.X.dropna(inplace=True)
        elif method in ['mean', 'median', '0']:
            fill_value = 0 if method == '0' else getattr(self.X, method)()
            self.X.fillna(fill_value, inplace=True)
        
        
    # TODO: FIX NORMALIZE FUNCTION
    def normalize(self, X=None, method='minmax', train_test_set=False, X_test=None):
        initial_data=False
        
        if X is None:
            X = self.X
            initial_data=True
            print(f'Converting the raw data with {method} normalization method....')
            
        if method in ['minmax', 'standard']:
            self.scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            if train_test_set:
                X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
        else:
            raise Exception("Unsupported normalization method.")
        if train_test_set:
            pass
        else:
            print('Normalization completed.')
            
        if initial_data:
            self.X = X
        elif train_test_set:
            return X, X_test
        else: 
            return X
    
    def feature_selection(self, X=None, y=None, method='mrmr', num_features=10, inner_method='chi2'):        
            """
            Perform feature selection on the input dataset.

            :param X: The input features, defaults to None
            :type X: array-like, optional
            :param y: The target variable, defaults to None
            :type y: array-like, optional
            :param method: The feature selection method to use, defaults to 'mrmr'
            :type method: str, optional
            :param num_features: The number of features to select, defaults to 10
            :type num_features: int, optional
            :param inner_method: The scoring method used for feature selection, defaults to 'chi2'
            :type inner_method: str, optional
            :return: The selected features or None if X and y are provided
            :rtype: array-like or None
            :raises Exception: If an unsupported feature selection method or inner method is selected
            """

            if method not in ['mrmr', 'kbest', 'percentile']:
                raise Exception("Unsupported feature selection method. Select one of 'mrmr', 'kbest', 'percentile'")
            if inner_method not in ['chi2', 'f_classif', 'mutual_info_classif']:
                raise Exception("Unsupported inner method. Select one of 'chi2', 'f_classif', 'mutual_info_classif'")
            if method == 'percentile' and num_features is not None and (num_features >= 100 or num_features <= 0):
                raise Exception("num_features for percentile option must be between 0 and 100.")
            
            datasetXy=False
            if X is None and y is None:
                X = self.X
                y = self.y
                datasetXy=True


            method_mapping = {
                'chi2': chi2,
                'f_classif': f_classif,
                'mutual_info_classif': mutual_info_classif
            }
            scoring_function = method_mapping[inner_method]    
            
            if method == 'mrmr':
                self.selected_features = mrmr_classif(X, y, K=num_features, show_progress=False)
                X = X[self.selected_features]

            else:
                if not isinstance(self.scaler, MinMaxScaler):
                    raise Exception("Feature selection method requires MinMaxScaler.")
                if method == 'kbest':
                    X = SelectKBest(scoring_function, k=num_features).fit_transform(X, y)

                elif method == 'percentile':
                    X = SelectPercentile(scoring_function, percentile=num_features).fit_transform(X, y)
                else: 
                    raise Exception("Unsupported feature selection method.")
            
            if datasetXy:
                self.X = X
                self.y = y
            else: 
                return self.selected_features
        

    def create_test_data(self, output_dir='./test_data.csv'):
        """
        Create a test dataset.

        :param output_dir: The directory where the test data will be saved. Defaults to './test_data.csv'.
        :type output_dir: str, optional
        """
        test_data = pd.DataFrame(columns=self.X.columns.values)
        test_data.to_csv(output_dir)
        
    def __str__(self):
        ''' Function to print the dataset information.'''
        return f'Number of rows: {self.data.shape[0]} \nNumber of columns: {self.data.shape[1]}'
    
    def __getitem__(self, idx):
        ''' Function to get a sample from the dataset.'''
        return self.data.iloc[idx]