import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from mrmr import mrmr_classif
from numba import jit, prange


class DataLoader:
    def __init__(self, label, csv_dir):
        """Class to load a metabolomics dataset from a file.
        Args:
            label (str): name of the target column.
            csv_dir (str): path to the csv file.
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
        self.__load_data() 
        self.__encode_labels()
    
    def __load_data(self, index_col=0):
        """Function to load data from a file.
        Args:
            index_col (int, optional): index column of dataset. Defaults to 0.
        """
        file_extension = self.csv_dir.split('.')[-1]
        if file_extension in self.supported_extensions:
            sep = ',' if file_extension == 'csv' else '\t'
            self.data = pd.read_csv(self.csv_dir, sep=sep, index_col=index_col)
        else:
            raise Exception("Unsupported file type.")
                
        self.X = self.data.drop(self.label, axis=1)
        self.y = self.data[self.label].copy()
    
    def __encode_labels(self):
        """ Function to encode the target varable. From str to 1/0. """
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.y)
        # self.label_mapping = {class_label: index for index, class_label in enumerate(label_encoder.classes_)}
        # print("Label mapping:", self.label_mapping)
        self.label_mapping = {index: class_label for index, class_label in enumerate(label_encoder.classes_)}
        print("Label mapping:", self.label_mapping)
        
    def missing_values(self, method='drop'):
        ''' Function to handle missing values in the dataset.'''
        total_missing = self.data.isnull().sum().sum()
        print(f'Number of missing values: {total_missing}')
        if method == 'drop':
            self.data.dropna(inplace=True)
        elif method in ['mean', 'median']:
            fill_value = getattr(self.data, method)()
            self.data.fillna(fill_value, inplace=True)
        else:
            raise Exception("Unsupported missing values method.")
        
    def normalize(self, method='minmax'):
        """ Function to normalize the dataset.
        Args:
            method (str, optional): Method to use for normalization. Defaults to 'minmax'.
                - minmax: MinMaxScaler
                - standard: StandardScaler
        """
        if method in ['minmax', 'standard']:
            self.scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
            self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)
        else:
            raise Exception("Unsupported normalization method.")
    
    # @jit(parallel=True)
    def feature_selection(self, X=None, y=None, method='mrmr', num_features=10, inner_method='chi2'):        
        """ Function to perform Feature Selection.
        Args:
            method (str, optional): Method to use for feature selection. Defaults to 'mrmr'.
            n_features (int, optional): Number of features to be selected. Defaults to 10.
            inner_method (str, optional) ['chi2', 'f_classif', 'mutual_info_classif']: Inner method for SelectKBest. Defaults to 'chi2'.
        """
        if method not in ['mrmr', 'kbest', 'percentile']:
            raise Exception("Unsupported feature selection method. Select one of 'mrmr', 'kbest', 'percentile'")
        if inner_method not in ['chi2', 'f_classif', 'mutual_info_classif']:
            raise Exception("Unsupported inner method. Select one of 'chi2', 'f_classif', 'mutual_info_classif'")
        if method is 'percentile' and num_features is not None and (num_features >= 100 or num_features <= 0):
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
            # self.selected_features = mrmr_classif(self.X, self.y, K=n_features)
            # self.X = self.X[self.selected_features]
            self.selected_features = mrmr_classif(X, y, K=num_features,show_progress=False)
            X = X[self.selected_features]

        else:
            if not isinstance(self.scaler, MinMaxScaler):
                raise Exception("Feature selection method requires MinMaxScaler.")
            if method == 'kbest':
                X = SelectKBest(scoring_function, k=num_features).fit_transform(X, y)
                # self.X = SelectKBest(scoring_function, k=n_features).fit_transform(self.X, self.y)

            elif method == 'percentile':
                # self.X = SelectPercentile(scoring_function, percentile=percentile).fit_transform(self.X, self.y)
                X = SelectPercentile(scoring_function, percentile=num_features).fit_transform(X, y)
            else: raise Exception("Unsupported feature selection method.")
        
        if datasetXy:
            self.X = X
            self.y = y
        else: return self.selected_features
        

    def create_test_data(self, output_dir='./test_data.csv'):
        """ Function to generate a test data template csv file.
        Args:
            output_dir (str, optional): The directory to save the csv file. Defaults to './test_data.csv'.
        """
        test_data = pd.DataFrame(columns=self.X.columns.values)
        test_data.to_csv(output_dir)        
        
    def __str__(self):
        ''' Function to print the dataset information.'''
        return f'Number of rows: {self.data.shape[0]} \nNumber of columns: {self.data.shape[1]}'
    
    def __getitem__(self, idx):
        ''' Function to get a sample from the dataset.'''
        return self.data.iloc[idx]

    