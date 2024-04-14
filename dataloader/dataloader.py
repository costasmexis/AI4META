import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from mrmr import mrmr_classif

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
        
    def missing_values(self, data=None, method='drop'):
        ''' Function to handle missing values in the dataset.'''
        initial_data=False
        if data is None:
            data = self.X
            initial_data=True
            total_missing = data.isnull().sum().sum()
            print(f'Number of missing values: {total_missing}')
        if method == 'drop':
            data.dropna(inplace=True)
        elif method in ['mean', 'median', '0']:
            if method == '0':
                fill_value = 0  
            else:
                fill_value = getattr(data, method)()  
            data.fillna(fill_value, inplace=True)
        else:
            raise Exception("Unsupported missing values method.")
        
        if initial_data:
            self.X = data
        else:return data
        
        
    def normalize(self, X=None, method='minmax', train_test_set=False, X_test=None):
        """
        Normalizes the dataset using specified method.

        This method supports normalization using either MinMaxScaler or StandardScaler from the pandas library. 
        If no dataset is explicitly passed to it, the method defaults to normalizing the instance's dataset.

        Parameters:
        - X (pandas.DataFrame, optional): The dataset to normalize. If not provided, the instance's dataset is used.
        - method (str, optional): The normalization method to use. Defaults to 'minmax'. Options include:
            - 'minmax': Uses MinMaxScaler for normalization.
            - 'standard': Uses StandardScaler for normalization.

        Returns:
        - pandas.DataFrame: The normalized dataset. If `X` is not provided, the instance's dataset is updated in place and nothing is returned.

        Raises:
        - Exception: If an unsupported normalization method is specified.

        Note:
        - This method updates the instance's dataset if no `X` is provided. It also updates the instance's scaler attribute to the scaler used for normalization.
        - A message indicating the completion of normalization is printed. If the instance's dataset is used, a message specifying the normalization method is also printed.
        """
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
        else: return X
    
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
            self.selected_features = mrmr_classif(X, y, K=num_features,show_progress=False)
            X = X[self.selected_features]

        else:
            if not isinstance(self.scaler, MinMaxScaler):
                raise Exception("Feature selection method requires MinMaxScaler.")
            if method == 'kbest':
                X = SelectKBest(scoring_function, k=num_features).fit_transform(X, y)

            elif method == 'percentile':
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

    