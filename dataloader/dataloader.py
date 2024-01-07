import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mrmr import mrmr_classif
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    def __init__(self, label, csv_dir):
        ''' Class to load a metabolomics dataset from a file. 
            - label: the name of the target column.
            - csv_dir: the path to the csv file.
        '''
        self.csv_dir = csv_dir
        self.label = label
        self.data = None
        self.X = None
        self.y = None
        self.selected_features = None
        self.supported_extensions = ['csv', 'tsv', 'txt']
        self.__load_data() 
    
    def __load_data(self, index_col=0):
        ''' Function to load data from a file. 
            - index_col (0): the index column of the dataset
        '''
        file_extension = self.csv_dir.split('.')[-1]
        if file_extension in self.supported_extensions:
            sep = ',' if file_extension == 'csv' else '\t'
            self.data = pd.read_csv(self.csv_dir, sep=sep, index_col=index_col)
        else:
            raise Exception("Unsupported file type.")
                
        self.X = self.data.drop(self.label, axis=1)
        self.y = self.data[self.label].copy()
    
    # def encode_categorical(self):
        
    #     self.y = self.y.astype('category').cat.codes
        
    def encode_labels(self):
        ''' Function to encode the target varable. From str to 1/0.'''
        self.label_encoder = LabelEncoder()
        
        self.y = self.label_encoder.fit_transform(self.y)
        
        label_mapping = {class_label: index for index, class_label in enumerate(self.label_encoder.classes_)}
        print("Label mapping:", label_mapping)
    
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
        ''' Function to normalize the dataset.'''
        if method in ['minmax', 'standard']:
            scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
            self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
        else:
            raise Exception("Unsupported normalization method.")
        
    def feature_selection(self, method='mrmr', n_features=10, inner_method='chi2', percentile=10):
        ''' Function to perform feature selection.'''
        if method == 'mrmr':
            self.selected_features = mrmr_classif(self.X, self.y.values, K=n_features)
            self.X = self.X[self.selected_features]
        elif method == 'selectkbest':
            if inner_method == 'chi2':
                self.X = SelectKBest(chi2, k=n_features).fit_transform(self.X, self.y)
            elif inner_method == 'f_classif':
                self.X = SelectKBest(f_classif, k=n_features).fit_transform(self.X, self.y)
            elif inner_method == 'mutual_info_classif':
                self.X = SelectKBest(mutual_info_classif, k=n_features).fit_transform(self.X, self.y)
            else: raise Exception("Unsupported inner function. \nChoose from chi2, f_classif and mutual_info_classif.")
        elif method == 'selectpercentile':
            if percentile<=100 and percentile>=0:
                if inner_method == 'chi2':
                    self.X = SelectPercentile(chi2, percentile=percentile).fit_transform(self.X, self.y)
                elif inner_method == 'f_classif':
                    self.X = SelectPercentile(f_classif, percentile=percentile).fit_transform(self.X, self.y)
                elif inner_method == 'mutual_info_classif':
                    self.X = SelectPercentile(mutual_info_classif, percentile=percentile).fit_transform(self.X, self.y)
                else: raise Exception("Unsupported inner function. \nChoose from chi2, f_classif and mutual_info_classif.")
            else: raise Exception("Percentile must be between 0 and 100.")
        else:
            raise Exception("Unsupported feature selection method.")
    
    def create_test_data(self, output_dir='./test_data.csv'):
        ''' Function to create a test data file '''
        test_data = pd.DataFrame(columns=self.X.columns.values)
        test_data.to_csv(output_dir)        
        
    def __str__(self):
        ''' Function to print the dataset information.'''
        return f'Number of rows: {self.data.shape[0]} \nNumber of columns: {self.data.shape[1]}'
    
    def __getitem__(self, idx):
        ''' Function to get a sample from the dataset.'''
        return self.data.iloc[idx]

    