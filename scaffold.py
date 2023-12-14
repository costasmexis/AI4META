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

    def encode_categorical(self):
        ''' Function to encode the target varable. From str to 1/0.'''
        self.y = self.y.astype('category').cat.codes

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
        
    def feature_selection(self, method='mrmr', n_features=10):
        ''' Function to perform feature selection.'''
        if method == 'mrmr':
            self.selected_features = mrmr_classif(self.X, self.y.values, K=n_features)
            self.X = self.X[self.selected_features]
        else:
            raise Exception("Unsupported feature selection method.")

    def create_test_data(self):
    	''' Function to create a test data file '''
    	pass

    def __str__(self):
        ''' Function to print the dataset information.'''
        return f'Number of rows: {self.data.shape[0]} \nNumber of columns: {self.data.shape[1]}'
    
    def __getitem__(self, idx):
        ''' Function to get a sample from the dataset.'''
        return self.data.iloc[idx]

class MachineLearningEstimator(DataLoader):
    def __init__(self, estimator, param_grid, label, csv_dir):
        ''' Class to hold the machine learning estimator and related data 
            Inherits from DataLoader class
            - estimator (sklearn estimator): estimator to be used
            - param_grid (dict): hyperparameters to be tuned
            - label (str): name of the target column
            - csv_dir (str): path to the csv file
        '''
        super().__init__(label, csv_dir)
        self.estimator = estimator
        self.name = estimator.__class__.__name__
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None
        self.best_estimator = None
        
        self.available_clfs = {
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'LogisticRegression': LogisticRegression(),
            'XGBClassifier': XGBClassifier(),
            'NaiveBayes': GaussianNB(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'SVC': SVC()
        }

        self.bayesian_grid = {
            'RandomForestClassifier': lambda trial: RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 2, 200),
                criterion='gini',  # or trial.suggest_categorical('criterion', ['gini', 'entropy'])
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                n_jobs=-1,
            ),
            'KNeighborsClassifier': lambda trial: KNeighborsClassifier(
                n_neighbors=trial.suggest_int('n_neighbors', 2, 15),
                weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
                algorithm=trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                p=trial.suggest_int('p', 1, 2),
                leaf_size=trial.suggest_int('leaf_size', 5, 50),
                n_jobs=-1
            ),
            'DecisionTreeClassifier': lambda trial: DecisionTreeClassifier(
                trial.suggest_categorical('criterion', ['gini', 'entropy']),
                splitter=trial.suggest_categorical('splitter', ['best', 'random']),
                max_depth=trial.suggest_int('max_depth', 1, 100),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_weight_fraction_leaf=trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
            ),
            'SVC': lambda trial: SVC(
                C=trial.suggest_int('C', 1, 10),
                kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']),
                probability=trial.suggest_categorical('probability', [True, False]),
                shrinking=trial.suggest_categorical('shrinking', [True, False]),
                decision_function_shape=trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr'])
            ),
            'GradientBoostingClassifier': lambda trial: GradientBoostingClassifier(
                loss=trial.suggest_categorical('loss', ['log_loss', 'exponential']),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5),
                n_estimators=trial.suggest_int('n_estimators', 2, 200),
                criterion=trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10)
            ),
            'XGBClassifier': lambda trial: XGBClassifier(
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5),
                n_estimators=trial.suggest_int('n_estimators', 2, 200),
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                gamma=trial.suggest_float('gamma', 0.0, 0.5),
                subsample=trial.suggest_float('subsample', 0.1, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.1, 1.0),
                nthread=-1,
                verbosity=0
            ),
            'LinearDiscriminantAnalysis': lambda trial: LinearDiscriminantAnalysis(
                solver=trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
                shrinkage=trial.suggest_float('shrinkage', 0.0, 1.0),
                n_components=trial.suggest_int('n_components', 1, 10)
            ),
            'LogisticRegression': lambda trial: LogisticRegression(
                penalty=trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none']),
                C=trial.suggest_float('C', 0.1, 10.0),
                solver=trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                max_iter=trial.suggest_int('max_iter', 100, 1000),
                n_jobs=-1
            ),
            'NaiveBayes': lambda trial: GaussianNB(
                var_smoothing=trial.suggest_float('var_smoothing', 1e-9, 1e-5)
            )
        }
        
        # Check if the estimator is valid 
        if self.name not in self.available_clfs.keys():
            raise ValueError(f'Invalid estimator: {self.name}. Select one of the following: {list(self.available_clfs.keys())}')
        
	def grid_search(self, X=None, y=None, scoring='accuracy', cv=5, verbose=True):
	    ''' Function to perform a grid search
	        - X (array): features
	        - y (array): target
	        - scoring (str): scoring metric
	        - cv (int): number of folds for cross-validation
	        - verbose (bool): whether to print the results
	    '''
	    if scoring not in sklearn.metrics.SCORERS.keys():
	        raise ValueError(
	            f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

	    if X is None and y is None:
	        X = self.X 
	        y = self.y

	    grid_search = GridSearchCV(self.estimator, self.param_grid, scoring=scoring, cv=cv)
	    grid_search.fit(X, y)
	    self.best_params = grid_search.best_params_
	    self.best_score = grid_search.best_score_
	    self.best_estimator = grid_search.best_estimator_
	    if verbose:
	        print(f'Best parameters: {self.best_params}')
	        print(f'Best {scoring}: {self.best_score}')

	def random_search(self, X=None, y=None, scoring='accuracy', cv=5, n_iter=100, verbose=True):
        ''' Function to perform a random search
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
            - n_iter (int): number of iterations
            - verbose (bool): whether to print the results
        '''
        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

        if X is None and y is None:
            X = self.X 
            y = self.y

        random_search = RandomizedSearchCV(self.estimator, self.param_grid, scoring=scoring, cv=cv, n_iter=n_iter)
        random_search.fit(X, y)
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_estimator = random_search.best_estimator_
        if verbose:
            print(f'Best parameters: {self.best_params}')
            print(f'Best {scoring}: {self.best_score}')

	def bayesian_search():
		''' TODO: To be implemented... '''
		pass






class MLPipelines(MachineLearningEstimator):

    def __init__(self, estimator, param_grid, label, csv_dir):
        ''' Class to perform machine learning pipelines 
            Inherits from MachineLearningEstimator class
            - estimator (sklearn estimator): estimator to be used
            - param_grid (dict): hyperparameters to be tuned
            - label (str): name of the target column
            - csv_dir (str): path to the csv file
        '''
        super().__init__(estimator, param_grid, label, csv_dir)

	def cross_validation(self, scoring='accuracy', cv=5) -> list:
        ''' Function to perform a simple cross validation
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
        returns:
            - scores (list): list of scores for each fold
        '''
        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

        scores = cross_val_score(self.estimator, self.X, self.y, cv=cv, scoring=scoring)
        print(f'Average {scoring}: {np.mean(scores)}')
        print(f'Standard deviation {scoring}: {np.std(scores)}')
        return scores

    def bootsrap(self, n_iter=100, test_size=0.2, optimizer='grid_search', random_iter=25, n_trials=100, cv=5, scoring='accuracy'):
        ''' Performs boostrap validation on a given estimator.
            - n_iter: number of iterations to perform boostrap validation
            - test_size: test size for each iteration
            - optimizer: 'grid_search' for GridSearchCV
                         'reandom_search' for RandomizedSearchCV
                         'bayesian_search' for optuna
            - random_iter: number of iterations for RandomizedSearchCV
            - n_trials: number of trials for optuna
            - cv: number of folds for cross-validation
            - scoring: scoring metric
    
        returns:
            - eval_metrics (list): list of evaluation metrics for each iteration
        '''                
        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

        eval_metrics = []

        for i in range(n_iter):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=i)
            if self.param_grid is None or self.param_grid == {}:
                self.estimator.fit(X_train, y_train)
            else:
                if optimizer == 'grid_search':
                    self.grid_search(X_train, y_train, scoring=scoring, cv=cv, verbose=False)
                elif optimizer == 'random_search':
                    self.random_search(X_train, y_train, scoring=scoring, cv=cv, n_iter=random_iter, verbose=False)
                elif optimizer == 'bayesian_search':
                    self.bayesian_search(X_train, y_train, scoring=scoring, direction='maximize', cv=cv, n_trials=n_trials, verbose=False)
                    self.best_estimator.fit(X_train, y_train)
                else:
                    raise ValueError(f'Invalid optimizer: {optimizer}. Select one of the following: grid_search, bayesian_search')
            
            y_pred = self.best_estimator.predict(X_test)
            eval_metrics.append(get_scorer(scoring)._score_func(y_test, y_pred))

        print(f'Average {scoring}: {np.mean(eval_metrics)}')
        print(f'Standard deviation {scoring}: {np.std(eval_metrics)}')
        return eval_metrics

	def nested_cross_validation(self):
		pass