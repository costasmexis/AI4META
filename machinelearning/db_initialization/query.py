# SQL queries for creating tables
CREATE_TABLE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS Datasets (
        dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT NOT NULL UNIQUE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Classifiers (
        classifier_id INTEGER PRIMARY KEY AUTOINCREMENT,
        estimator TEXT NOT NULL,
        inner_selection TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Feature_Selection (
        selection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        way_of_selection TEXT,
        numbers_of_features INTEGER
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Hyperparameters (
        hyperparameter_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hyperparameters TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Performance_Metrics (
        performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
        matthews_corrcoef TEXT,
        roc_auc TEXT,
        accuracy TEXT,
        balanced_accuracy TEXT,
        recall TEXT,
        precision TEXT,
        f1 TEXT,
        specificity TEXT,
        average_precision TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Samples_Classification_Rates (
        sample_rate_id INTEGER PRIMARY KEY AUTOINCREMENT,
        samples_classification_rates TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Job_Parameters (
        job_id INTEGER PRIMARY KEY AUTOINCREMENT,
        n_trials INTEGER,
        rounds INTEGER,
        feature_selection_type TEXT,
        feature_selection_method TEXT,
        inner_scoring TEXT,
        outer_scoring TEXT,
        inner_splits INTEGER,
        outer_splits INTEGER,
        normalization TEXT,
        missing_values_method TEXT,
        class_balance TEXT,
        evaluation_mthd TEXT,
        param_grid TEXT,
        features_names_list TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Shap_Values (
        shap_values_id INTEGER PRIMARY KEY AUTOINCREMENT,
        shap_values TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Job_Combinations (
        combination_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER REFERENCES Job_Parameters(job_id) ON DELETE CASCADE,
        classifier_id INTEGER REFERENCES Classifiers(classifier_id) ON DELETE CASCADE,
        dataset_id INTEGER REFERENCES Datasets(dataset_id) ON DELETE CASCADE,
        selection_id INTEGER REFERENCES Feature_Selection(selection_id) ON DELETE CASCADE,
        hyperparameter_id INTEGER REFERENCES Hyperparameters(hyperparameter_id) ON DELETE CASCADE,
        performance_id INTEGER REFERENCES Performance_Metrics(performance_id) ON DELETE CASCADE,
        sample_rate_id INTEGER REFERENCES Samples_Classification_Rates(sample_rate_id) ON DELETE CASCADE,
        shap_values_id INTEGER REFERENCES Shap_Values(shap_values_id) ON DELETE CASCADE,
        model_selection_type TEXT,
        feature_count_ids TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Feature_Counts (
        count_id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_name TEXT NOT NULL,
        count INTEGER NOT NULL,
        combination_id INTEGER REFERENCES Job_Combinations(combination_id) ON DELETE CASCADE
    );
    """
]