import psycopg2
from psycopg2 import sql
import json
import os

# Construct the path to the credentials JSON file using os
credentials_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "db_credentials",
    "credentials.json"
)

# Load the credentials from the JSON file
with open(credentials_path, "r") as file:
    db_credentials = json.load(file)

# Establish a connection to the PostgreSQL database
connection = psycopg2.connect(
    dbname=db_credentials["db_name"],
    user=db_credentials["db_user"],
    password=db_credentials["db_password"],
    host=db_credentials["db_host"],
    port=db_credentials["db_port"]
)

# Create a cursor object
cursor = connection.cursor()

# SQL statements to create each table
create_tables_sql = [
    """
    CREATE TABLE IF NOT EXISTS Datasets (
        dataset_id SERIAL PRIMARY KEY,
        dataset_name VARCHAR(255) NOT NULL UNIQUE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Classifiers (
        classifier_id SERIAL PRIMARY KEY,
        estimator VARCHAR(255) NOT NULL,
        inner_selection VARCHAR(255)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Feature_Selection (
        selection_id SERIAL PRIMARY KEY,
        way_of_selection VARCHAR(255),
        numbers_of_features INT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Hyperparameters (
        hyperparameter_id SERIAL PRIMARY KEY,
        hyperparameters JSON NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Performance_Metrics (
        performance_id SERIAL PRIMARY KEY,
        matthews_corrcoef JSON,
        roc_auc JSON,
        accuracy JSON,
        balanced_accuracy JSON,
        recall JSON,
        precision JSON,
        f1 JSON,
        specificity JSON,
        average_precision JSON
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Samples_Classification_Rates (
        sample_rate_id SERIAL PRIMARY KEY,
        samples_classification_rates JSON
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Job_Parameters (
        job_id SERIAL PRIMARY KEY,
        n_trials INT,
        rounds INT,
        feature_selection_type VARCHAR(255),
        feature_selection_method VARCHAR(255),
        inner_scoring VARCHAR(255),
        outer_scoring VARCHAR(255),
        inner_splits INT,
        outer_splits INT,
        normalization VARCHAR(255),
        missing_values_method VARCHAR(255),
        class_balance VARCHAR(255),
        evaluation_mthd VARCHAR(255),
        param_grid JSON,
        features_names_list JSON
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Shap_Values (
        shap_values_id SERIAL PRIMARY KEY,
        shap_values JSON
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Job_Combinations (
        combination_id SERIAL PRIMARY KEY,
        job_id INT REFERENCES Job_Parameters(job_id) ON DELETE CASCADE,
        classifier_id INT REFERENCES Classifiers(classifier_id) ON DELETE CASCADE,
        dataset_id INT REFERENCES Datasets(dataset_id) ON DELETE CASCADE,
        selection_id INT REFERENCES Feature_Selection(selection_id) ON DELETE CASCADE,
        hyperparameter_id INT REFERENCES Hyperparameters(hyperparameter_id) ON DELETE CASCADE,
        performance_id INT REFERENCES Performance_Metrics(performance_id) ON DELETE CASCADE,
        sample_rate_id INT REFERENCES Samples_Classification_Rates(sample_rate_id) ON DELETE CASCADE,
        shap_values_id INT REFERENCES Shap_Values(shap_values_id) ON DELETE CASCADE,
        model_selection_type VARCHAR(255) 
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Feature_Counts (
        count_id SERIAL PRIMARY KEY,
        feature_name VARCHAR(255) NOT NULL,
        count INT NOT NULL,
        combination_id INT REFERENCES Job_Combinations(combination_id) ON DELETE CASCADE
    );
    """
]

# Execute each CREATE TABLE statement
for statement in create_tables_sql:
    cursor.execute(statement)

# Commit the changes and close the connection
connection.commit()
cursor.close()
connection.close()

print("Database schema created successfully.")