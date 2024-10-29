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
        dataset_id INT REFERENCES Datasets(dataset_id) ON DELETE CASCADE,
        estimator VARCHAR(255) NOT NULL,
        inner_selection VARCHAR(255)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Feature_Selection (
        selection_id SERIAL PRIMARY KEY,
        classifier_id INT REFERENCES Classifiers(classifier_id) ON DELETE CASCADE,
        dataset_id INT REFERENCES Datasets(dataset_id) ON DELETE CASCADE,
        way_of_selection VARCHAR(255),
        numbers_of_features INT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Hyperparameters (
        hyperparameter_id SERIAL PRIMARY KEY,
        classifier_id INT REFERENCES Classifiers(classifier_id) ON DELETE CASCADE,
        dataset_id INT REFERENCES Datasets(dataset_id) ON DELETE CASCADE,
        selection_id INT REFERENCES Feature_Selection(selection_id) ON DELETE CASCADE,
        hyperparameters JSON NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Performance_Metrics (
        performance_id SERIAL PRIMARY KEY,
        classifier_id INT REFERENCES Classifiers(classifier_id) ON DELETE CASCADE,
        dataset_id INT REFERENCES Datasets(dataset_id) ON DELETE CASCADE,
        selection_id INT REFERENCES Feature_Selection(selection_id) ON DELETE CASCADE,
        matthews_corrcoef JSON,
        roc_auc JSON,
        accuracy JSON,
        balanced_accuracy JSON,
        recall JSON,
        precision JSON,
        f1_score JSON,
        specificity JSON,
        average_precision JSON
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Samples_Classification_Rates (
        sample_rate_id SERIAL PRIMARY KEY,
        classifier_id INT REFERENCES Classifiers(classifier_id) ON DELETE CASCADE,
        dataset_id INT REFERENCES Datasets(dataset_id) ON DELETE CASCADE,
        selection_id INT REFERENCES Feature_Selection(selection_id) ON DELETE CASCADE,
        samples_classification_rates JSON
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Feature_Counts (
        count_id SERIAL PRIMARY KEY,
        feature_name VARCHAR(255) NOT NULL,
        count INT NOT NULL,
        selection_id INT REFERENCES Feature_Selection(selection_id) ON DELETE CASCADE,
        dataset_id INT REFERENCES Datasets(dataset_id) ON DELETE CASCADE
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