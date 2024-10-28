import psycopg2
from psycopg2 import sql

# Establish a connection to the PostgreSQL database
connection = psycopg2.connect(
    dbname="ai4meta",
    user="postgres",
    password="spyros212121",
    host="localhost",
    port="5432"
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
        classifier VARCHAR(255) NOT NULL,
        inner_selection VARCHAR(255)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Hyperparameters (
        hyperparameter_id SERIAL PRIMARY KEY,
        classifier_id INT REFERENCES Classifiers(classifier_id) ON DELETE CASCADE,
        dataset_id INT REFERENCES Datasets(dataset_id) ON DELETE CASCADE,
        hyperparameters JSON NOT NULL
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
    CREATE TABLE IF NOT EXISTS Performance_Metrics (
        performance_id SERIAL PRIMARY KEY,
        classifier_id INT REFERENCES Classifiers(classifier_id) ON DELETE CASCADE,
        dataset_id INT REFERENCES Datasets(dataset_id) ON DELETE CASCADE,
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
        samples_classification_rates JSON
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