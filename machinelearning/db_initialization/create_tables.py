import sqlite3
import os
from query import CREATE_TABLE_SQL

def initialize_database(db_file, create_table_queries):
    # Ensure the directory for the database exists
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    for query in create_table_queries:
        cursor.execute(query)
    connection.commit()
    connection.close()

# Specify the SQLite database file inside the "database" folder
database_folder = "Database"
database_file = os.path.join(database_folder, "ai4meta.db")

# Initialize the database
initialize_database(database_file, CREATE_TABLE_SQL)
print(f"Database initialized and tables created in {database_file}.")
