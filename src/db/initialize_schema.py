from src.database.manager import DatabaseManager  
from src.constants.query import CREATE_TABLE_SQL  

def db_init():
    # Prompt user for database name
    db_name = input("Enter the desired name for the database (default: ai4meta.db): ")
    if not db_name:
        db_name = "ai4meta.db"
    
    # Initialize the DatabaseManager
    database_manager = DatabaseManager(db_name=db_name)
    
    # Initialize the schema
    database_manager.initialize_schema(CREATE_TABLE_SQL)
    
    print(f"Database initialized and tables created in {database_manager.db_path}.")




