import os
import logging
import argparse
from typing import Optional
from src.db.manager import DatabaseManager
from src.constants.query import CREATE_TABLE_SQL

def init_database(db_name: Optional[str] = None) -> str:
    """
    Initialize a new SQLite database with predefined schema.

    Parameters
    ----------
    db_name : str, optional
        Name for the database. If None, defaults to 'ai4meta.db'

    """
    logger = logging.getLogger(__name__)

    print(db_name)
    
    try:
        # Use default name if none provided
        db_name = db_name or "ai4meta.db"
        
        # Initialize database manager
        database_manager = DatabaseManager(db_name=db_name)
        
        # Initialize schema
        database_manager.initialize_schema(CREATE_TABLE_SQL)
        
        logger.info(f"✓ Database initialized successfully at {database_manager.db_path}")
        return database_manager.db_path
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    finally:
        if 'database_manager' in locals():
            database_manager.close_connection()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Initialize a database with predefined schema.')
    parser.add_argument('--db_name', type=str, help='Name for the database (default: ai4meta.db)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize database with provided name or default
    db_path = init_database(args.db_name)
