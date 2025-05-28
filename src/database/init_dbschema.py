import os
import logging
import argparse
from typing import Optional

from src.database.manager import DatabaseManager

def init_database(
        db_name: Optional[str] = None, 
        db_folder: str = "database"
    ) -> str:
    """
    Initialize a new SQLite database with SQLAlchemy schema.

    Parameters
    ----------
    db_name : str, optional
        Name for the database. If None, defaults to 'ai4meta.db'
    db_folder : str, optional
        Folder to store the database. Defaults to 'database'

    Returns
    -------
    str
        Path to the initialized database
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Use default name if none provided
        db_name = db_name or "ai4meta.db"
        
        # Initialize database manager
        logger.info(f"Initializing database: {db_name}")
        database_manager = DatabaseManager(db_name=db_name, db_folder=db_folder)
        
        # Initialize schema using SQLAlchemy models
        database_manager.initialize_schema()
        
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
    parser = argparse.ArgumentParser(description='Initialize a database with SQLAlchemy schema.')
    parser.add_argument('--db_name', type=str, help='Name for the database (default: ai4meta.db)')
    parser.add_argument('--db_folder', type=str, default='database', help='Folder to store the database (default: database)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize database with provided name or default
    db_path = init_database(args.db_name, args.db_folder)
    print(f"Database initialized at: {db_path}")