"""
Database Manager

Provides database connection management, migrations, and operations
for the AI Automation Bot.
"""

import os
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
from psycopg2.extras import RealDictCursor

from ...core.exceptions import DatabaseError
from ...core.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class DatabaseManager:
    """Database manager for handling connections and operations."""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize database manager.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string or self._get_default_connection_string()
        self.engine = None
        self.SessionLocal = None
        self.metadata = MetaData()
        
        # Initialize database
        self._initialize_database()
    
    def _get_default_connection_string(self) -> str:
        """Get default database connection string from environment."""
        host = os.environ.get('DB_HOST', 'localhost')
        port = os.environ.get('DB_PORT', '5432')
        database = os.environ.get('DB_NAME', 'ai_automation_bot')
        username = os.environ.get('DB_USER', 'postgres')
        password = os.environ.get('DB_PASSWORD', 'password')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def _initialize_database(self):
        """Initialize database connection and session."""
        try:
            # Create engine
            self.engine = create_engine(
                self.connection_string,
                echo=False,  # Set to True for SQL logging
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Test connection
            self._test_connection()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    def _test_connection(self):
        """Test database connection."""
        try:
            with self.engine.connect() as connection:
                connection.execute("SELECT 1")
            logger.info("Database connection test successful")
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise DatabaseError(f"Database connection failed: {str(e)}")
    
    def get_session(self) -> Session:
        """
        Get database session.
        
        Returns:
            Database session
            
        Raises:
            DatabaseError: If session creation fails
        """
        try:
            return self.SessionLocal()
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            raise DatabaseError(f"Failed to create database session: {str(e)}")
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Table creation error: {e}")
            raise DatabaseError(f"Failed to create database tables: {str(e)}")
    
    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Table drop error: {e}")
            raise DatabaseError(f"Failed to drop database tables: {str(e)}")
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results as list of dictionaries
            
        Raises:
            DatabaseError: If query execution fails
        """
        try:
            with psycopg2.connect(self.connection_string) as connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params or {})
                    
                    if query.strip().upper().startswith('SELECT'):
                        return [dict(row) for row in cursor.fetchall()]
                    else:
                        connection.commit()
                        return [{'affected_rows': cursor.rowcount}]
                        
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise DatabaseError(f"Failed to execute query: {str(e)}")
    
    def execute_transaction(self, queries: List[Dict[str, Any]]):
        """
        Execute multiple queries in a transaction.
        
        Args:
            queries: List of query dictionaries with 'query' and 'params' keys
            
        Raises:
            DatabaseError: If transaction fails
        """
        try:
            with psycopg2.connect(self.connection_string) as connection:
                with connection.cursor() as cursor:
                    for query_data in queries:
                        query = query_data['query']
                        params = query_data.get('params', {})
                        cursor.execute(query, params)
                    
                    connection.commit()
                    logger.info("Transaction executed successfully")
                    
        except Exception as e:
            logger.error(f"Transaction error: {e}")
            raise DatabaseError(f"Failed to execute transaction: {str(e)}")
    
    def backup_database(self, backup_path: str):
        """
        Create database backup.
        
        Args:
            backup_path: Path to save backup file
            
        Raises:
            DatabaseError: If backup fails
        """
        try:
            import subprocess
            
            # Extract connection details
            from urllib.parse import urlparse
            parsed = urlparse(self.connection_string)
            
            # Build pg_dump command
            cmd = [
                'pg_dump',
                '-h', parsed.hostname,
                '-p', str(parsed.port),
                '-U', parsed.username,
                '-d', parsed.path[1:],  # Remove leading slash
                '-f', backup_path
            ]
            
            # Set password environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password
            
            # Execute backup
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")
            
            logger.info(f"Database backup created: {backup_path}")
            
        except Exception as e:
            logger.error(f"Database backup error: {e}")
            raise DatabaseError(f"Failed to create database backup: {str(e)}")
    
    def restore_database(self, backup_path: str):
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Raises:
            DatabaseError: If restore fails
        """
        try:
            import subprocess
            
            # Extract connection details
            from urllib.parse import urlparse
            parsed = urlparse(self.connection_string)
            
            # Build psql command
            cmd = [
                'psql',
                '-h', parsed.hostname,
                '-p', str(parsed.port),
                '-U', parsed.username,
                '-d', parsed.path[1:],  # Remove leading slash
                '-f', backup_path
            ]
            
            # Set password environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password
            
            # Execute restore
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"psql restore failed: {result.stderr}")
            
            logger.info(f"Database restored from: {backup_path}")
            
        except Exception as e:
            logger.error(f"Database restore error: {e}")
            raise DatabaseError(f"Failed to restore database: {str(e)}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.
        
        Returns:
            Dictionary with database information
        """
        try:
            info = {
                'connection_string': self.connection_string.replace(
                    self.connection_string.split('@')[0].split(':')[-1], '***'
                ),
                'engine_info': str(self.engine),
                'pool_size': self.engine.pool.size(),
                'checked_in': self.engine.pool.checkedin(),
                'checked_out': self.engine.pool.checkedout(),
                'overflow': self.engine.pool.overflow()
            }
            
            # Get database statistics
            with self.get_session() as session:
                # Table count
                result = session.execute("""
                    SELECT COUNT(*) as table_count 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                info['table_count'] = result.scalar()
                
                # Database size
                result = session.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size
                """)
                info['database_size'] = result.scalar()
                
                # Active connections
                result = session.execute("""
                    SELECT COUNT(*) as active_connections 
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                """)
                info['active_connections'] = result.scalar()
            
            return info
            
        except Exception as e:
            logger.error(f"Database info error: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Close database connections."""
        try:
            if self.engine:
                self.engine.dispose()
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Database close error: {e}")


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get global database manager instance.
    
    Returns:
        Database manager instance
    """
    global db_manager
    
    if db_manager is None:
        db_manager = DatabaseManager()
    
    return db_manager


def init_database(connection_string: str = None):
    """
    Initialize global database manager.
    
    Args:
        connection_string: Database connection string
    """
    global db_manager
    
    db_manager = DatabaseManager(connection_string)
    db_manager.create_tables()
    
    logger.info("Global database manager initialized") 