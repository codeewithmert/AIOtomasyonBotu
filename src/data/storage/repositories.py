"""
Data Repositories

Repository classes for data access and manipulation operations
in the AI Automation Bot.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from sqlalchemy.exc import SQLAlchemyError

from .models import (
    CollectionModel, ExecutionModel, UserModel, MLModelModel,
    AutomationTaskModel, ReportModel, ReportTemplateModel,
    ReportScheduleModel, SystemLogModel
)
from ...core.exceptions import DatabaseError
from ...core.logger import get_logger

logger = get_logger(__name__)


class BaseRepository:
    """Base repository with common operations."""
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session.
        
        Args:
            session: Database session
        """
        self.session = session
    
    def create(self, model_instance) -> Any:
        """
        Create a new model instance.
        
        Args:
            model_instance: Model instance to create
            
        Returns:
            Created model instance
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            self.session.add(model_instance)
            self.session.commit()
            self.session.refresh(model_instance)
            return model_instance
            
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Create operation failed: {e}")
            raise DatabaseError(f"Failed to create record: {str(e)}")
    
    def get_by_id(self, model_class, record_id: str) -> Optional[Any]:
        """
        Get record by ID.
        
        Args:
            model_class: Model class
            record_id: Record ID
            
        Returns:
            Model instance or None
        """
        try:
            return self.session.query(model_class).filter(model_class.id == record_id).first()
            
        except SQLAlchemyError as e:
            logger.error(f"Get by ID operation failed: {e}")
            raise DatabaseError(f"Failed to get record by ID: {str(e)}")
    
    def get_all(self, model_class, limit: int = None, offset: int = None) -> List[Any]:
        """
        Get all records.
        
        Args:
            model_class: Model class
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of model instances
        """
        try:
            query = self.session.query(model_class)
            
            if offset:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get all operation failed: {e}")
            raise DatabaseError(f"Failed to get records: {str(e)}")
    
    def update(self, model_instance) -> Any:
        """
        Update a model instance.
        
        Args:
            model_instance: Model instance to update
            
        Returns:
            Updated model instance
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            self.session.commit()
            self.session.refresh(model_instance)
            return model_instance
            
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Update operation failed: {e}")
            raise DatabaseError(f"Failed to update record: {str(e)}")
    
    def delete(self, model_instance) -> bool:
        """
        Delete a model instance.
        
        Args:
            model_instance: Model instance to delete
            
        Returns:
            True if successful
            
        Raises:
            DatabaseError: If deletion fails
        """
        try:
            self.session.delete(model_instance)
            self.session.commit()
            return True
            
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Delete operation failed: {e}")
            raise DatabaseError(f"Failed to delete record: {str(e)}")
    
    def count(self, model_class, filters: Dict[str, Any] = None) -> int:
        """
        Count records with optional filters.
        
        Args:
            model_class: Model class
            filters: Filter conditions
            
        Returns:
            Number of records
        """
        try:
            query = self.session.query(model_class)
            
            if filters:
                for key, value in filters.items():
                    if hasattr(model_class, key):
                        query = query.filter(getattr(model_class, key) == value)
            
            return query.count()
            
        except SQLAlchemyError as e:
            logger.error(f"Count operation failed: {e}")
            raise DatabaseError(f"Failed to count records: {str(e)}")


class DataRepository(BaseRepository):
    """Repository for data collections."""
    
    def get_by_user(self, user_id: str, limit: int = None, offset: int = None) -> List[CollectionModel]:
        """
        Get collections by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of collection models
        """
        try:
            query = self.session.query(CollectionModel).filter(
                CollectionModel.user_id == user_id
            ).order_by(desc(CollectionModel.created_at))
            
            if offset:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get by user operation failed: {e}")
            raise DatabaseError(f"Failed to get collections by user: {str(e)}")
    
    def get_by_source_type(self, user_id: str, source_type: str) -> List[CollectionModel]:
        """
        Get collections by source type.
        
        Args:
            user_id: User ID
            source_type: Source type
            
        Returns:
            List of collection models
        """
        try:
            return self.session.query(CollectionModel).filter(
                and_(
                    CollectionModel.user_id == user_id,
                    CollectionModel.source_type == source_type
                )
            ).order_by(desc(CollectionModel.created_at)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get by source type operation failed: {e}")
            raise DatabaseError(f"Failed to get collections by source type: {str(e)}")
    
    def search_by_name(self, user_id: str, name: str) -> List[CollectionModel]:
        """
        Search collections by name.
        
        Args:
            user_id: User ID
            name: Name to search for
            
        Returns:
            List of collection models
        """
        try:
            return self.session.query(CollectionModel).filter(
                and_(
                    CollectionModel.user_id == user_id,
                    CollectionModel.name.ilike(f"%{name}%")
                )
            ).order_by(desc(CollectionModel.created_at)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Search by name operation failed: {e}")
            raise DatabaseError(f"Failed to search collections by name: {str(e)}")


class CollectionRepository(BaseRepository):
    """Repository for data collections with specialized operations."""
    
    def get_with_executions(self, collection_id: str) -> Optional[CollectionModel]:
        """
        Get collection with its executions.
        
        Args:
            collection_id: Collection ID
            
        Returns:
            Collection model with executions
        """
        try:
            return self.session.query(CollectionModel).filter(
                CollectionModel.id == collection_id
            ).first()
            
        except SQLAlchemyError as e:
            logger.error(f"Get with executions operation failed: {e}")
            raise DatabaseError(f"Failed to get collection with executions: {str(e)}")
    
    def get_recent_collections(self, user_id: str, days: int = 7) -> List[CollectionModel]:
        """
        Get recent collections.
        
        Args:
            user_id: User ID
            days: Number of days to look back
            
        Returns:
            List of recent collection models
        """
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            return self.session.query(CollectionModel).filter(
                and_(
                    CollectionModel.user_id == user_id,
                    CollectionModel.created_at >= cutoff_date
                )
            ).order_by(desc(CollectionModel.created_at)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get recent collections operation failed: {e}")
            raise DatabaseError(f"Failed to get recent collections: {str(e)}")


class ExecutionRepository(BaseRepository):
    """Repository for task executions."""
    
    def get_by_user(self, user_id: str, limit: int = None, offset: int = None) -> List[ExecutionModel]:
        """
        Get executions by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of execution models
        """
        try:
            query = self.session.query(ExecutionModel).filter(
                ExecutionModel.user_id == user_id
            ).order_by(desc(ExecutionModel.created_at))
            
            if offset:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get executions by user operation failed: {e}")
            raise DatabaseError(f"Failed to get executions by user: {str(e)}")
    
    def get_by_task(self, task_id: str) -> List[ExecutionModel]:
        """
        Get executions by task ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            List of execution models
        """
        try:
            return self.session.query(ExecutionModel).filter(
                ExecutionModel.task_id == task_id
            ).order_by(desc(ExecutionModel.created_at)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get executions by task operation failed: {e}")
            raise DatabaseError(f"Failed to get executions by task: {str(e)}")
    
    def get_by_status(self, user_id: str, status: str) -> List[ExecutionModel]:
        """
        Get executions by status.
        
        Args:
            user_id: User ID
            status: Execution status
            
        Returns:
            List of execution models
        """
        try:
            return self.session.query(ExecutionModel).filter(
                and_(
                    ExecutionModel.user_id == user_id,
                    ExecutionModel.status == status
                )
            ).order_by(desc(ExecutionModel.created_at)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get executions by status operation failed: {e}")
            raise DatabaseError(f"Failed to get executions by status: {str(e)}")


class UserRepository(BaseRepository):
    """Repository for users."""
    
    def get_by_username(self, username: str) -> Optional[UserModel]:
        """
        Get user by username.
        
        Args:
            username: Username
            
        Returns:
            User model or None
        """
        try:
            return self.session.query(UserModel).filter(
                UserModel.username == username
            ).first()
            
        except SQLAlchemyError as e:
            logger.error(f"Get user by username operation failed: {e}")
            raise DatabaseError(f"Failed to get user by username: {str(e)}")
    
    def get_by_email(self, email: str) -> Optional[UserModel]:
        """
        Get user by email.
        
        Args:
            email: Email address
            
        Returns:
            User model or None
        """
        try:
            return self.session.query(UserModel).filter(
                UserModel.email == email
            ).first()
            
        except SQLAlchemyError as e:
            logger.error(f"Get user by email operation failed: {e}")
            raise DatabaseError(f"Failed to get user by email: {str(e)}")
    
    def get_active_users(self) -> List[UserModel]:
        """
        Get all active users.
        
        Returns:
            List of active user models
        """
        try:
            return self.session.query(UserModel).filter(
                UserModel.is_active == True
            ).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get active users operation failed: {e}")
            raise DatabaseError(f"Failed to get active users: {str(e)}")


class MLModelRepository(BaseRepository):
    """Repository for ML models."""
    
    def get_by_user(self, user_id: str, limit: int = None, offset: int = None) -> List[MLModelModel]:
        """
        Get ML models by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of ML model models
        """
        try:
            query = self.session.query(MLModelModel).filter(
                MLModelModel.user_id == user_id
            ).order_by(desc(MLModelModel.created_at))
            
            if offset:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get ML models by user operation failed: {e}")
            raise DatabaseError(f"Failed to get ML models by user: {str(e)}")
    
    def get_by_type(self, user_id: str, model_type: str) -> List[MLModelModel]:
        """
        Get ML models by type.
        
        Args:
            user_id: User ID
            model_type: Model type
            
        Returns:
            List of ML model models
        """
        try:
            return self.session.query(MLModelModel).filter(
                and_(
                    MLModelModel.user_id == user_id,
                    MLModelModel.model_type == model_type
                )
            ).order_by(desc(MLModelModel.created_at)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get ML models by type operation failed: {e}")
            raise DatabaseError(f"Failed to get ML models by type: {str(e)}")
    
    def get_active_models(self, user_id: str) -> List[MLModelModel]:
        """
        Get active ML models.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active ML model models
        """
        try:
            return self.session.query(MLModelModel).filter(
                and_(
                    MLModelModel.user_id == user_id,
                    MLModelModel.is_active == True
                )
            ).order_by(desc(MLModelModel.created_at)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get active ML models operation failed: {e}")
            raise DatabaseError(f"Failed to get active ML models: {str(e)}")


class SystemLogRepository(BaseRepository):
    """Repository for system logs."""
    
    def get_by_level(self, level: str, limit: int = None, offset: int = None) -> List[SystemLogModel]:
        """
        Get logs by level.
        
        Args:
            level: Log level
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of system log models
        """
        try:
            query = self.session.query(SystemLogModel).filter(
                SystemLogModel.level == level
            ).order_by(desc(SystemLogModel.timestamp))
            
            if offset:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get logs by level operation failed: {e}")
            raise DatabaseError(f"Failed to get logs by level: {str(e)}")
    
    def get_by_user(self, user_id: str, limit: int = None, offset: int = None) -> List[SystemLogModel]:
        """
        Get logs by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of system log models
        """
        try:
            query = self.session.query(SystemLogModel).filter(
                SystemLogModel.user_id == user_id
            ).order_by(desc(SystemLogModel.timestamp))
            
            if offset:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get logs by user operation failed: {e}")
            raise DatabaseError(f"Failed to get logs by user: {str(e)}")
    
    def get_recent_logs(self, hours: int = 24) -> List[SystemLogModel]:
        """
        Get recent logs.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent system log models
        """
        try:
            from datetime import datetime, timedelta
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            return self.session.query(SystemLogModel).filter(
                SystemLogModel.timestamp >= cutoff_time
            ).order_by(desc(SystemLogModel.timestamp)).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Get recent logs operation failed: {e}")
            raise DatabaseError(f"Failed to get recent logs: {str(e)}") 