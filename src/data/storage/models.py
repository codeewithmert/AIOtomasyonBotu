"""
Data Models

SQLAlchemy models for data storage in the AI Automation Bot.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DataModel(Base):
    """Base model with common fields."""
    
    __abstract__ = True
    
    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class CollectionModel(DataModel):
    """Model for data collections."""
    
    __tablename__ = 'collections'
    
    user_id = Column(String(36), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    source_type = Column(String(50), nullable=False)  # 'api', 'web', 'file'
    source_config = Column(JSON, nullable=False)
    data = Column(JSON)
    metadata = Column(JSON)
    status = Column(String(20), default='active')  # 'active', 'archived', 'deleted'
    
    # Relationships
    executions = relationship("ExecutionModel", back_populates="collection")
    
    def __repr__(self):
        return f"<CollectionModel(id='{self.id}', name='{self.name}', source_type='{self.source_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'source_type': self.source_type,
            'source_config': self.source_config,
            'data': self.data,
            'metadata': self.metadata,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ExecutionModel(DataModel):
    """Model for task executions."""
    
    __tablename__ = 'executions'
    
    user_id = Column(String(36), nullable=False, index=True)
    task_id = Column(String(36), nullable=False, index=True)
    collection_id = Column(String(36), ForeignKey('collections.id'), nullable=True)
    task_type = Column(String(50), nullable=False)  # 'data_collection', 'ml_training', 'reporting'
    status = Column(String(20), default='pending')  # 'pending', 'running', 'completed', 'failed'
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    result = Column(JSON)
    error = Column(Text)
    metadata = Column(JSON)
    
    # Relationships
    collection = relationship("CollectionModel", back_populates="executions")
    
    def __repr__(self):
        return f"<ExecutionModel(id='{self.id}', task_id='{self.task_id}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'task_id': self.task_id,
            'collection_id': self.collection_id,
            'task_type': self.task_type,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class UserModel(DataModel):
    """Model for users."""
    
    __tablename__ = 'users'
    
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    roles = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    
    def __repr__(self):
        return f"<UserModel(id='{self.id}', username='{self.username}', email='{self.email}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'roles': self.roles,
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class MLModelModel(DataModel):
    """Model for ML models."""
    
    __tablename__ = 'ml_models'
    
    user_id = Column(String(36), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    model_type = Column(String(50), nullable=False)  # 'random_forest', 'linear_regression', etc.
    task_type = Column(String(20), nullable=False)  # 'classification', 'regression'
    hyperparameters = Column(JSON)
    metrics = Column(JSON)
    model_path = Column(String(500))
    is_active = Column(Boolean, default=True)
    version = Column(String(20), default='1.0.0')
    
    def __repr__(self):
        return f"<MLModelModel(id='{self.id}', name='{self.name}', model_type='{self.model_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'model_path': self.model_path,
            'is_active': self.is_active,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class AutomationTaskModel(DataModel):
    """Model for automation tasks."""
    
    __tablename__ = 'automation_tasks'
    
    user_id = Column(String(36), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    task_type = Column(String(50), nullable=False)  # 'data_collection', 'ml_training', 'reporting'
    schedule = Column(JSON, nullable=False)
    config = Column(JSON)
    enabled = Column(Boolean, default=True)
    last_execution = Column(DateTime)
    next_execution = Column(DateTime)
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<AutomationTaskModel(id='{self.id}', name='{self.name}', task_type='{self.task_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type,
            'schedule': self.schedule,
            'config': self.config,
            'enabled': self.enabled,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'next_execution': self.next_execution.isoformat() if self.next_execution else None,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ReportModel(DataModel):
    """Model for reports."""
    
    __tablename__ = 'reports'
    
    user_id = Column(String(36), nullable=False, index=True)
    template_id = Column(String(36), nullable=False)
    name = Column(String(255), nullable=False)
    output_format = Column(String(20), nullable=False)  # 'html', 'pdf', 'excel', 'json'
    parameters = Column(JSON)
    file_path = Column(String(500))
    file_size = Column(Integer)
    download_count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<ReportModel(id='{self.id}', name='{self.name}', output_format='{self.output_format}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'template_id': self.template_id,
            'name': self.name,
            'output_format': self.output_format,
            'parameters': self.parameters,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'download_count': self.download_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ReportTemplateModel(DataModel):
    """Model for report templates."""
    
    __tablename__ = 'report_templates'
    
    user_id = Column(String(36), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    template_type = Column(String(20), nullable=False)  # 'html', 'pdf', 'excel', 'json'
    content = Column(Text, nullable=False)
    parameters = Column(JSON)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<ReportTemplateModel(id='{self.id}', name='{self.name}', template_type='{self.template_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'template_type': self.template_type,
            'content': self.content,
            'parameters': self.parameters,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ReportScheduleModel(DataModel):
    """Model for report schedules."""
    
    __tablename__ = 'report_schedules'
    
    user_id = Column(String(36), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    template_id = Column(String(36), nullable=False)
    schedule = Column(JSON, nullable=False)
    parameters = Column(JSON)
    output_format = Column(String(20), nullable=False)
    recipients = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    last_generated = Column(DateTime)
    next_generation = Column(DateTime)
    
    def __repr__(self):
        return f"<ReportScheduleModel(id='{self.id}', name='{self.name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'template_id': self.template_id,
            'schedule': self.schedule,
            'parameters': self.parameters,
            'output_format': self.output_format,
            'recipients': self.recipients,
            'is_active': self.is_active,
            'last_generated': self.last_generated.isoformat() if self.last_generated else None,
            'next_generation': self.next_generation.isoformat() if self.next_generation else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class SystemLogModel(DataModel):
    """Model for system logs."""
    
    __tablename__ = 'system_logs'
    
    level = Column(String(20), nullable=False, index=True)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    logger_name = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    user_id = Column(String(36), nullable=True, index=True)
    request_id = Column(String(36), nullable=True, index=True)
    extra_data = Column(JSON)
    
    def __repr__(self):
        return f"<SystemLogModel(id='{self.id}', level='{self.level}', message='{self.message[:50]}...')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'level': self.level,
            'logger_name': self.logger_name,
            'message': self.message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'user_id': self.user_id,
            'request_id': self.request_id,
            'extra_data': self.extra_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 