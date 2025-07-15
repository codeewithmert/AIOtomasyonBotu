"""
Configuration Management System
Merkezi konfigürasyon yönetimi için gelişmiş sistem
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, validator
import logging
from dotenv import load_dotenv

# Environment variables yükle
load_dotenv()


class DatabaseConfig(BaseModel):
    """Veritabanı konfigürasyonu"""
    host: str = "localhost"
    port: int = 5432
    name: str = "automation_db"
    user: str = "user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    
    @validator('password', pre=True)
    def get_password_from_env(cls, v):
        return v or os.getenv('DB_PASSWORD', '')


class APIConfig(BaseModel):
    """API konfigürasyonu"""
    base_url: str = "https://api.example.com"
    timeout: int = 30
    retry_attempts: int = 3
    api_key: Optional[str] = None
    
    @validator('api_key', pre=True)
    def get_api_key_from_env(cls, v):
        return v or os.getenv('API_KEY')


class MLConfig(BaseModel):
    """Makine öğrenmesi konfigürasyonu"""
    models: List[str] = field(default_factory=lambda: ["random_forest", "xgboost"])
    hyperparameter_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "method": "optuna",
        "n_trials": 100,
        "timeout": 3600
    })
    evaluation: Dict[str, Any] = field(default_factory=lambda: {
        "test_size": 0.2,
        "cv_folds": 5,
        "random_state": 42
    })
    model_storage: str = "models/"


class AutomationConfig(BaseModel):
    """Otomasyon konfigürasyonu"""
    scheduler: Dict[str, Any] = field(default_factory=lambda: {
        "timezone": "UTC",
        "max_workers": 4,
        "job_defaults": {
            "coalesce": True,
            "max_instances": 3
        }
    })
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "data_collection": {
            "schedule": "0 */6 * * *",
            "enabled": True,
            "retry_attempts": 3
        },
        "model_training": {
            "schedule": "0 2 * * *",
            "enabled": True,
            "retry_attempts": 3
        }
    })


class ReportingConfig(BaseModel):
    """Raporlama konfigürasyonu"""
    output_dir: str = "reports/"
    formats: List[str] = field(default_factory=lambda: ["pdf", "excel", "html"])
    templates_dir: str = "templates/"
    
    notifications: Dict[str, Any] = field(default_factory=lambda: {
        "email": {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "",
            "password": ""
        },
        "slack": {
            "enabled": False,
            "webhook_url": ""
        }
    })


class SecurityConfig(BaseModel):
    """Güvenlik konfigürasyonu"""
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    session_timeout: int = 3600
    
    @validator('encryption_key', pre=True)
    def get_encryption_key_from_env(cls, v):
        return v or os.getenv('ENCRYPTION_KEY')
    
    @validator('jwt_secret', pre=True)
    def get_jwt_secret_from_env(cls, v):
        return v or os.getenv('JWT_SECRET')


class LoggingConfig(BaseModel):
    """Loglama konfigürasyonu"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/automation_bot.log"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    rotation: str = "1 day"


class AppConfig(BaseModel):
    """Ana uygulama konfigürasyonu"""
    name: str = "AI Automation Bot"
    version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # Alt konfigürasyonlar
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigManager:
    """
    Konfigürasyon yöneticisi
    Çoklu kaynak desteği ile merkezi konfigürasyon yönetimi
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/config.yaml"
        self._config: Optional[AppConfig] = None
        self._logger = logging.getLogger(__name__)
        
    def load_config(self) -> AppConfig:
        """Konfigürasyonu yükle"""
        if self._config is None:
            self._config = self._load_from_file()
            self._validate_config()
            self._setup_logging()
            self._logger.info(f"Konfigürasyon yüklendi: {self.config_path}")
        
        return self._config
    
    def _load_from_file(self) -> AppConfig:
        """Dosyadan konfigürasyon yükle"""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            self._logger.warning(f"Konfigürasyon dosyası bulunamadı: {config_path}")
            return self._create_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Environment variables ile override et
            config_data = self._override_with_env(config_data)
            
            return AppConfig(**config_data)
            
        except Exception as e:
            self._logger.error(f"Konfigürasyon yükleme hatası: {e}")
            return self._create_default_config()
    
    def _override_with_env(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Environment variables ile konfigürasyonu override et"""
        env_mappings = {
            'APP_DEBUG': ('app', 'debug'),
            'APP_ENVIRONMENT': ('app', 'environment'),
            'DB_HOST': ('database', 'host'),
            'DB_PORT': ('database', 'port'),
            'DB_NAME': ('database', 'name'),
            'DB_USER': ('database', 'user'),
            'API_BASE_URL': ('api', 'base_url'),
            'API_TIMEOUT': ('api', 'timeout'),
            'LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Nested dictionary path'i takip et
                current = config_data
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Değeri set et
                key = config_path[-1]
                if key in ['debug', 'port', 'timeout']:
                    current[key] = self._parse_value(env_value)
                else:
                    current[key] = env_value
        
        return config_data
    
    def _parse_value(self, value: str) -> Union[str, int, bool]:
        """String değeri uygun tipe çevir"""
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        try:
            return int(value)
        except ValueError:
            return value
    
    def _create_default_config(self) -> AppConfig:
        """Varsayılan konfigürasyon oluştur"""
        self._logger.info("Varsayılan konfigürasyon kullanılıyor")
        return AppConfig()
    
    def _validate_config(self):
        """Konfigürasyonu doğrula"""
        if not self._config:
            return
        
        # Gerekli dizinleri oluştur
        Path(self._config.logging.file).parent.mkdir(parents=True, exist_ok=True)
        Path(self._config.reporting.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self._config.ml.model_storage).mkdir(parents=True, exist_ok=True)
        
        # Güvenlik kontrolleri
        if self._config.environment == "production":
            if not self._config.security.encryption_key:
                self._logger.warning("Production ortamında encryption key tanımlanmamış")
            if not self._config.security.jwt_secret:
                self._logger.warning("Production ortamında JWT secret tanımlanmamış")
    
    def _setup_logging(self):
        """Loglama sistemini kur"""
        if not self._config:
            return
        
        log_config = self._config.logging
        
        # Log dizinini oluştur
        log_file = Path(log_config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Logging konfigürasyonu
        logging.basicConfig(
            level=getattr(logging, log_config.level.upper()),
            format=log_config.format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def get_config(self) -> AppConfig:
        """Konfigürasyonu döndür"""
        return self.load_config()
    
    def reload_config(self) -> AppConfig:
        """Konfigürasyonu yeniden yükle"""
        self._config = None
        return self.load_config()
    
    def save_config(self, config: AppConfig, path: Optional[str] = None):
        """Konfigürasyonu dosyaya kaydet"""
        save_path = path or self.config_path
        
        # Pydantic modelini dict'e çevir
        config_dict = config.dict()
        
        # Dosyaya kaydet
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        self._logger.info(f"Konfigürasyon kaydedildi: {save_path}")
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Belirli bir bölümü döndür"""
        config = self.get_config()
        if hasattr(config, section_name):
            section = getattr(config, section_name)
            return section.dict() if hasattr(section, 'dict') else section
        return {}
    
    def update_section(self, section_name: str, updates: Dict[str, Any]):
        """Belirli bir bölümü güncelle"""
        config = self.get_config()
        if hasattr(config, section_name):
            section = getattr(config, section_name)
            if hasattr(section, 'dict'):
                current_data = section.dict()
                current_data.update(updates)
                # Yeni section oluştur
                section_class = type(section)
                new_section = section_class(**current_data)
                setattr(config, section_name, new_section)
                
                # Konfigürasyonu kaydet
                self.save_config(config)
                self._logger.info(f"{section_name} bölümü güncellendi")


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> AppConfig:
    """Global config instance'ını döndür"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get_config()


def get_config_manager() -> ConfigManager:
    """Global config manager instance'ını döndür"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_config(config_path: Optional[str] = None) -> ConfigManager:
    """Konfigürasyonu başlat"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager


# Convenience functions
def get_database_config() -> DatabaseConfig:
    """Veritabanı konfigürasyonunu döndür"""
    return get_config().database


def get_api_config() -> APIConfig:
    """API konfigürasyonunu döndür"""
    return get_config().api


def get_ml_config() -> MLConfig:
    """ML konfigürasyonunu döndür"""
    return get_config().ml


def get_automation_config() -> AutomationConfig:
    """Otomasyon konfigürasyonunu döndür"""
    return get_config().automation


def get_reporting_config() -> ReportingConfig:
    """Raporlama konfigürasyonunu döndür"""
    return get_config().reporting


def get_security_config() -> SecurityConfig:
    """Güvenlik konfigürasyonunu döndür"""
    return get_config().security


def get_logging_config() -> LoggingConfig:
    """Loglama konfigürasyonunu döndür"""
    return get_config().logging


if __name__ == "__main__":
    # Test
    config = get_config()
    print(f"App Name: {config.name}")
    print(f"Database Host: {config.database.host}")
    print(f"ML Models: {config.ml.models}") 