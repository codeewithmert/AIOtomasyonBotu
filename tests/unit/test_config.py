"""
Configuration Management Unit Tests
Konfigürasyon yönetimi için unit testler
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.config import (
    ConfigManager, AppConfig, DatabaseConfig, APIConfig, 
    MLConfig, AutomationConfig, ReportingConfig, SecurityConfig, LoggingConfig,
    get_config, init_config
)


class TestDatabaseConfig:
    """DatabaseConfig test sınıfı"""
    
    def test_default_values(self):
        """Varsayılan değerleri test et"""
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "automation_db"
        assert config.user == "user"
        assert config.password == ""
        assert config.pool_size == 10
        assert config.max_overflow == 20
    
    def test_custom_values(self):
        """Özel değerleri test et"""
        config = DatabaseConfig(
            host="test-host",
            port=5433,
            name="test_db",
            user="test_user",
            password="test_pass",
            pool_size=20,
            max_overflow=30
        )
        
        assert config.host == "test-host"
        assert config.port == 5433
        assert config.name == "test_db"
        assert config.user == "test_user"
        assert config.password == "test_pass"
        assert config.pool_size == 20
        assert config.max_overflow == 30
    
    @patch.dict(os.environ, {'DB_PASSWORD': 'env_password'})
    def test_password_from_env(self):
        """Environment variable'dan password al"""
        config = DatabaseConfig()
        assert config.password == "env_password"
    
    def test_validation(self):
        """Validasyon testleri"""
        # Geçerli port
        config = DatabaseConfig(port=5432)
        assert config.port == 5432
        
        # Geçersiz port (negatif)
        with pytest.raises(ValueError):
            DatabaseConfig(port=-1)


class TestAPIConfig:
    """APIConfig test sınıfı"""
    
    def test_default_values(self):
        """Varsayılan değerleri test et"""
        config = APIConfig()
        
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 30
        assert config.retry_attempts == 3
        assert config.api_key is None
    
    @patch.dict(os.environ, {'API_KEY': 'test_api_key'})
    def test_api_key_from_env(self):
        """Environment variable'dan API key al"""
        config = APIConfig()
        assert config.api_key == "test_api_key"


class TestMLConfig:
    """MLConfig test sınıfı"""
    
    def test_default_values(self):
        """Varsayılan değerleri test et"""
        config = MLConfig()
        
        assert "random_forest" in config.models
        assert "xgboost" in config.models
        assert config.hyperparameter_optimization["method"] == "optuna"
        assert config.hyperparameter_optimization["n_trials"] == 100
        assert config.evaluation["test_size"] == 0.2
        assert config.evaluation["cv_folds"] == 5
        assert config.model_storage == "models/"


class TestAutomationConfig:
    """AutomationConfig test sınıfı"""
    
    def test_default_values(self):
        """Varsayılan değerleri test et"""
        config = AutomationConfig()
        
        assert config.scheduler["timezone"] == "UTC"
        assert config.scheduler["max_workers"] == 4
        assert config.tasks["data_collection"]["enabled"] is True
        assert config.tasks["model_training"]["enabled"] is True


class TestReportingConfig:
    """ReportingConfig test sınıfı"""
    
    def test_default_values(self):
        """Varsayılan değerleri test et"""
        config = ReportingConfig()
        
        assert config.output_dir == "reports/"
        assert "pdf" in config.formats
        assert "excel" in config.formats
        assert "html" in config.formats
        assert config.notifications["email"]["enabled"] is False
        assert config.notifications["slack"]["enabled"] is False


class TestSecurityConfig:
    """SecurityConfig test sınıfı"""
    
    def test_default_values(self):
        """Varsayılan değerleri test et"""
        config = SecurityConfig()
        
        assert config.encryption_key is None
        assert config.jwt_secret is None
        assert config.session_timeout == 3600
    
    @patch.dict(os.environ, {
        'ENCRYPTION_KEY': 'test_encryption_key',
        'JWT_SECRET': 'test_jwt_secret'
    })
    def test_secrets_from_env(self):
        """Environment variable'dan secret'ları al"""
        config = SecurityConfig()
        assert config.encryption_key == "test_encryption_key"
        assert config.jwt_secret == "test_jwt_secret"


class TestLoggingConfig:
    """LoggingConfig test sınıfı"""
    
    def test_default_values(self):
        """Varsayılan değerleri test et"""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert "asctime" in config.format
        assert config.file == "logs/automation_bot.log"
        assert config.max_size == 10 * 1024 * 1024  # 10MB
        assert config.backup_count == 5
        assert config.rotation == "1 day"


class TestAppConfig:
    """AppConfig test sınıfı"""
    
    def test_default_values(self):
        """Varsayılan değerleri test et"""
        config = AppConfig()
        
        assert config.name == "AI Automation Bot"
        assert config.version == "1.0.0"
        assert config.debug is False
        assert config.environment == "development"
        
        # Alt konfigürasyonların varlığını kontrol et
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.ml, MLConfig)
        assert isinstance(config.automation, AutomationConfig)
        assert isinstance(config.reporting, ReportingConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.logging, LoggingConfig)


class TestConfigManager:
    """ConfigManager test sınıfı"""
    
    def setup_method(self):
        """Her test öncesi çalışır"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def teardown_method(self):
        """Her test sonrası çalışır"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_default_path(self):
        """Varsayılan path ile başlat"""
        manager = ConfigManager()
        assert manager.config_path == "config/config.yaml"
    
    def test_init_with_custom_path(self):
        """Özel path ile başlat"""
        manager = ConfigManager("custom/path/config.yaml")
        assert manager.config_path == "custom/path/config.yaml"
    
    def test_load_config_file_not_exists(self):
        """Dosya yoksa varsayılan konfigürasyon kullan"""
        manager = ConfigManager(str(self.config_path))
        config = manager.load_config()
        
        assert isinstance(config, AppConfig)
        assert config.name == "AI Automation Bot"
    
    def test_load_config_from_file(self):
        """Dosyadan konfigürasyon yükle"""
        # Test konfigürasyon dosyası oluştur
        test_config = {
            'app': {
                'name': 'Test Bot',
                'version': '2.0.0',
                'debug': True,
                'environment': 'testing'
            },
            'database': {
                'host': 'test-host',
                'port': 5433,
                'name': 'test_db'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Konfigürasyonu yükle
        manager = ConfigManager(str(self.config_path))
        config = manager.load_config()
        
        assert config.name == "Test Bot"
        assert config.version == "2.0.0"
        assert config.debug is True
        assert config.environment == "testing"
        assert config.database.host == "test-host"
        assert config.database.port == 5433
        assert config.database.name == "test_db"
    
    @patch.dict(os.environ, {
        'APP_DEBUG': 'true',
        'APP_ENVIRONMENT': 'production',
        'DB_HOST': 'env-host',
        'DB_PORT': '5434',
        'API_BASE_URL': 'https://env-api.com'
    })
    def test_override_with_env_variables(self):
        """Environment variables ile override et"""
        test_config = {
            'app': {
                'name': 'Test Bot',
                'debug': False,
                'environment': 'development'
            },
            'database': {
                'host': 'file-host',
                'port': 5432
            },
            'api': {
                'base_url': 'https://file-api.com'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        manager = ConfigManager(str(self.config_path))
        config = manager.load_config()
        
        # Environment variables ile override edilmiş değerler
        assert config.debug is True
        assert config.environment == "production"
        assert config.database.host == "env-host"
        assert config.database.port == 5434
        assert config.api.base_url == "https://env-api.com"
        
        # Override edilmemiş değerler
        assert config.name == "Test Bot"
    
    def test_save_config(self):
        """Konfigürasyonu kaydet"""
        manager = ConfigManager(str(self.config_path))
        config = AppConfig()
        config.name = "Saved Bot"
        config.version = "3.0.0"
        
        manager.save_config(config)
        
        # Kaydedilen dosyayı kontrol et
        assert self.config_path.exists()
        
        with open(self.config_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['app']['name'] == "Saved Bot"
        assert saved_data['app']['version'] == "3.0.0"
    
    def test_get_section(self):
        """Belirli bölümü al"""
        manager = ConfigManager(str(self.config_path))
        config = manager.load_config()
        
        database_section = manager.get_section('database')
        assert 'host' in database_section
        assert database_section['host'] == "localhost"
    
    def test_update_section(self):
        """Bölümü güncelle"""
        manager = ConfigManager(str(self.config_path))
        config = manager.load_config()
        
        updates = {
            'host': 'updated-host',
            'port': 5435
        }
        
        manager.update_section('database', updates)
        
        # Güncellenmiş değerleri kontrol et
        updated_config = manager.get_config()
        assert updated_config.database.host == "updated-host"
        assert updated_config.database.port == 5435
    
    def test_reload_config(self):
        """Konfigürasyonu yeniden yükle"""
        manager = ConfigManager(str(self.config_path))
        
        # İlk yükleme
        config1 = manager.load_config()
        
        # Dosyayı güncelle
        test_config = {
            'app': {
                'name': 'Reloaded Bot',
                'version': '4.0.0'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Yeniden yükle
        config2 = manager.reload_config()
        
        assert config2.name == "Reloaded Bot"
        assert config2.version == "4.0.0"


class TestGlobalFunctions:
    """Global fonksiyonlar test sınıfı"""
    
    def test_get_config(self):
        """get_config fonksiyonunu test et"""
        config = get_config()
        assert isinstance(config, AppConfig)
        assert config.name == "AI Automation Bot"
    
    def test_init_config(self):
        """init_config fonksiyonunu test et"""
        manager = init_config()
        assert isinstance(manager, ConfigManager)
        
        config = manager.get_config()
        assert isinstance(config, AppConfig)
    
    def test_get_database_config(self):
        """get_database_config fonksiyonunu test et"""
        from src.core.config import get_database_config
        
        config = get_database_config()
        assert isinstance(config, DatabaseConfig)
        assert config.host == "localhost"
    
    def test_get_api_config(self):
        """get_api_config fonksiyonunu test et"""
        from src.core.config import get_api_config
        
        config = get_api_config()
        assert isinstance(config, APIConfig)
        assert config.base_url == "https://api.example.com"
    
    def test_get_ml_config(self):
        """get_ml_config fonksiyonunu test et"""
        from src.core.config import get_ml_config
        
        config = get_ml_config()
        assert isinstance(config, MLConfig)
        assert "random_forest" in config.models


if __name__ == "__main__":
    pytest.main([__file__]) 