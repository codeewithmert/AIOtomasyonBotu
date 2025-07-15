"""
AI Automation Bot - Main Application
Ana uygulama giriş noktası
"""

import sys
import signal
import argparse
from pathlib import Path
from typing import Optional
import logging

from .core.config import init_config, get_config
from .core.logger import init_logger, get_logger
from .core.exceptions import CriticalError
from .automation.scheduler import TaskScheduler
from .ml.pipeline import MLPipeline
from .data.collectors.api_collector import APICollector
from .data.collectors.web_scraper import WebScraper


class AIAutomationBot:
    """
    AI Automation Bot ana sınıfı
    Tüm bileşenleri koordine eder
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.config = None
        self.scheduler = None
        self.ml_pipeline = None
        self.api_collector = None
        self.web_scraper = None
        self.is_running = False
        
        # Sinyal handler'ları
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Konfigürasyonu yükle
        self._load_config(config_path)
        
        # Bileşenleri başlat
        self._initialize_components()
        
        self.logger.info("AI Automation Bot başlatıldı")
    
    def _load_config(self, config_path: Optional[str] = None):
        """Konfigürasyonu yükle"""
        try:
            # Konfigürasyon yöneticisini başlat
            config_manager = init_config(config_path)
            self.config = config_manager.get_config()
            
            # Logger'ı başlat
            init_logger(self.config.logging)
            
            self.logger.info(f"Konfigürasyon yüklendi: {self.config.name} v{self.config.version}")
            
        except Exception as e:
            raise CriticalError(
                f"Konfigürasyon yükleme hatası: {e}",
                component="config"
            )
    
    def _initialize_components(self):
        """Bileşenleri başlat"""
        try:
            # Task Scheduler
            self.scheduler = TaskScheduler(self.config.automation)
            
            # ML Pipeline
            self.ml_pipeline = MLPipeline(self.config.ml)
            
            # Data Collectors
            self.api_collector = APICollector(self.config.api)
            self.web_scraper = WebScraper()
            
            self.logger.info("Tüm bileşenler başlatıldı")
            
        except Exception as e:
            raise CriticalError(
                f"Bileşen başlatma hatası: {e}",
                component="components"
            )
    
    def _signal_handler(self, signum, frame):
        """Sinyal handler"""
        self.logger.info(f"Sinyal alındı: {signum}")
        self.stop()
    
    def setup_default_tasks(self):
        """Varsayılan görevleri kur"""
        try:
            # Veri toplama görevi
            def data_collection_task():
                self.logger.info("Veri toplama görevi başlatılıyor")
                # Burada veri toplama işlemleri yapılacak
                pass
            
            # Model eğitimi görevi
            def model_training_task():
                self.logger.info("Model eğitimi görevi başlatılıyor")
                # Burada model eğitimi yapılacak
                pass
            
            # Rapor oluşturma görevi
            def report_generation_task():
                self.logger.info("Rapor oluşturma görevi başlatılıyor")
                # Burada rapor oluşturma yapılacak
                pass
            
            # Görevleri ekle
            task_configs = self.config.automation.tasks
            
            if task_configs.get('data_collection', {}).get('enabled', False):
                self.scheduler.add_task(
                    name="data_collection",
                    func=data_collection_task,
                    schedule_expr=task_configs['data_collection']['schedule'],
                    max_retries=task_configs['data_collection'].get('retry_attempts', 3)
                )
            
            if task_configs.get('model_training', {}).get('enabled', False):
                self.scheduler.add_task(
                    name="model_training",
                    func=model_training_task,
                    schedule_expr=task_configs['model_training']['schedule'],
                    max_retries=task_configs['model_training'].get('retry_attempts', 3)
                )
            
            # Rapor görevi (günlük)
            self.scheduler.add_task(
                name="report_generation",
                func=report_generation_task,
                schedule_expr="0 9 * * *",  # Her gün saat 9'da
                max_retries=2
            )
            
            self.logger.info("Varsayılan görevler kuruldu")
            
        except Exception as e:
            self.logger.error(f"Varsayılan görev kurma hatası: {e}")
    
    def start(self):
        """Bot'u başlat"""
        if self.is_running:
            self.logger.warning("Bot zaten çalışıyor")
            return
        
        try:
            self.logger.info("AI Automation Bot başlatılıyor...")
            
            # Varsayılan görevleri kur
            self.setup_default_tasks()
            
            # Scheduler'ı başlat
            self.scheduler.start()
            
            self.is_running = True
            self.logger.info("AI Automation Bot başlatıldı ve çalışıyor")
            
        except Exception as e:
            raise CriticalError(
                f"Bot başlatma hatası: {e}",
                component="main"
            )
    
    def stop(self):
        """Bot'u durdur"""
        if not self.is_running:
            return
        
        try:
            self.logger.info("AI Automation Bot durduruluyor...")
            
            # Scheduler'ı durdur
            if self.scheduler:
                self.scheduler.stop()
            
            # Diğer bileşenleri temizle
            if self.api_collector:
                self.api_collector.close()
            
            if self.web_scraper:
                self.web_scraper.close()
            
            self.is_running = False
            self.logger.info("AI Automation Bot durduruldu")
            
        except Exception as e:
            self.logger.error(f"Bot durdurma hatası: {e}")
    
    def run_task(self, task_name: str):
        """Belirli bir görevi çalıştır"""
        if not self.scheduler:
            raise CriticalError("Scheduler başlatılmamış")
        
        try:
            result = self.scheduler.run_task(task_name)
            self.logger.info(f"Görev çalıştırıldı: {task_name} - {result['status']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Görev çalıştırma hatası: {task_name} - {e}")
            raise
    
    def get_status(self) -> dict:
        """Bot durumunu al"""
        status = {
            'is_running': self.is_running,
            'config': {
                'name': self.config.name,
                'version': self.config.version,
                'environment': self.config.environment
            },
            'components': {
                'scheduler': self.scheduler is not None,
                'ml_pipeline': self.ml_pipeline is not None,
                'api_collector': self.api_collector is not None,
                'web_scraper': self.web_scraper is not None
            }
        }
        
        if self.scheduler:
            status['scheduler_stats'] = self.scheduler.get_task_stats()
        
        return status
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='AI Automation Bot')
    parser.add_argument('--config', '-c', help='Konfigürasyon dosyası yolu')
    parser.add_argument('--task', '-t', help='Çalıştırılacak görev adı')
    parser.add_argument('--status', '-s', action='store_true', help='Bot durumunu göster')
    parser.add_argument('--daemon', '-d', action='store_true', help='Daemon modunda çalıştır')
    
    args = parser.parse_args()
    
    try:
        # Bot'u oluştur
        bot = AIAutomationBot(config_path=args.config)
        
        if args.status:
            # Durum göster
            status = bot.get_status()
            print("AI Automation Bot Durumu:")
            print(f"  Çalışıyor: {status['is_running']}")
            print(f"  Versiyon: {status['config']['version']}")
            print(f"  Ortam: {status['config']['environment']}")
            
            if 'scheduler_stats' in status:
                print(f"  Toplam Görev: {status['scheduler_stats']['total_tasks']}")
                print(f"  Etkin Görev: {status['scheduler_stats']['enabled_tasks']}")
        
        elif args.task:
            # Belirli görevi çalıştır
            result = bot.run_task(args.task)
            print(f"Görev sonucu: {result}")
        
        elif args.daemon:
            # Daemon modunda çalıştır
            print("AI Automation Bot daemon modunda başlatılıyor...")
            bot.start()
            
            # Sonsuz döngü
            try:
                while bot.is_running:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nBot durduruluyor...")
                bot.stop()
        
        else:
            # Normal modda çalıştır
            print("AI Automation Bot başlatılıyor...")
            with bot:
                print("Bot çalışıyor. Durdurmak için Ctrl+C'ye basın.")
                try:
                    while True:
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nBot durduruluyor...")
    
    except CriticalError as e:
        print(f"Kritik hata: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 