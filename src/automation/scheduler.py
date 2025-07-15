"""
Task Scheduler
Görev zamanlama ve yönetim sistemi
"""

import time
import threading
import queue
import signal
import sys
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.threadpool import ThreadPoolExecutor as APSThreadPoolExecutor

from ..core.logger import get_logger, performance_context
from ..core.exceptions import TaskSchedulingError, TaskExecutionError, RetryableError
from ..core.utils import retry_on_failure
from ..core.config import get_automation_config


class Task:
    """Görev sınıfı"""
    
    def __init__(self, name: str, func: Callable, schedule_expr: str = None,
                 args: tuple = None, kwargs: dict = None, 
                 max_retries: int = 3, retry_delay: float = 1.0,
                 timeout: float = None, enabled: bool = True):
        self.name = name
        self.func = func
        self.schedule_expr = schedule_expr
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.enabled = enabled
        
        # Görev durumu
        self.last_run = None
        self.next_run = None
        self.run_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self.last_error = None
        
        # Görev geçmişi
        self.execution_history = []
    
    def execute(self) -> Dict[str, Any]:
        """Görevi çalıştır"""
        start_time = time.time()
        execution_id = f"{self.name}_{int(start_time)}"
        
        result = {
            'execution_id': execution_id,
            'task_name': self.name,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'result': None,
            'error': None,
            'execution_time': 0.0,
            'retry_count': 0
        }
        
        try:
            # Retry mekanizması
            for attempt in range(self.max_retries + 1):
                try:
                    if self.timeout:
                        # Timeout ile çalıştır
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(self.func, *self.args, **self.kwargs)
                            result['result'] = future.result(timeout=self.timeout)
                    else:
                        # Normal çalıştır
                        result['result'] = self.func(*self.args, **self.kwargs)
                    
                    # Başarılı çalıştırma
                    result['status'] = 'success'
                    result['retry_count'] = attempt
                    self.success_count += 1
                    break
                    
                except Exception as e:
                    result['retry_count'] = attempt
                    result['error'] = str(e)
                    
                    if attempt < self.max_retries:
                        # Retry yapılabilir hata
                        if isinstance(e, RetryableError):
                            time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                            continue
                        else:
                            # Retry yapılamaz hata
                            break
                    else:
                        # Maksimum retry sayısına ulaşıldı
                        result['status'] = 'failed'
                        self.failure_count += 1
                        self.last_error = str(e)
                        break
            
            # Çalıştırma süresini hesapla
            execution_time = time.time() - start_time
            result['execution_time'] = execution_time
            self.total_execution_time += execution_time
            
            # Görev durumunu güncelle
            self.last_run = datetime.now()
            self.run_count += 1
            
            # Geçmişe ekle
            self.execution_history.append(result)
            if len(self.execution_history) > 100:  # Son 100 çalıştırmayı tut
                self.execution_history = self.execution_history[-100:]
            
            return result
            
        except Exception as e:
            # Beklenmeyen hata
            result['status'] = 'error'
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            self.failure_count += 1
            self.last_error = str(e)
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Görev istatistiklerini al"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'run_count': self.run_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / max(self.run_count, 1),
            'avg_execution_time': self.total_execution_time / max(self.run_count, 1),
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'last_error': self.last_error
        }


class TaskScheduler:
    """
    Görev zamanlayıcı
    Çoklu backend desteği ile gelişmiş zamanlama sistemi
    """
    
    def __init__(self, config=None, backend: str = 'apscheduler'):
        self.config = config or get_automation_config()
        self.backend = backend.lower()
        self.logger = get_logger(__name__)
        
        # Görev yönetimi
        self.tasks: Dict[str, Task] = {}
        self.scheduler = None
        self.is_running = False
        
        # Thread yönetimi
        self.executor = ThreadPoolExecutor(max_workers=self.config.scheduler.get('max_workers', 4))
        self.task_queue = queue.Queue()
        self.worker_thread = None
        
        # Scheduler'ı başlat
        self._setup_scheduler()
        
        self.logger.info(f"Task Scheduler başlatıldı: {self.backend}")
    
    def _setup_scheduler(self):
        """Scheduler backend'ini kur"""
        if self.backend == 'apscheduler':
            # APScheduler backend
            jobstores = {
                'default': MemoryJobStore()
            }
            
            executors = {
                'default': APSThreadPoolExecutor(max_workers=self.config.scheduler.get('max_workers', 4))
            }
            
            job_defaults = {
                'coalesce': True,
                'max_instances': 3,
                'misfire_grace_time': 60
            }
            
            self.scheduler = BackgroundScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=self.config.scheduler.get('timezone', 'UTC')
            )
            
        elif self.backend == 'schedule':
            # Schedule backend (basit)
            self.scheduler = schedule
        else:
            raise ValueError(f"Desteklenmeyen backend: {self.backend}")
    
    def add_task(self, name: str, func: Callable, schedule_expr: str = None,
                 args: tuple = None, kwargs: dict = None, **task_kwargs) -> Task:
        """
        Görev ekle
        
        Args:
            name: Görev adı
            func: Çalıştırılacak fonksiyon
            schedule_expr: Zamanlama ifadesi (cron, interval, date)
            args: Fonksiyon argümanları
            kwargs: Fonksiyon keyword argümanları
            **task_kwargs: Ek görev parametreleri
            
        Returns:
            Eklenen görev
        """
        if name in self.tasks:
            raise TaskSchedulingError(f"Görev zaten mevcut: {name}")
        
        # Görev oluştur
        task = Task(
            name=name,
            func=func,
            schedule_expr=schedule_expr,
            args=args,
            kwargs=kwargs,
            **task_kwargs
        )
        
        self.tasks[name] = task
        
        # Scheduler'a ekle
        if self.backend == 'apscheduler':
            self._add_apscheduler_job(task)
        elif self.backend == 'schedule':
            self._add_schedule_job(task)
        
        self.logger.info(f"Görev eklendi: {name} - {schedule_expr}")
        return task
    
    def _add_apscheduler_job(self, task: Task):
        """APScheduler'a görev ekle"""
        if not task.schedule_expr:
            return
        
        try:
            # Cron expression parse et
            if task.schedule_expr.startswith('cron:'):
                cron_expr = task.schedule_expr[5:]
                parts = cron_expr.split()
                if len(parts) == 5:
                    minute, hour, day, month, day_of_week = parts
                    trigger = CronTrigger(
                        minute=minute,
                        hour=hour,
                        day=day,
                        month=month,
                        day_of_week=day_of_week
                    )
                else:
                    raise ValueError(f"Geçersiz cron expression: {cron_expr}")
            
            # Interval expression parse et
            elif task.schedule_expr.startswith('interval:'):
                interval_expr = task.schedule_expr[9:]
                if 'seconds=' in interval_expr:
                    seconds = int(interval_expr.split('seconds=')[1].split()[0])
                    trigger = IntervalTrigger(seconds=seconds)
                elif 'minutes=' in interval_expr:
                    minutes = int(interval_expr.split('minutes=')[1].split()[0])
                    trigger = IntervalTrigger(minutes=minutes)
                elif 'hours=' in interval_expr:
                    hours = int(interval_expr.split('hours=')[1].split()[0])
                    trigger = IntervalTrigger(hours=hours)
                else:
                    raise ValueError(f"Geçersiz interval expression: {interval_expr}")
            
            # Date expression parse et
            elif task.schedule_expr.startswith('date:'):
                date_expr = task.schedule_expr[5:]
                trigger = DateTrigger(run_date=date_expr)
            
            else:
                # Cron expression varsayımı
                parts = task.schedule_expr.split()
                if len(parts) == 5:
                    minute, hour, day, month, day_of_week = parts
                    trigger = CronTrigger(
                        minute=minute,
                        hour=hour,
                        day=day,
                        month=month,
                        day_of_week=day_of_week
                    )
                else:
                    raise ValueError(f"Geçersiz schedule expression: {task.schedule_expr}")
            
            # Job ekle
            job = self.scheduler.add_job(
                func=self._execute_task_wrapper,
                trigger=trigger,
                args=[task.name],
                id=task.name,
                name=task.name,
                replace_existing=True
            )
            
            # Next run time'ı güncelle
            task.next_run = job.next_run_time
            
        except Exception as e:
            raise TaskSchedulingError(
                f"APScheduler job ekleme hatası: {e}",
                task_name=task.name,
                schedule=task.schedule_expr
            )
    
    def _add_schedule_job(self, task: Task):
        """Schedule backend'e görev ekle"""
        if not task.schedule_expr:
            return
        
        try:
            # Basit schedule expressions
            if task.schedule_expr == 'every_minute':
                self.scheduler.every().minute.do(self._execute_task_wrapper, task.name)
            elif task.schedule_expr == 'every_hour':
                self.scheduler.every().hour.do(self._execute_task_wrapper, task.name)
            elif task.schedule_expr == 'daily':
                self.scheduler.every().day.do(self._execute_task_wrapper, task.name)
            elif task.schedule_expr.startswith('every_'):
                # every_5_minutes, every_2_hours gibi
                parts = task.schedule_expr.split('_')
                if len(parts) >= 3:
                    interval = int(parts[1])
                    unit = parts[2]
                    if unit == 'minutes':
                        self.scheduler.every(interval).minutes.do(self._execute_task_wrapper, task.name)
                    elif unit == 'hours':
                        self.scheduler.every(interval).hours.do(self._execute_task_wrapper, task.name)
                    elif unit == 'days':
                        self.scheduler.every(interval).days.do(self._execute_task_wrapper, task.name)
            
        except Exception as e:
            raise TaskSchedulingError(
                f"Schedule job ekleme hatası: {e}",
                task_name=task.name,
                schedule=task.schedule_expr
            )
    
    def _execute_task_wrapper(self, task_name: str):
        """Görev çalıştırma wrapper'ı"""
        if task_name not in self.tasks:
            self.logger.error(f"Görev bulunamadı: {task_name}")
            return
        
        task = self.tasks[task_name]
        if not task.enabled:
            self.logger.debug(f"Görev devre dışı: {task_name}")
            return
        
        # Görevi kuyruğa ekle
        self.task_queue.put(task)
    
    def _worker_thread_func(self):
        """Worker thread fonksiyonu"""
        while self.is_running:
            try:
                # Görev al
                task = self.task_queue.get(timeout=1)
                
                # Görevi çalıştır
                with performance_context(self.logger.name, f"task_execution_{task.name}"):
                    result = task.execute()
                
                # Sonucu logla
                if result['status'] == 'success':
                    self.logger.info(f"Görev başarılı: {task.name} - {result['execution_time']:.2f}s")
                else:
                    self.logger.error(f"Görev başarısız: {task.name} - {result['error']}")
                
                # Kuyruk işaretini kaldır
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker thread hatası: {e}")
    
    def start(self):
        """Scheduler'ı başlat"""
        if self.is_running:
            self.logger.warning("Scheduler zaten çalışıyor")
            return
        
        try:
            # APScheduler başlat
            if self.backend == 'apscheduler':
                self.scheduler.start()
            
            # Worker thread başlat
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_thread_func, daemon=True)
            self.worker_thread.start()
            
            self.logger.info("Task Scheduler başlatıldı")
            
        except Exception as e:
            raise TaskSchedulingError(f"Scheduler başlatma hatası: {e}")
    
    def stop(self):
        """Scheduler'ı durdur"""
        if not self.is_running:
            return
        
        try:
            # Worker thread durdur
            self.is_running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=5)
            
            # APScheduler durdur
            if self.backend == 'apscheduler' and self.scheduler:
                self.scheduler.shutdown()
            
            # Executor durdur
            self.executor.shutdown(wait=True)
            
            self.logger.info("Task Scheduler durduruldu")
            
        except Exception as e:
            self.logger.error(f"Scheduler durdurma hatası: {e}")
    
    def run_task(self, task_name: str) -> Dict[str, Any]:
        """Görevi manuel olarak çalıştır"""
        if task_name not in self.tasks:
            raise TaskSchedulingError(f"Görev bulunamadı: {task_name}")
        
        task = self.tasks[task_name]
        return task.execute()
    
    def enable_task(self, task_name: str):
        """Görevi etkinleştir"""
        if task_name not in self.tasks:
            raise TaskSchedulingError(f"Görev bulunamadı: {task_name}")
        
        self.tasks[task_name].enabled = True
        self.logger.info(f"Görev etkinleştirildi: {task_name}")
    
    def disable_task(self, task_name: str):
        """Görevi devre dışı bırak"""
        if task_name not in self.tasks:
            raise TaskSchedulingError(f"Görev bulunamadı: {task_name}")
        
        self.tasks[task_name].enabled = False
        self.logger.info(f"Görev devre dışı bırakıldı: {task_name}")
    
    def remove_task(self, task_name: str):
        """Görevi kaldır"""
        if task_name not in self.tasks:
            raise TaskSchedulingError(f"Görev bulunamadı: {task_name}")
        
        # APScheduler'dan kaldır
        if self.backend == 'apscheduler' and self.scheduler:
            try:
                self.scheduler.remove_job(task_name)
            except Exception:
                pass
        
        # Görevi kaldır
        del self.tasks[task_name]
        self.logger.info(f"Görev kaldırıldı: {task_name}")
    
    def get_task_stats(self, task_name: str = None) -> Dict[str, Any]:
        """Görev istatistiklerini al"""
        if task_name:
            if task_name not in self.tasks:
                raise TaskSchedulingError(f"Görev bulunamadı: {task_name}")
            return self.tasks[task_name].get_stats()
        
        # Tüm görevlerin istatistikleri
        stats = {
            'total_tasks': len(self.tasks),
            'enabled_tasks': sum(1 for task in self.tasks.values() if task.enabled),
            'running_tasks': self.task_queue.qsize(),
            'tasks': {}
        }
        
        for name, task in self.tasks.items():
            stats['tasks'][name] = task.get_stats()
        
        return stats
    
    def save_scheduler_state(self, filepath: str):
        """Scheduler durumunu kaydet"""
        try:
            state = {
                'backend': self.backend,
                'is_running': self.is_running,
                'tasks': {}
            }
            
            for name, task in self.tasks.items():
                state['tasks'][name] = {
                    'name': task.name,
                    'schedule_expr': task.schedule_expr,
                    'enabled': task.enabled,
                    'max_retries': task.max_retries,
                    'retry_delay': task.retry_delay,
                    'timeout': task.timeout,
                    'stats': task.get_stats()
                }
            
            write_file(state, filepath, '.json')
            self.logger.info(f"Scheduler durumu kaydedildi: {filepath}")
            
        except Exception as e:
            raise TaskSchedulingError(f"Scheduler durumu kaydetme hatası: {e}")
    
    def load_scheduler_state(self, filepath: str):
        """Scheduler durumunu yükle"""
        try:
            state = read_file(filepath, '.json')
            
            # Görevleri yükle
            for name, task_data in state['tasks'].items():
                # Fonksiyon bilgisi kaydedilmediği için sadece istatistikleri yükle
                if name in self.tasks:
                    task = self.tasks[name]
                    task.enabled = task_data['enabled']
                    # Diğer istatistikler zaten mevcut
            
            self.logger.info(f"Scheduler durumu yüklendi: {filepath}")
            
        except Exception as e:
            raise TaskSchedulingError(f"Scheduler durumu yükleme hatası: {e}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Convenience functions
def create_scheduler(backend: str = 'apscheduler', config=None) -> TaskScheduler:
    """Scheduler oluştur"""
    return TaskScheduler(config=config, backend=backend)


def run_scheduled_task(func: Callable, schedule_expr: str, 
                      args: tuple = None, kwargs: dict = None,
                      **task_kwargs) -> Task:
    """Hızlı görev zamanlama"""
    scheduler = create_scheduler()
    task = scheduler.add_task(
        name=func.__name__,
        func=func,
        schedule_expr=schedule_expr,
        args=args,
        kwargs=kwargs,
        **task_kwargs
    )
    scheduler.start()
    return task


if __name__ == "__main__":
    # Test
    logger = get_logger(__name__)
    
    def test_task():
        logger.info("Test görevi çalıştırıldı")
        time.sleep(1)
        return "success"
    
    def error_task():
        logger.info("Hata görevi çalıştırıldı")
        raise Exception("Test hatası")
    
    try:
        # Scheduler oluştur
        scheduler = create_scheduler()
        
        # Görevler ekle
        scheduler.add_task(
            name="test_task",
            func=test_task,
            schedule_expr="every_10_seconds",
            max_retries=2
        )
        
        scheduler.add_task(
            name="error_task",
            func=error_task,
            schedule_expr="every_30_seconds",
            max_retries=3
        )
        
        # Scheduler'ı başlat
        scheduler.start()
        
        # 1 dakika çalıştır
        time.sleep(60)
        
        # İstatistikleri al
        stats = scheduler.get_task_stats()
        logger.info(f"Scheduler istatistikleri: {stats}")
        
        # Scheduler'ı durdur
        scheduler.stop()
        
    except Exception as e:
        logger.error(f"Test hatası: {e}") 