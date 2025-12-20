import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """Enhanced logging class with file rotation and structured output"""
    
    def __init__(self, name: str = "MeetingCaptioning", config_manager=None):
        self.name = name
        self.config_manager = config_manager
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            self._setup_logging()
            
    def _setup_logging(self):
        """Configure logging with file and console handlers"""
        if self.config_manager:
            log_level = self.config_manager.get("logging.level", "INFO")
            log_filename = self.config_manager.get("logging.log_filename", "application.log")
            max_file_size_mb = self.config_manager.get("logging.max_file_size_mb", 10)
            backup_count = self.config_manager.get("logging.backup_count", 5)
            console_output = self.config_manager.get("logging.console_output", True)
            logs_dir = self.config_manager.get_logs_dir()
        else:
            log_level = "INFO"
            log_filename = "application.log"
            max_file_size_mb = 10
            backup_count = 5
            console_output = True
            logs_dir = Path.home() / ".meeting_captioning" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        log_file_path = logs_dir / log_filename
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)
        
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(simple_formatter)
            console_handler.setLevel(level)
            self.logger.addHandler(console_handler)
            
        self.logger.info(f"Logging initialized - Level: {log_level}, File: {log_file_path}")
        
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)
        
    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, **kwargs)
        
    def log_processing_start(self, video_path: str, input_type: str):
        self.info(f"Starting video processing - Type: {input_type}, Path: {video_path}")
        
    def log_processing_step(self, step: str, progress: float = None):
        message = f"Processing step: {step}"
        if progress is not None:
            message += f" ({progress:.1f}%)"
        self.info(message)
        
    def log_processing_complete(self, output_files: list, processing_time: float):
        self.info(f"Processing completed in {processing_time:.2f}s")
        for file_path in output_files:
            self.info(f"Generated: {file_path}")
            
    def log_error_with_context(self, error: Exception, context: dict):
        error_msg = f"Error: {str(error)}"
        for key, value in context.items():
            error_msg += f"\n  {key}: {value}"
        self.error(error_msg, exc_info=True)
        
    def log_performance_metrics(self, metrics: dict):
        self.info("Performance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {metric}: {value:.3f}")
            else:
                self.info(f"  {metric}: {value}")
                
    def create_session_logger(self, session_id: str) -> 'SessionLogger':
        return SessionLogger(self, session_id)


class SessionLogger:
    """Session-specific logger for tracking individual processing sessions"""
    
    def __init__(self, base_logger: Logger, session_id: str):
        self.base_logger = base_logger
        self.session_id = session_id
        self.start_time = datetime.now()
        self.steps = []
        
        self.info(f"Session started - ID: {session_id}")
        
    def _log_with_session(self, level: str, message: str, **kwargs):
        prefixed_message = f"[{self.session_id}] {message}"
        getattr(self.base_logger, level)(prefixed_message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        self._log_with_session("debug", message, **kwargs)
        
    def info(self, message: str, **kwargs):
        self._log_with_session("info", message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self._log_with_session("warning", message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self._log_with_session("error", message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        self._log_with_session("critical", message, **kwargs)
        
    def log_step(self, step_name: str, status: str = "started"):
        timestamp = datetime.now()
        step_info = {
            "name": step_name,
            "status": status,
            "timestamp": timestamp
        }
        
        if status == "completed" and self.steps:
            for step in reversed(self.steps):
                if step["name"] == step_name and step["status"] == "started":
                    duration = (timestamp - step["timestamp"]).total_seconds()
                    step_info["duration"] = duration
                    self.info(f"Step '{step_name}' completed in {duration:.2f}s")
                    break
        
        self.steps.append(step_info)
        
        if status == "started":
            self.info(f"Step '{step_name}' started")
        elif status == "failed":
            self.error(f"Step '{step_name}' failed")
            
    def get_session_summary(self) -> dict:
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        completed_steps = [s for s in self.steps if s["status"] == "completed"]
        failed_steps = [s for s in self.steps if s["status"] == "failed"]
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "total_duration": total_duration,
            "total_steps": len(self.steps),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "steps": self.steps
        }
        
    def finalize_session(self, success: bool = True):
        summary = self.get_session_summary()
        
        if success:
            self.info(f"Session completed successfully in {summary['total_duration']:.2f}s")
            self.info(f"Steps completed: {summary['completed_steps']}/{summary['total_steps']}")
        else:
            self.error(f"Session failed after {summary['total_duration']:.2f}s")
            self.error(f"Failed steps: {summary['failed_steps']}")
            
        return summary