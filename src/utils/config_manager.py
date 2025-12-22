
import json
import os
import platform
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages application configuration and settings"""
    
    # Platform detection
    SYSTEM = platform.system()
    IS_WINDOWS = SYSTEM == "Windows"
    IS_LINUX = SYSTEM == "Linux"
    IS_MACOS = SYSTEM == "Darwin"
    
    def __init__(self, config_file: Optional[str] = None):
        if config_file is None:
            self.config_file = Path(__file__).parent.parent.parent / "config" / "settings.json"
        else:
            self.config_file = Path(config_file)
            
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config file: {e}")
                return self._get_default_config()
        else:
            config = self._get_default_config()
            self.save_config(config)
            return config
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings"""
        return {
            "video_processing": {
                "supported_formats": [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"],
                "max_resolution": "1920x1080",
                "fps_threshold": 30,
                "scene_detection_threshold": 0.3,
                "frame_extraction_interval": 1.0
            },
            "audio_transcription": {
                "model": "whisper",
                "language": "en",
                "chunk_duration": 30,
                "enable_speaker_detection": True,
                "confidence_threshold": 0.8,
                "azure_region": "eastus"
            },
            "caption_generation": {
                "font_size": 24,
                "font_family": "Arial",
                "font_family_linux": "Liberation Sans",
                "font_family_fallback": "Liberation Sans",
                "font_color": "white",
                "background_color": "black",
                "background_opacity": 0.7,
                "position": "bottom",
                "max_chars_per_line": 50,
                "max_lines": 2,
                "caption_duration": 3.0
            },
            "report_generation": {
                "format": "pdf",
                "include_screenshots": True,
                "include_timestamps": True,
                "include_summaries": True,
                "summary_model": "openai"
            },
            "youtube": {
                "download_format": "best[height<=720]",
                "audio_format": "bestaudio/best",
                "download_subtitles": True
            },
            "web_video": {
                "user_agent_windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "user_agent_linux": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "user_agent_darwin": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "timeout": 30,
                "max_retries": 3,
                "chunk_size": 8192
            },
            "output": {
                "default_output_dir": "",
                "create_subfolders": True,
                "preserve_original": True,
                "compression_quality": 23,
                "image_quality": 85
            },
            "logging": {
                "level": "INFO",
                "log_filename": "application.log",
                "max_file_size_mb": 10,
                "backup_count": 5,
                "console_output": True
            },
            "performance": {
                "max_concurrent_processes": 4,
                "memory_limit_mb": 2048,
                "temp_cleanup": True,
                "gpu_acceleration": False
            },
            "security": {
                "encrypt_credentials": True,
                "session_timeout": 3600,
                "secure_download": True
            },
            "paths": {
                "ffmpeg_path": "",
                "ffprobe_path": "",
                "temp_dir": "",
                "cache_dir": ""
            }
        }
        
    def get(self, key_path: str, default=None) -> Any:
        """Get configuration value using dot notation (e.g., 'audio_transcription.model')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_secure(self, key_path: str, env_var_name: str = None, default=None) -> Any:
        """
        Securely get configuration value, preferring environment variables over config file
        This ensures API keys are never stored in code
        """
        # First try to get from environment variable
        if env_var_name:
            env_value = os.environ.get(env_var_name)
            if env_value and env_value.strip():
                return env_value.strip()
        
        # Auto-detect common environment variable names
        auto_env_names = {
            'audio_transcription.google_api_key': 'GOOGLE_API_KEY',
            'audio_transcription.azure_api_key': 'AZURE_API_KEY', 
            'audio_transcription.azure_region': 'AZURE_REGION',
            'report_generation.openai_api_key': 'OPENAI_API_KEY',
            'youtube.youtube_api_key': 'YOUTUBE_API_KEY'
        }
        
        if key_path in auto_env_names:
            env_value = os.environ.get(auto_env_names[key_path])
            if env_value and env_value.strip():
                return env_value.strip()
        
        # Fall back to config file (for non-sensitive values)
        config_value = self.get(key_path, default)
        
        # Never return placeholder values
        if isinstance(config_value, str) and config_value.startswith("YOUR "):
            return default
            
        return config_value
            
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config_dict = self.config
        
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]
            
        config_dict[keys[-1]] = value
        
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file"""
        if config is not None:
            self.config = config
            
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving config file: {e}")
            
    def update_api_key(self, service: str, api_key: str) -> None:
        """Update API key for a specific service"""
        key_mapping = {
            "google": "audio_transcription.google_api_key",
            "azure": "audio_transcription.azure_api_key",
            "openai": "report_generation.openai_api_key",
            "youtube": "youtube.youtube_api_key"
        }
        
        if service in key_mapping:
            self.set(key_mapping[service], api_key)
            self.save_config()
        else:
            raise ValueError(f"Unknown service: {service}")
            
    def validate_config(self) -> Dict[str, list]:
        """Validate configuration and return any issues"""
        issues = {
            "missing": [],
            "invalid": [],
            "warnings": []
        }
        
        api_keys = [
            ("audio_transcription.google_api_key", "Google Speech-to-Text"),
            ("audio_transcription.azure_api_key", "Azure Speech Services"),
            ("report_generation.openai_api_key", "OpenAI GPT"),
            ("youtube.youtube_api_key", "YouTube API (optional)")
        ]
        
        for key_path, service_name in api_keys:
            value = self.get(key_path)
            if value and "YOUR_" in str(value) and "_KEY_HERE" in str(value):
                if "optional" not in service_name.lower():
                    issues["warnings"].append(f"API key not configured for {service_name}")
                    
        supported_formats = self.get("video_processing.supported_formats", [])
        if not supported_formats:
            issues["missing"].append("No supported video formats configured")
                
        return issues
    
    def get_user_agent(self) -> str:
        """Get platform-specific user agent"""
        if self.IS_WINDOWS:
            return self.get("web_video.user_agent_windows", 
                          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        elif self.IS_MACOS:
            return self.get("web_video.user_agent_darwin",
                          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        else:
            return self.get("web_video.user_agent_linux",
                          "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
    
    def get_font_family(self) -> str:
        """Get platform-specific font family"""
        if self.IS_WINDOWS:
            return self.get("caption_generation.font_family", "Arial")
        elif self.IS_LINUX:
            return self.get("caption_generation.font_family_linux", "Liberation Sans")
        else:
            return self.get("caption_generation.font_family", "Arial")
        
    def get_temp_dir(self) -> Path:
        """Get temporary directory for processing"""
        custom_temp = self.get("paths.temp_dir")
        if custom_temp:
            temp_dir = Path(custom_temp)
        else:
            temp_dir = Path.home() / ".meeting_captioning" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
        
    def get_output_dir(self) -> Path:
        """Get default output directory (cross-platform)"""
        custom_output = self.get("output.default_output_dir")
        if custom_output:
            output_dir = Path(custom_output)
        else:
            # Try common locations in order
            possible_dirs = [
                Path.home() / "Videos" / "MeetingCaptioning",
                Path.home() / "Documents" / "MeetingCaptioning",
                Path.home() / "MeetingCaptioning"
            ]
            
            # On Windows, also try Desktop
            if self.IS_WINDOWS:
                possible_dirs.insert(0, Path.home() / "Desktop" / "video_output")
            
            # Use first valid location or fallback
            output_dir = None
            for dir_path in possible_dirs:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    output_dir = dir_path
                    break
                except (OSError, PermissionError):
                    continue
                    
            if output_dir is None:
                output_dir = Path.home() / "MeetingCaptioning"
                output_dir.mkdir(parents=True, exist_ok=True)
                
        return output_dir
        
    def get_logs_dir(self) -> Path:
        """Get logs directory"""
        logs_dir = Path.home() / ".meeting_captioning" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir