import os
import re
import mimetypes
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Optional, List, Tuple, Dict, Any


class InputValidator:
    """Validates various types of inputs for the application"""
    
    def __init__(self):
        self.supported_video_formats = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.webm': 'video/webm',
            '.m4v': 'video/x-m4v',
            '.mpg': 'video/mpeg',
            '.mpeg': 'video/mpeg',
            '.3gp': 'video/3gpp'
        }
        
        self.youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})'
        ]
        
        self.video_hosting_domains = {
            'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
            'twitch.tv', 'facebook.com', 'instagram.com', 'twitter.com',
            'tiktok.com', 'streamable.com', 'wistia.com', 'brightcove.com'
        }
        
    def validate_local_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate local video file
        Returns: (is_valid, error_message)
        """
        if not file_path or not file_path.strip():
            return False, "File path is empty"
            
        path = Path(file_path)
        
        if not path.exists():
            return False, f"File does not exist: {file_path}"
            
        if not path.is_file():
            return False, f"Path is not a file: {file_path}"
            
        file_extension = path.suffix.lower()
        if file_extension not in self.supported_video_formats:
            supported = ', '.join(self.supported_video_formats.keys())
            return False, f"Unsupported file format: {file_extension}. Supported: {supported}"
            
        try:
            file_size = path.stat().st_size
            if file_size > 5 * 1024 * 1024 * 1024:  # 5GB
                return True, f"Warning: File is very large ({file_size / (1024**3):.1f}GB). Processing may be slow."
            elif file_size == 0:
                return False, "File is empty"
        except OSError as e:
            return False, f"Error accessing file: {e}"
            
        if not os.access(path, os.R_OK):
            return False, "File is not readable (permission denied)"
            
        return True, None
        
    def validate_youtube_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate YouTube URL
        Returns: (is_valid, error_message)
        """
        if not url or not url.strip():
            return False, "URL is empty"
            
        url = url.strip()
        
        video_id = self.extract_youtube_video_id(url)
        if not video_id:
            return False, "Invalid YouTube URL format"
            
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            return False, "Invalid YouTube video ID format"
            
        return True, None
        
    def validate_web_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate web URL for video content
        Returns: (is_valid, error_message)
        """
        if not url or not url.strip():
            return False, "URL is empty"
            
        url = url.strip()
        
        try:
            parsed = urlparse(url)
        except Exception as e:
            return False, f"Invalid URL format: {e}"
            
        if parsed.scheme not in ['http', 'https']:
            return False, "URL must use HTTP or HTTPS protocol"
            
        if not parsed.netloc:
            return False, "Invalid domain in URL"
            
        path_lower = parsed.path.lower()
        video_extensions = list(self.supported_video_formats.keys())
        
        has_video_extension = any(path_lower.endswith(ext) for ext in video_extensions)
        is_known_platform = any(domain in parsed.netloc.lower() for domain in self.video_hosting_domains)
        
        if not has_video_extension and not is_known_platform:
            return True, "Warning: URL may not contain video content"
            
        return True, None
        
    def extract_youtube_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        for pattern in self.youtube_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
        
    def validate_output_directory(self, dir_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate output directory
        Returns: (is_valid, error_message)
        """
        if not dir_path or not dir_path.strip():
            return False, "Output directory path is empty"
            
        path = Path(dir_path)
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            test_file = path / ".write_test.tmp"
            try:
                test_file.touch()
                test_file.unlink(missing_ok=True)
            except OSError:
                return False, "Directory is not writable (permission denied)"
                
        except OSError as e:
            return False, f"Cannot create or access directory: {e}"
            
        return True, None
        
    def validate_processing_options(self, options: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate processing options
        Returns: (is_valid, list_of_warnings)
        """
        warnings = []
        
        burn_captions = options.get('burn_captions', False)
        generate_report = options.get('generate_report', False)
        
        if not burn_captions and not generate_report:
            return False, ["At least one output option (captions or report) must be selected"]
            
        if burn_captions:
            font_size = options.get('font_size', 24)
            if not isinstance(font_size, (int, float)) or font_size < 8 or font_size > 72:
                warnings.append("Font size should be between 8 and 72 pixels")
                
            max_chars = options.get('max_chars_per_line', 50)
            if not isinstance(max_chars, int) or max_chars < 20 or max_chars > 100:
                warnings.append("Characters per line should be between 20 and 100")
                
        if generate_report:
            report_format = options.get('report_format', 'pdf')
            if report_format not in ['pdf', 'docx', 'txt']:
                warnings.append("Invalid report format. Supported: pdf, docx, txt")
                
        return True, warnings
        
    def validate_video_file_integrity(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Perform basic integrity check on video file
        Returns: (is_valid, error_message)
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False, "Video file is corrupted or unreadable"
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            if frame_count <= 0:
                return False, "Video file contains no frames"
            if fps <= 0:
                return False, "Video file has invalid frame rate"
            if width <= 0 or height <= 0:
                return False, "Video file has invalid dimensions"
                
            duration = frame_count / fps if fps > 0 else 0
            if duration > 7200:  # 2 hours
                return True, f"Warning: Video is very long ({duration/3600:.1f} hours). Processing may take significant time."
                
        except ImportError:
            return True, "Warning: Could not verify video integrity (OpenCV not available)"
        except Exception as e:
            return False, f"Error checking video integrity: {e}"
            
        return True, None
        
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage"""
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, '_', filename)
        
        sanitized = sanitized.strip(' .')
        
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:200-len(ext)] + ext
            
        if not sanitized:
            sanitized = "untitled"
            
        return sanitized
        
    def validate_api_credentials(self, service: str, credentials: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """
        Validate API credentials for external services
        Returns: (is_valid, error_message)
        """
        if service == "openai":
            api_key = credentials.get("api_key", "")
            if not api_key or "YOUR_" in api_key:
                return False, "OpenAI API key is not configured"
            if not api_key.startswith("sk-"):
                return False, "Invalid OpenAI API key format"
                
        elif service == "google":
            api_key = credentials.get("api_key", "")
            if not api_key or "YOUR_" in api_key:
                return False, "Google API key is not configured"
                
        elif service == "azure":
            api_key = credentials.get("api_key", "")
            region = credentials.get("region", "")
            if not api_key or "YOUR_" in api_key:
                return False, "Azure API key is not configured"
            if not region:
                return False, "Azure region is not configured"
                
        else:
            return False, f"Unknown service: {service}"
            
        return True, None