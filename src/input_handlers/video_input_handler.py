import os
import re
import time
import platform
import subprocess
import json
import shutil
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse, parse_qs


class VideoInputHandler:
    """Handles video input from multiple sources"""
    
    SYSTEM = platform.system()
    IS_WINDOWS = SYSTEM == "Windows"
    IS_LINUX = SYSTEM == "Linux"
    IS_MACOS = SYSTEM == "Darwin"
    
    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger
        
        # Configuration
        self.download_format = config_manager.get("youtube.download_format", "best[height<=720]")
        self.audio_format = config_manager.get("youtube.audio_format", "bestaudio/best")
        self.download_subtitles = config_manager.get("youtube.download_subtitles", True)
        self.youtube_api_key = config_manager.get_secure("youtube.youtube_api_key")
        
        # Web video settings
        self.user_agent = config_manager.get_user_agent() if hasattr(config_manager, 'get_user_agent') else self._get_default_user_agent()
        self.timeout = config_manager.get("web_video.timeout", 30)
        self.max_retries = config_manager.get("web_video.max_retries", 3)
        self.chunk_size = config_manager.get("web_video.chunk_size", 8192)
    
    def _get_default_user_agent(self) -> str:
        """Get default user agent based on platform"""
        if self.IS_WINDOWS:
            return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        elif self.IS_MACOS:
            return "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        else:
            return "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        
    def download_youtube_video(self, url: str, output_dir: str) -> str:
        """Download YouTube video using yt-dlp"""
        self.logger.info(f"Downloading YouTube video: {url}")
        
        try:
            output_path = Path(output_dir) / "downloads"
            output_path.mkdir(parents=True, exist_ok=True)
            
            video_info = self._get_youtube_info(url)
            self.logger.info(f"Video title: {video_info.get('title', 'Unknown')}")
            
            safe_title = self._sanitize_filename(video_info.get('title', 'youtube_video'))
            output_template = str(output_path / f"{safe_title}.%(ext)s")
            
            download_path = self._download_with_ytdlp(url, output_template)
            
            if not Path(download_path).exists():
                raise Exception("Download completed but file not found")
                
            self.logger.info(f"YouTube video downloaded: {download_path}")
            return download_path
            
        except Exception as e:
            self.logger.error(f"YouTube download failed: {str(e)}")
            raise Exception(f"Failed to download YouTube video: {str(e)}")
            
    def download_web_video(self, url: str, output_dir: str) -> str:
        """Download video from web URL"""
        self.logger.info(f"Downloading web video: {url}")
        
        try:
            output_path = Path(output_dir) / "downloads"
            output_path.mkdir(parents=True, exist_ok=True)
            
            parsed_url = urlparse(url)
            original_filename = Path(parsed_url.path).name
            
            if not original_filename or '.' not in original_filename:
                original_filename = f"web_video_{int(time.time())}.mp4"
                
            safe_filename = self._sanitize_filename(original_filename)
            download_path = output_path / safe_filename
            
            if self._is_direct_video_url(url):
                self._download_direct_video(url, str(download_path))
            else:
                download_path = self._download_with_ytdlp(url, str(download_path))
                
            if not Path(download_path).exists():
                raise Exception("Download completed but file not found")
                
            self.logger.info(f"Web video downloaded: {download_path}")
            return str(download_path)
            
        except Exception as e:
            self.logger.error(f"Web video download failed: {str(e)}")
            raise Exception(f"Failed to download web video: {str(e)}")
            
    def download_from_cloud_storage(self, url: str, output_dir: str, 
                                  credentials: Optional[Dict[str, str]] = None) -> str:
        """Download video from cloud storage (Google Drive, Dropbox, etc.)"""
        self.logger.info(f"Downloading from cloud storage: {url}")
        
        try:
            if "drive.google.com" in url:
                return self._download_from_google_drive(url, output_dir, credentials)
            elif "dropbox.com" in url:
                return self._download_from_dropbox(url, output_dir, credentials)
            elif "onedrive" in url or "sharepoint.com" in url:
                return self._download_from_onedrive(url, output_dir, credentials)
            else:
                return self._download_generic_cloud_file(url, output_dir, credentials)
                
        except Exception as e:
            self.logger.error(f"Cloud storage download failed: {str(e)}")
            raise Exception(f"Failed to download from cloud storage: {str(e)}")

    def _get_youtube_info(self, url: str) -> Dict[str, Any]:
        """Get YouTube video information using yt-dlp"""
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise Exception(f"yt-dlp error: {result.stderr}")
                
            video_info = json.loads(result.stdout)
            
            return {
                'title': video_info.get('title', 'Unknown'),
                'duration': video_info.get('duration', 0),
                'uploader': video_info.get('uploader', 'Unknown'),
                'upload_date': video_info.get('upload_date', ''),
                'view_count': video_info.get('view_count', 0),
                'description': video_info.get('description', ''),
                'formats': len(video_info.get('formats', [])),
                'id': video_info.get('id', ''),
                'webpage_url': video_info.get('webpage_url', url)
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("YouTube info request timed out")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse YouTube info: {e}")
        except FileNotFoundError:
            raise Exception("yt-dlp not found. Please install yt-dlp: pip install yt-dlp")
        except Exception as e:
            self.logger.warning(f"Could not get YouTube info: {e}")
            return {'title': 'Unknown YouTube Video'}

    # ✅ ONLY ONE _download_with_ytdlp METHOD - WITH FIREFOX FIX
    def _download_with_ytdlp(self, url: str, output_template: str) -> str:
        """Download video using yt-dlp"""
        try:
            cmd = [
                'yt-dlp',
                '--format', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                '--output', output_template,
                '--no-playlist',
                '--merge-output-format', 'mp4',
            ]
            
            # Cookie handling for YouTube
            if "youtube.com" in url or "youtu.be" in url:
                cookie_file = self.config_manager.get("youtube.cookie_file", None)
                cookie_browser = self.config_manager.get("youtube.cookie_browser", None)
                
                if cookie_file and Path(cookie_file).exists():
                    # Use cookie file if provided
                    cmd.extend(['--cookies', cookie_file])
                    self.logger.info(f"Using cookie file: {cookie_file}")
                elif cookie_browser:
                    # Use specified browser from config
                    cmd.extend(['--cookies-from-browser', cookie_browser])
                    self.logger.info(f"Using cookies from: {cookie_browser}")
                else:
                    # Default to Firefox (works on most systems)
                    cmd.extend(['--cookies-from-browser', 'firefox'])
                    self.logger.info("Using cookies from: firefox (default)")
            
            if self.download_subtitles:
                cmd.extend(['--write-auto-subs', '--sub-langs', 'en,en-US'])
            
            cmd.append(url)
                
            self.logger.info(f"Running yt-dlp: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise Exception(f"yt-dlp error: {result.stderr}")
                
            output_dir = Path(output_template).parent
            pattern = Path(output_template).stem.replace('.%(ext)s', '').replace('%(ext)s', '')
            
            downloaded_files = list(output_dir.glob(f"{pattern}*"))
            video_files = [f for f in downloaded_files 
                        if f.suffix.lower() in ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.m4a']]
            
            if not video_files:
                raise Exception("No video file found after download")
                
            return str(video_files[0])
            
        except subprocess.TimeoutExpired:
            raise Exception("Download timed out")
        except FileNotFoundError:
            raise Exception("yt-dlp not found. Please install yt-dlp: pip install yt-dlp")
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")
            
    def _is_direct_video_url(self, url: str) -> bool:
        """Check if URL is a direct link to a video file"""
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        
        return any(path_lower.endswith(ext) for ext in video_extensions)
        
    def _download_direct_video(self, url: str, output_path: str):
        """Download video file directly via HTTP"""
        headers = {
            'User-Agent': self.user_agent,
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Download attempt {attempt + 1}/{self.max_retries}")
                
                response = session.head(url, timeout=self.timeout)
                response.raise_for_status()
                
                file_size = int(response.headers.get('content-length', 0))
                self.logger.info(f"File size: {file_size / (1024*1024):.1f} MB")
                
                response = session.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                downloaded_size = 0
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if file_size > 0 and downloaded_size % (1024 * 1024) == 0:
                                progress = (downloaded_size / file_size) * 100
                                self.logger.info(f"Download progress: {progress:.1f}%")
                                
                self.logger.info(f"Direct download completed: {output_path}")
                return
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Direct download failed after {self.max_retries} attempts: {e}")
                time.sleep(2)
                
    def _download_from_google_drive(self, url: str, output_dir: str, 
                                  credentials: Optional[Dict[str, str]] = None) -> str:
        """Download video from Google Drive"""
        try:
            file_id = self._extract_google_drive_file_id(url)
            if not file_id:
                raise Exception("Could not extract Google Drive file ID")
                
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            output_path = Path(output_dir) / "downloads"
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"gdrive_video_{file_id}.mp4"
            file_path = output_path / filename
            
            self._download_direct_video(download_url, str(file_path))
            
            return str(file_path)
            
        except Exception as e:
            try:
                output_path = Path(output_dir) / "downloads"
                return self._download_with_ytdlp(url, str(output_path / "gdrive_video.%(ext)s"))
            except Exception:
                raise Exception(f"Google Drive download failed: {str(e)}")
                
    def _download_from_dropbox(self, url: str, output_dir: str, 
                             credentials: Optional[Dict[str, str]] = None) -> str:
        """Download video from Dropbox"""
        try:
            if "dropbox.com" in url and "?dl=0" in url:
                direct_url = url.replace("?dl=0", "?dl=1")
            else:
                direct_url = url
                
            output_path = Path(output_dir) / "downloads"
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"dropbox_video_{int(time.time())}.mp4"
            file_path = output_path / filename
            
            self._download_direct_video(direct_url, str(file_path))
            
            return str(file_path)
            
        except Exception as e:
            raise Exception(f"Dropbox download failed: {str(e)}")
            
    def _download_from_onedrive(self, url: str, output_dir: str, 
                              credentials: Optional[Dict[str, str]] = None) -> str:
        """Download video from OneDrive/SharePoint"""
        try:
            output_path = Path(output_dir) / "downloads"
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"onedrive_video_{int(time.time())}.mp4"
            file_path = output_path / filename
            
            self._download_direct_video(url, str(file_path))
            
            return str(file_path)
            
        except Exception as e:
            raise Exception(f"OneDrive download failed: {str(e)}")
            
    def _download_generic_cloud_file(self, url: str, output_dir: str, 
                                   credentials: Optional[Dict[str, str]] = None) -> str:
        """Generic cloud file download"""
        try:
            output_path = Path(output_dir) / "downloads"
            output_path.mkdir(parents=True, exist_ok=True)
            
            parsed = urlparse(url)
            filename = Path(parsed.path).name
            if not filename:
                filename = f"cloud_video_{int(time.time())}.mp4"
                
            filename = self._sanitize_filename(filename)
            file_path = output_path / filename
            
            self._download_direct_video(url, str(file_path))
            
            return str(file_path)
            
        except Exception as e:
            raise Exception(f"Generic cloud download failed: {str(e)}")
            
    def _extract_google_drive_file_id(self, url: str) -> Optional[str]:
        """Extract file ID from Google Drive URL"""
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/open\?id=([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
                
        return None
        
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage"""
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, '_', filename)
        sanitized = sanitized.strip(' .')
        
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:200-len(ext)] + ext
            
        if not sanitized:
            sanitized = f"video_{int(time.time())}.mp4"
            
        return sanitized
        
    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get comprehensive metadata for any video file"""
        try:
            import cv2
            
            file_path = Path(video_path)
            metadata = {
                "file_info": {
                    "name": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "size_mb": file_path.stat().st_size / (1024 * 1024),
                    "created": file_path.stat().st_ctime,
                    "modified": file_path.stat().st_mtime,
                    "extension": file_path.suffix.lower()
                }
            }
            
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                metadata["video_info"] = {
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "duration_seconds": 0,
                    "codec": "unknown"
                }
                
                if metadata["video_info"]["fps"] > 0:
                    duration = metadata["video_info"]["frame_count"] / metadata["video_info"]["fps"]
                    metadata["video_info"]["duration_seconds"] = duration
                    metadata["video_info"]["duration_formatted"] = self._format_duration(duration)
                    
                cap.release()
            else:
                metadata["video_info"] = {"error": "Could not read video properties"}
                
            try:
                ffprobe_info = self._get_ffprobe_metadata(video_path)
                metadata["detailed_info"] = ffprobe_info
            except Exception as e:
                self.logger.warning(f"Could not get detailed metadata: {e}")
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error getting video metadata: {e}")
            return {"error": str(e)}
            
    def _get_ffprobe_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get detailed video metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"ffprobe failed: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            return {"error": "ffprobe timed out"}
        except FileNotFoundError:
            return {"error": "ffprobe not found"}
        except json.JSONDecodeError as e:
            return {"error": f"Could not parse ffprobe output: {e}"}
        except Exception as e:
            return {"error": str(e)}
            
    def _format_duration(self, secs: float) -> str:
        """Format duration in human readable format"""
        hours = int(secs // 3600)
        minutes = int((secs % 3600) // 60)
        seconds = int(secs % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
            
    def verify_download(self, file_path: str) -> bool:
        """Verify that downloaded file is a valid video"""
        try:
            import cv2
            
            if not Path(file_path).exists():
                return False
                
            file_size = Path(file_path).stat().st_size
            if file_size < 1024:
                return False
                
            cap = cv2.VideoCapture(file_path)
            is_valid = cap.isOpened()
            
            if is_valid:
                ret, frame = cap.read()
                is_valid = ret and frame is not None
                
            cap.release()
            return is_valid
            
        except Exception as e:
            self.logger.warning(f"Could not verify download: {e}")
            return False
            
    def cleanup_downloads(self, keep_recent: int = 5):
        """Clean up old downloaded files"""
        try:
            downloads_dir = self.config_manager.get_temp_dir() / "downloads"
            if not downloads_dir.exists():
                return
                
            video_files = []
            for ext in ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']:
                video_files.extend(downloads_dir.glob(f"*{ext}"))
                
            video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            files_to_delete = video_files[keep_recent:]
            
            for file_path in files_to_delete:
                try:
                    file_path.unlink(missing_ok=True)
                    self.logger.info(f"Cleaned up old download: {file_path.name}")
                except Exception as e:
                    self.logger.warning(f"Could not delete {file_path}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Download cleanup failed: {e}")