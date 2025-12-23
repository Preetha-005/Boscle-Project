"""
Video format converter utility for the Meeting Video Captioning Program
Handles conversion of various video formats (avi, webm, mov, mkv, etc.) to MP4
for reliable processing across all pipeline stages.
Cross-platform compatible (Windows/Linux/macOS)
"""

import os
import time
import subprocess
import platform
from pathlib import Path
from typing import Dict, Optional, Tuple


class VideoConverter:
    """Handles video format conversion for processing compatibility"""
    
    SYSTEM = platform.system()
    IS_WINDOWS = SYSTEM == "Windows"
    
    # Formats that might need conversion for reliable processing
    # MP4 with H.264 is the most universally compatible format
    FORMATS_NEEDING_CONVERSION = {'.avi', '.webm', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.3gp'}
    SAFE_FORMATS = {'.mp4'}  # MP4 with standard codecs usually works fine
    
    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger
        
    def needs_conversion(self, video_path: str) -> bool:
        """
        Check if a video file needs conversion for reliable processing.
        Some formats may work but are converted for consistency.
        """
        path = Path(video_path)
        extension = path.suffix.lower()
        
        # Check if it's a format that typically needs conversion
        if extension in self.FORMATS_NEEDING_CONVERSION:
            return True
            
        # Even for MP4, check if it has problematic codecs
        if extension in self.SAFE_FORMATS:
            codec_info = self._get_video_codec(video_path)
            # Some MP4 files might have non-standard codecs
            problematic_codecs = ['hevc', 'h265', 'av1', 'vp8', 'vp9']
            if codec_info and any(codec.lower() in codec_info.lower() for codec in problematic_codecs):
                self.logger.info(f"MP4 with codec '{codec_info}' may benefit from conversion")
                return True
                
        return False
        
    def _get_video_codec(self, video_path: str) -> Optional[str]:
        """Get the video codec using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return result.stdout.strip()
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not detect codec: {e}")
            return None
            
    def convert_to_mp4(self, input_path: str, output_dir: Optional[str] = None) -> str:
        """
        Convert video to MP4 format with H.264 codec for maximum compatibility.
        
        Args:
            input_path: Path to the input video file
            output_dir: Optional output directory. If None, uses temp directory.
            
        Returns:
            Path to the converted MP4 file
        """
        input_file = Path(input_path)
        
        # If already a compatible MP4, return as-is unless codec check fails
        if input_file.suffix.lower() == '.mp4' and not self.needs_conversion(input_path):
            self.logger.info(f"File {input_file.name} is already in compatible MP4 format")
            return input_path
            
        self.logger.info(f"Converting {input_file.name} to MP4 format for reliable processing...")
        
        # Determine output path
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.config_manager.get_temp_dir()
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create output filename
        output_filename = f"{input_file.stem}_converted.mp4"
        output_file = output_path / output_filename
        
        try:
            # Use FFmpeg for conversion with optimized settings
            converted_path = self._ffmpeg_convert(input_path, str(output_file))
            
            self.logger.info(f"Successfully converted to: {converted_path}")
            return converted_path
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            # Return original file - let processing continue and potentially fail later
            # with a more specific error message
            self.logger.warning("Continuing with original file - processing may fail")
            return input_path
            
    def _ffmpeg_convert(self, input_path: str, output_path: str) -> str:
        """
        Convert video using FFmpeg with settings optimized for processing compatibility.
        """
        try:
            # Build FFmpeg command for maximum compatibility
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',           # H.264 video codec (most compatible)
                '-preset', 'fast',            # Good balance of speed and compression
                '-crf', '23',                 # Constant rate factor (quality)
                '-c:a', 'aac',                # AAC audio codec
                '-b:a', '192k',               # Audio bitrate
                '-ar', '44100',               # Audio sample rate
                '-ac', '2',                   # Stereo audio
                '-movflags', '+faststart',    # Enable fast start for streaming
                '-pix_fmt', 'yuv420p',        # Pixel format for compatibility
                '-max_muxing_queue_size', '1024',  # Prevent queue issues
                '-y',                         # Overwrite output
                output_path
            ]
            
            self.logger.info(f"Running FFmpeg conversion: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            stderr_output = []
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    stderr_output.append(output.strip())
                    # Log progress
                    if "time=" in output:
                        self._log_conversion_progress(output)
                        
            process.wait()
            
            if process.returncode != 0:
                error_msg = '\n'.join(stderr_output[-10:])
                raise Exception(f"FFmpeg conversion failed with return code {process.returncode}:\n{error_msg}")
                
            if not Path(output_path).exists():
                raise Exception("Conversion completed but output file not created")
                
            # Verify the converted file is valid
            if not self._verify_converted_file(output_path):
                raise Exception("Converted file appears to be invalid")
                
            return output_path
            
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in PATH")
        except subprocess.TimeoutExpired:
            raise Exception("Video conversion timed out")
        except Exception as e:
            raise Exception(f"FFmpeg conversion error: {str(e)}")
            
    def _log_conversion_progress(self, output: str):
        """Parse and log FFmpeg conversion progress"""
        try:
            if "time=" in output:
                time_part = output.split("time=")[1].split()[0]
                self.logger.debug(f"Conversion progress: {time_part}")
        except (IndexError, ValueError):
            pass
            
    def _verify_converted_file(self, file_path: str) -> bool:
        """Verify that the converted file is a valid video"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False
                
            # Try to read a frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return False
                
            # Check file size is reasonable
            file_size = Path(file_path).stat().st_size
            if file_size < 1024:  # Less than 1KB is suspicious
                return False
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not verify converted file: {e}")
            return True  # Give benefit of doubt
            
    def get_conversion_info(self, video_path: str) -> Dict:
        """
        Get information about whether/why conversion is needed.
        Useful for logging and debugging.
        """
        path = Path(video_path)
        extension = path.suffix.lower()
        
        info = {
            "file_name": path.name,
            "extension": extension,
            "needs_conversion": self.needs_conversion(video_path),
            "reason": None
        }
        
        if extension in self.FORMATS_NEEDING_CONVERSION:
            info["reason"] = f"Format '{extension}' is converted to MP4 for reliability"
        elif extension in self.SAFE_FORMATS:
            codec = self._get_video_codec(video_path)
            if codec and self.needs_conversion(video_path):
                info["reason"] = f"MP4 with codec '{codec}' converted for compatibility"
            else:
                info["reason"] = "Compatible MP4 format - no conversion needed"
        else:
            info["reason"] = f"Unknown format '{extension}'"
            
        return info


def ensure_compatible_format(video_path: str, config_manager, logger, output_dir: Optional[str] = None) -> str:
    """
    Convenience function to ensure a video is in a compatible format.
    
    Args:
        video_path: Path to the video file
        config_manager: Configuration manager instance
        logger: Logger instance
        output_dir: Optional output directory for converted files
        
    Returns:
        Path to the compatible video file (may be original or converted)
    """
    converter = VideoConverter(config_manager, logger)
    
    # Log conversion info
    info = converter.get_conversion_info(video_path)
    logger.info(f"Format check for {info['file_name']}: {info['reason']}")
    
    if info['needs_conversion']:
        return converter.convert_to_mp4(video_path, output_dir)
    else:
        return video_path
