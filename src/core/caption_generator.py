"""
Caption generation module for the Meeting Video Captioning Program
Handles burning captions into video with proper timing and formatting
Cross-platform compatible (Windows/Linux/macOS)
"""

import os
import time
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import math
import textwrap


class CaptionGenerator:
    """Generates and burns captions into video files"""
    
    SYSTEM = platform.system()
    IS_WINDOWS = SYSTEM == "Windows"
    
    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger
        
        # Caption configuration - YouTube-style
        self.font_size = config_manager.get("caption_generation.font_size", 18)
        
        # Get platform-specific font
        self.font_family = config_manager.get_font_family() if hasattr(config_manager, 'get_font_family') else self._get_default_font()
        
        self.font_color = config_manager.get("caption_generation.font_color", "white")
        self.background_color = config_manager.get("caption_generation.background_color", "black")
        self.background_opacity = config_manager.get("caption_generation.background_opacity", 0.2)  #
        self.position = config_manager.get("caption_generation.position", "bottom")
        self.max_chars_per_line = config_manager.get("caption_generation.max_chars_per_line", 42)
        self.max_lines = config_manager.get("caption_generation.max_lines", 2)
        self.caption_duration = config_manager.get("caption_generation.caption_duration", 3.0)
        
        # Output configuration - Higher quality
        self.compression_quality = config_manager.get("output.compression_quality", 18)
    
    def _get_default_font(self) -> str:
        """Get default font based on platform"""
        if self.IS_WINDOWS:
            return "Arial"
        elif self.SYSTEM == "Darwin":  # macOS
            return "Helvetica"
        else:  # Linux - Use Liberation Sans as fallback (widely available)
            return "Liberation Sans"
        
    def generate_captioned_video(self, video_path: str, transcript: Dict[str, Any], 
                               output_dir: str) -> str:
        """
        Generate video with burned-in captions
        Returns path to the captioned video
        """
        start_time = time.time()
        self.logger.info("Starting caption generation")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            input_name = Path(video_path).stem
            output_filename = f"{input_name}_captioned.mp4"
            output_video_path = output_path / output_filename
            
            self.logger.log_processing_step("Processing transcript segments")
            caption_segments = self._process_transcript_segments(transcript)
            
            self.logger.log_processing_step("Generating SRT subtitle file")
            srt_path = self._generate_srt_file(caption_segments, output_path)
            
            self.logger.log_processing_step("Generating ASS subtitle file with formatting")
            ass_path = self._generate_ass_file(caption_segments, output_path)
            
            self.logger.log_processing_step("Burning captions into video")
            
            # Try burning captions with retry logic for OOM errors
            try:
                self._burn_captions_with_ffmpeg(video_path, ass_path, str(output_video_path))
            except Exception as e:
                error_msg = str(e)
                # If OOM error, try with downscaled video
                if "insufficient memory" in error_msg.lower() or "terminated" in error_msg.lower():
                    self.logger.warning("First attempt failed due to memory constraints, retrying with downscaled video...")
                    self.logger.log_processing_step("Retrying with memory-optimized settings")
                    self._burn_captions_with_ffmpeg_lowmem(video_path, ass_path, str(output_video_path))
                else:
                    raise
            
            if not output_video_path.exists():
                raise Exception("Captioned video was not created")
                
            # Clean up temporary subtitle files
            try:
                srt_path.unlink(missing_ok=True)
                ass_path.unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(f"Could not remove temp subtitle files: {e}")
                
            processing_time = time.time() - start_time
            self.logger.info(f"Caption generation completed in {processing_time:.2f}s")
            
            return str(output_video_path)
            
        except Exception as e:
            self.logger.error(f"Caption generation failed: {str(e)}", exc_info=True)
            raise Exception(f"Failed to generate captioned video: {str(e)}")
            
    def _process_transcript_segments(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process transcript segments for optimal YouTube-style caption display"""
        segments = transcript.get("segments", [])
        
        processed_segments = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
                
            start_time = segment.get("start", 0)
            end_time = segment.get("end", start_time + 3.0)
            words = segment.get("words", [])
            
            if words:
                youtube_segments = self._create_youtube_style_captions(words, start_time, end_time)
                processed_segments.extend(youtube_segments)
            else:
                youtube_segments = self._split_into_short_segments(text, start_time, end_time)
                processed_segments.extend(youtube_segments)
                
        processed_segments = self._resolve_overlapping_captions(processed_segments)
        
        self.logger.info(f"Processed {len(processed_segments)} caption segments")
        return processed_segments
    
    def _create_youtube_style_captions(self, words: List[Dict[str, Any]], 
                                      segment_start: float, segment_end: float) -> List[Dict[str, Any]]:
        """Create YouTube-style captions with 3-5 second duration and 5-8 words per caption"""
        segments = []
        current_words = []
        current_start = None
        
        MIN_WORDS = 3
        MAX_WORDS = 8
        MAX_DURATION = 5.0
        MAX_CHARS = 42
        
        for i, word_info in enumerate(words):
            word = word_info.get("word", "").strip()
            if not word:
                continue
            
            word_start = word_info.get("start", segment_start)
            word_end = word_info.get("end", word_start + 0.3)
            
            if current_start is None:
                current_start = word_start
            
            current_words.append({
                "text": word,
                "start": word_start,
                "end": word_end
            })
            
            caption_text = " ".join([w["text"] for w in current_words])
            caption_duration = word_end - current_start
            word_count = len(current_words)
            char_count = len(caption_text)
            
            should_break = False
            
            if word_count >= MIN_WORDS:
                if word.endswith((',', '.', '!', '?', ';', ':')):
                    should_break = True
                elif word_count >= MAX_WORDS or caption_duration >= MAX_DURATION or char_count >= MAX_CHARS:
                    should_break = True
                elif i == len(words) - 1:
                    should_break = True
            elif word_count > MAX_WORDS or caption_duration > MAX_DURATION or char_count > MAX_CHARS:
                should_break = True
            elif i == len(words) - 1:
                should_break = True
            
            if should_break and current_words:
                caption_text = " ".join([w["text"] for w in current_words])
                caption_end = current_words[-1]["end"]
                
                wrapped_lines = self._wrap_caption_text(caption_text)
                
                segment = {
                    "text": "\n".join(wrapped_lines[:2]),
                    "start": current_start,
                    "end": caption_end,
                    "confidence": 0.9
                }
                segments.append(segment)
                
                current_words = []
                current_start = None
        
        return segments
        
    def _wrap_caption_text(self, text: str) -> List[str]:
        """Wrap text to fit caption display requirements"""
        text = " ".join(text.split())
        
        wrapper = textwrap.TextWrapper(
            width=self.max_chars_per_line,
            break_long_words=False,
            break_on_hyphens=False,
            expand_tabs=False,
            replace_whitespace=True
        )
        
        lines = wrapper.wrap(text)
        return lines[:self.max_lines]
    
    def _split_into_short_segments(self, text: str, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Split text into short 3-5 second segments (fallback when no word timestamps)"""
        words = text.split()
        if not words:
            return []
        
        segments = []
        duration = end_time - start_time
        words_per_second = len(words) / duration if duration > 0 else 3
        
        TARGET_WORDS = 6
        current_words = []
        current_start = start_time
        
        for i, word in enumerate(words):
            current_words.append(word)
            
            should_break = False
            if len(current_words) >= TARGET_WORDS:
                if word.endswith(('.', '!', '?', ',', ';', ':')):
                    should_break = True
                elif len(current_words) >= 8:
                    should_break = True
            
            if i == len(words) - 1:
                should_break = True
            
            if should_break and current_words:
                caption_text = " ".join(current_words)
                word_count = len(current_words)
                
                caption_duration = word_count / words_per_second if words_per_second > 0 else 3.0
                caption_end = min(current_start + caption_duration, end_time)
                
                wrapped_lines = self._wrap_caption_text(caption_text)
                
                segment = {
                    "text": "\n".join(wrapped_lines[:2]),
                    "start": current_start,
                    "end": caption_end,
                    "confidence": 0.9
                }
                segments.append(segment)
                
                current_words = []
                current_start = caption_end
        
        return segments
        
    def _resolve_overlapping_captions(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve overlapping caption segments"""
        if not segments:
            return segments
            
        segments.sort(key=lambda x: x["start"])
        
        resolved_segments = []
        
        for i, segment in enumerate(segments):
            current_segment = segment.copy()
            
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                
                if current_segment["end"] > next_segment["start"]:
                    gap = 0.1
                    current_segment["end"] = max(
                        current_segment["start"] + 0.5,
                        next_segment["start"] - gap
                    )
                    
            resolved_segments.append(current_segment)
            
        return resolved_segments
        
    def _generate_srt_file(self, segments: List[Dict[str, Any]], output_dir: Path) -> Path:
        """Generate SRT subtitle file"""
        srt_path = output_dir / "captions.srt"
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._format_srt_timestamp(segment["start"])
                end_time = self._format_srt_timestamp(segment["end"])
                text = segment["text"]
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
                
        self.logger.info(f"Generated SRT file: {srt_path}")
        return srt_path
        
    def _generate_ass_file(self, segments: List[Dict[str, Any]], output_dir: Path) -> Path:
        """Generate ASS subtitle file with advanced formatting"""
        ass_path = output_dir / "captions.ass"
        
        ass_content = self._generate_ass_header()
        
        for segment in segments:
            start_time = self._format_ass_timestamp(segment["start"])
            end_time = self._format_ass_timestamp(segment["end"])
            text = segment["text"].replace("\n", "\\N")
            
            dialogue_line = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"
            ass_content += dialogue_line
            
        with open(ass_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)
            
        self.logger.info(f"Generated ASS file: {ass_path}")
        return ass_path
        
    def _generate_ass_header(self) -> str:
        """Generate ASS file header with styling"""
        if self.position == "top":
            alignment = 2
            margin_v = 30
        elif self.position == "bottom":
            alignment = 2
            margin_v = 30
        else:
            alignment = 2
            margin_v = 0
            
        primary_color = self._color_to_ass(self.font_color)
        outline_color = self._color_to_ass(self.background_color)
        
        # Calculate alpha for background (00 = opaque, FF = fully transparent)
        # background_opacity: 1.0 = fully opaque (alpha=00), 0.0 = fully transparent (alpha=FF)
        back_alpha = int((1 - self.background_opacity) * 255)
        
        # BackColour format: &HAABBGGRR where AA is alpha
        back_colour = f"&H{back_alpha:02X}000000"  # Semi-transparent black
        
        header = f"""[Script Info]
Title: Meeting Captions
ScriptType: v4.00+
PlayDepth: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,{self.font_family},{self.font_size},{primary_color},&H000000FF,{outline_color},{back_colour},-1,0,0,0,100,100,0,0,4,1,2,{alignment},10,10,{margin_v},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
        
        return header
        
    def _color_to_ass(self, color: str) -> str:
        """Convert color name/hex to ASS format"""
        color_map = {
            "white": "&HFFFFFF",
            "black": "&H000000",
            "red": "&H0000FF",
            "green": "&H00FF00",
            "blue": "&HFF0000",
            "yellow": "&H00FFFF",
            "cyan": "&HFFFF00",
            "magenta": "&HFF00FF"
        }
        
        if color.lower() in color_map:
            return color_map[color.lower()]
        elif color.startswith('#'):
            hex_color = color[1:]
            if len(hex_color) == 6:
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                return f"&H{b:02X}{g:02X}{r:02X}"
                
        return "&HFFFFFF"
        
    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
        
    def _format_ass_timestamp(self, seconds: float) -> str:
        """Format timestamp for ASS format (H:MM:SS.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
        
    def _burn_captions_with_ffmpeg(self, video_path: str, subtitle_path: str, output_path: str):
        """Burn captions into video using FFmpeg - Cross-platform with memory optimization"""
        try:
            subtitle_path_str = str(subtitle_path)
            
            # Cross-platform path handling for FFmpeg
            if self.IS_WINDOWS:
                # On Windows: use forward slashes and escape colons
                subtitle_path_normalized = subtitle_path_str.replace('\\', '/')
                # Escape the colon in drive letter (C: becomes C\\:)
                if len(subtitle_path_normalized) > 1 and subtitle_path_normalized[1] == ':':
                    subtitle_path_normalized = subtitle_path_normalized[0] + '\\:' + subtitle_path_normalized[2:]
            else:
                # On Unix: escape special characters
                subtitle_path_normalized = subtitle_path_str.replace("'", r"\'")
            
            # Memory-efficient FFmpeg command for container environments
            # Uses lower thread count, smaller buffers, and faster preset to reduce memory usage
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f"ass=filename='{subtitle_path_normalized}'",
                '-c:v', 'libx264',
                '-preset', 'veryfast',  # Faster encoding = less memory
                '-crf', str(self.compression_quality),
                '-threads', '2',  # Limit threads to reduce memory
                '-bufsize', '2M',  # Smaller buffer size
                '-maxrate', '5M',  # Limit bitrate
                '-c:a', 'copy',
                '-max_muxing_queue_size', '1024',  # Prevent queue overflow
                '-y',
                output_path
            ]
            
            self.logger.info(f"Running FFmpeg with memory-efficient settings: {' '.join(cmd)}")
            
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
                    if "time=" in output:
                        self._log_ffmpeg_progress(output)
                        
            process.wait()
            
            # Handle different error codes
            if process.returncode != 0:
                error_msg = '\n'.join(stderr_output[-10:])
                
                # Check if it's an OOM kill (signal -9)
                if process.returncode == -9:
                    raise Exception(
                        "FFmpeg was terminated due to insufficient memory. "
                        "The video file may be too large for the available server resources. "
                        "Please try with a smaller video file or contact support to increase memory limits."
                    )
                else:
                    raise Exception(f"FFmpeg failed with return code {process.returncode}:\n{error_msg}")
                
            self.logger.info(f"Successfully burned captions: {output_path}")
            
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in PATH")
        except Exception as e:
            raise Exception(f"Caption burning failed: {str(e)}")
    
    def _burn_captions_with_ffmpeg_lowmem(self, video_path: str, subtitle_path: str, output_path: str):
        """
        Burn captions with extreme memory optimization - downscales to 720p
        This is a fallback for when the normal method fails due to OOM
        """
        try:
            subtitle_path_str = str(subtitle_path)
            
            # Cross-platform path handling for FFmpeg
            if self.IS_WINDOWS:
                subtitle_path_normalized = subtitle_path_str.replace('\\', '/')
                if len(subtitle_path_normalized) > 1 and subtitle_path_normalized[1] == ':':
                    subtitle_path_normalized = subtitle_path_normalized[0] + '\\:' + subtitle_path_normalized[2:]
            else:
                subtitle_path_normalized = subtitle_path_str.replace("'", r"\'")
            
            # Ultra low-memory settings: downscale to 720p, single thread, minimal buffers
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f"scale=-2:720,ass=filename='{subtitle_path_normalized}'",  # Downscale to 720p
                '-c:v', 'libx264',
                '-preset', 'ultrafast',  # Fastest encoding
                '-crf', '28',  # Higher CRF = lower quality but much less memory
                '-threads', '1',  # Single thread
                '-bufsize', '1M',  # Minimal buffer
                '-maxrate', '2M',  # Lower bitrate
                '-c:a', 'aac',  # Re-encode audio to AAC (smaller)
                '-b:a', '128k',  # Lower audio bitrate
                '-max_muxing_queue_size', '512',
                '-y',
                output_path
            ]
            
            self.logger.info(f"Running FFmpeg with ultra-low-memory settings (720p): {' '.join(cmd)}")
            
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
                    if "time=" in output:
                        self._log_ffmpeg_progress(output)
                        
            process.wait()
            
            if process.returncode != 0:
                error_msg = '\n'.join(stderr_output[-10:])
                if process.returncode == -9:
                    raise Exception(
                        "Video processing failed even with reduced quality settings. "
                        "The server does not have enough memory to process this video. "
                        "Please try with a shorter or smaller video file."
                    )
                else:
                    raise Exception(f"FFmpeg (low-mem mode) failed with return code {process.returncode}:\n{error_msg}")
                
            self.logger.info(f"Successfully burned captions with low-memory mode: {output_path}")
            
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in PATH")
        except Exception as e:
            raise Exception(f"Low-memory caption burning failed: {str(e)}")
            
            
    def _log_ffmpeg_progress(self, output: str):
        """Parse and log FFmpeg progress"""
        try:
            if "time=" in output:
                time_part = output.split("time=")[1].split()[0]
                self.logger.debug(f"FFmpeg progress: {time_part}")
        except (IndexError, ValueError):
            pass
            
    def generate_subtitle_files_only(self, transcript: Dict[str, Any], 
                                   output_dir: str) -> Dict[str, str]:
        """Generate subtitle files without burning them into video"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            caption_segments = self._process_transcript_segments(transcript)
            
            srt_path = self._generate_srt_file(caption_segments, output_path)
            ass_path = self._generate_ass_file(caption_segments, output_path)
            vtt_path = self._generate_vtt_file(caption_segments, output_path)
            
            result = {
                "srt": str(srt_path),
                "ass": str(ass_path),
                "vtt": str(vtt_path)
            }
            
            self.logger.info(f"Generated subtitle files: {list(result.keys())}")
            return result
            
        except Exception as e:
            self.logger.error(f"Subtitle generation failed: {str(e)}")
            raise Exception(f"Failed to generate subtitle files: {str(e)}")
            
    def _generate_vtt_file(self, segments: List[Dict[str, Any]], output_dir: Path) -> Path:
        """Generate WebVTT subtitle file"""
        vtt_path = output_dir / "captions.vtt"
        
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for i, segment in enumerate(segments, 1):
                start_time = self._format_vtt_timestamp(segment["start"])
                end_time = self._format_vtt_timestamp(segment["end"])
                text = segment["text"]
                
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
                
        self.logger.info(f"Generated VTT file: {vtt_path}")
        return vtt_path
        
    def _format_vtt_timestamp(self, seconds: float) -> str:
        """Format timestamp for VTT format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"