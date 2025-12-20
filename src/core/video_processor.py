"""
Core video processing module for the Meeting Video Captioning Program
Handles frame extraction, scene detection, and video analysis
Cross-platform compatible (Windows/Linux/macOS)
"""

import cv2
import numpy as np
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime, timedelta
import hashlib
import json


class VideoProcessor:
    """Main video processing class for frame extraction and scene analysis"""
    
    SYSTEM = platform.system()
    IS_WINDOWS = SYSTEM == "Windows"
    
    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger
        
        # Configuration
        self.scene_threshold = config_manager.get("video_processing.scene_detection_threshold", 0.3)
        self.frame_interval = config_manager.get("video_processing.frame_extraction_interval", 1.0)
        self.max_resolution = config_manager.get("video_processing.max_resolution", "1920x1080")
        self.fps_threshold = config_manager.get("video_processing.fps_threshold", 30)
        
        # Processing state
        self.current_video = None
        self.video_info = {}
        self.extracted_frames = []
        self.scene_changes = []
        
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Main video processing function
        Returns comprehensive analysis of the video
        """
        start_time = time.time()
        self.logger.log_processing_start(video_path, "local")
        
        cap = None
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
                
            # Get video information
            self.video_info = self._extract_video_info(cap, video_path)
            self.logger.log_processing_step("Video info extraction", 10)
            
            # Extract frames and detect scenes
            frames_data = self._extract_frames_and_scenes(cap)
            self.logger.log_processing_step("Frame extraction and scene detection", 60)
            
            # Analyze interactions and transitions
            interaction_data = self._analyze_interactions(frames_data)
            self.logger.log_processing_step("Interaction analysis", 80)
            
            # Generate summary
            analysis_summary = self._generate_analysis_summary(frames_data, interaction_data)
            self.logger.log_processing_step("Analysis summary generation", 90)
            
            cap.release()
            
            # Compile results
            result = {
                "video_info": self.video_info,
                "frames": frames_data,
                "interactions": interaction_data,
                "summary": analysis_summary,
                "processing_time": time.time() - start_time
            }
            
            self.logger.log_processing_complete([video_path], time.time() - start_time)
            return result
            
        except Exception as e:
            if cap is not None:
                cap.release()
            self.logger.error(f"Video processing failed: {str(e)}", exc_info=True)
            raise
            
    def _extract_video_info(self, cap: cv2.VideoCapture, video_path: str) -> Dict[str, Any]:
        """Extract basic video information"""
        info = {
            "path": video_path,
            "filename": Path(video_path).name,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": 0,
            "file_size": Path(video_path).stat().st_size if Path(video_path).exists() else 0,
            "creation_time": datetime.now().isoformat(),
            "codec": None
        }
        
        # Calculate duration
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
            
        # Try to get codec information
        try:
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            if fourcc:
                info["codec"] = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not get codec info: {e}")
            
        self.logger.info(f"Video info: {info['width']}x{info['height']}, {info['fps']:.2f}fps, {info['duration']:.2f}s")
        return info
        
    def _extract_frames_and_scenes(self, cap: cv2.VideoCapture) -> List[Dict[str, Any]]:
        """Extract frames at intervals and detect scene changes"""
        frames_data = []
        previous_frame = None
        frame_number = 0
        
        fps = self.video_info["fps"]
        frame_interval_frames = max(1, int(fps * self.frame_interval))
        
        self.logger.info(f"Extracting frames every {self.frame_interval}s ({frame_interval_frames} frames)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract frame at specified intervals
            if frame_number % frame_interval_frames == 0:
                timestamp = frame_number / fps
                
                # Detect scene change
                is_scene_change = False
                scene_confidence = 0.0
                
                if previous_frame is not None:
                    is_scene_change, scene_confidence = self._detect_scene_change(
                        previous_frame, frame
                    )
                    
                # Analyze frame content
                frame_analysis = self._analyze_frame_content(frame, timestamp)
                
                # Store frame data
                frame_data = {
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "timestamp_formatted": self._format_timestamp(timestamp),
                    "is_scene_change": is_scene_change,
                    "scene_confidence": scene_confidence,
                    "analysis": frame_analysis,
                    "file_path": None
                }
                
                # Save frame if it's a scene change or significant
                if is_scene_change or len(frames_data) == 0 or frame_analysis.get("significant_change", False):
                    frame_path = self._save_frame(frame, timestamp)
                    frame_data["file_path"] = frame_path
                    
                frames_data.append(frame_data)
                previous_frame = frame.copy()
                
            frame_number += 1
            
            # Update progress periodically
            if frame_number % (fps * 10) == 0:  # Every 10 seconds
                progress = (frame_number / self.video_info["frame_count"]) * 50 + 10
                self.logger.log_processing_step(f"Processing frame {frame_number}", progress)
                
        self.logger.info(f"Extracted {len(frames_data)} frames, detected {sum(f['is_scene_change'] for f in frames_data)} scene changes")
        return frames_data
        
    def _detect_scene_change(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect scene change between two frames
        Returns: (is_scene_change, confidence_score)
        """
        try:
            # Resize frames for faster processing
            height, width = prev_frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                prev_frame = cv2.resize(prev_frame, (new_width, new_height))
                curr_frame = cv2.resize(curr_frame, (new_width, new_height))
                
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram difference
            hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
            hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist_prev, hist_prev, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)
            
            # Calculate correlation coefficient
            correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
            
            # Calculate structural similarity
            mean_prev = np.mean(prev_gray)
            mean_curr = np.mean(curr_gray)
            std_prev = np.std(prev_gray)
            std_curr = np.std(curr_gray)
            
            # Simple structural similarity approximation
            if std_prev == 0 or std_curr == 0:
                structural_sim = 0 if mean_prev != mean_curr else 1
            else:
                covariance = np.mean((prev_gray - mean_prev) * (curr_gray - mean_curr))
                structural_sim = (2 * mean_prev * mean_curr) / (mean_prev**2 + mean_curr**2) * \
                               (2 * covariance) / (std_prev**2 + std_curr**2)
                               
            # Combine metrics
            similarity_score = (correlation + max(0, structural_sim)) / 2
            difference_score = 1 - similarity_score
            
            is_scene_change = difference_score > self.scene_threshold
            
            return is_scene_change, difference_score
            
        except Exception as e:
            self.logger.warning(f"Error in scene detection: {e}")
            return False, 0.0
            
    def _analyze_frame_content(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Analyze frame content for interactions and significant elements"""
        try:
            analysis = {
                "brightness": 0,
                "contrast": 0,
                "has_text": False,
                "has_faces": False,
                "dominant_colors": [],
                "significant_change": False,
                "content_type": "unknown"
            }
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness and contrast
            analysis["brightness"] = float(np.mean(gray))
            analysis["contrast"] = float(np.std(gray))
            
            # Detect significant visual changes (high contrast areas)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            analysis["significant_change"] = edge_density > 0.1
            
            # Simple text detection using edge density in specific regions
            height, width = gray.shape
            text_regions = [
                gray[int(height*0.8):, :],  # Bottom region (common for subtitles)
                gray[:int(height*0.2), :],  # Top region (common for titles)
            ]
            
            for region in text_regions:
                if region.size > 0:
                    region_edges = cv2.Canny(region, 50, 150)
                    if np.sum(region_edges > 0) / region.size > 0.05:
                        analysis["has_text"] = True
                        break
                        
            # Detect dominant colors
            analysis["dominant_colors"] = self._get_dominant_colors(frame)
            
            # Classify content type based on characteristics
            if analysis["has_text"]:
                analysis["content_type"] = "presentation"
            elif analysis["brightness"] < 50:
                analysis["content_type"] = "dark_scene"
            elif analysis["contrast"] > 80:
                analysis["content_type"] = "high_contrast"
            else:
                analysis["content_type"] = "general"
                
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Error in frame analysis: {e}")
            return {"error": str(e)}
            
    def _get_dominant_colors(self, frame: np.ndarray, k: int = 3) -> List[List[int]]:
        """Extract dominant colors from frame using k-means clustering"""
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (150, 150))
            data = small_frame.reshape((-1, 3))
            data = np.float32(data)
            
            # Apply k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers back to uint8 and return as list
            centers = np.uint8(centers)
            dominant_colors = centers.tolist()
            
            return dominant_colors
            
        except Exception as e:
            self.logger.warning(f"Error extracting dominant colors: {e}")
            return []
            
    def _save_frame(self, frame: np.ndarray, timestamp: float) -> str:
        """Save frame to disk and return file path"""
        try:
            # Create frames directory
            output_dir = self.config_manager.get_temp_dir() / "frames"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp_str = f"{timestamp:.2f}".replace(".", "_")
            filename = f"frame_{timestamp_str}.jpg"
            file_path = output_dir / filename
            
            # Save frame with good quality
            cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            return str(file_path)
            
        except Exception as e:
            self.logger.warning(f"Error saving frame: {e}")
            return ""
            
    def _analyze_interactions(self, frames_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze potential interactions and user actions in the video"""
        interactions = []
        
        for i, frame_data in enumerate(frames_data):
            if frame_data["is_scene_change"]:
                interaction = {
                    "timestamp": frame_data["timestamp"],
                    "timestamp_formatted": frame_data["timestamp_formatted"],
                    "type": "scene_change",
                    "description": self._describe_scene_change(frame_data, frames_data[i-1] if i > 0 else None),
                    "confidence": frame_data["scene_confidence"],
                    "frame_path": frame_data.get("file_path")
                }
                interactions.append(interaction)
                
            # Detect potential clicks/interactions based on content changes
            analysis = frame_data.get("analysis", {})
            if analysis.get("significant_change", False):
                interaction = {
                    "timestamp": frame_data["timestamp"],
                    "timestamp_formatted": frame_data["timestamp_formatted"],
                    "type": "content_change",
                    "description": f"Significant content change detected - {analysis.get('content_type', 'unknown')}",
                    "confidence": 0.7,
                    "frame_path": frame_data.get("file_path")
                }
                interactions.append(interaction)
                
        self.logger.info(f"Detected {len(interactions)} potential interactions")
        return interactions
        
    def _describe_scene_change(self, current_frame: Dict[str, Any], previous_frame: Optional[Dict[str, Any]]) -> str:
        """Generate description for scene change"""
        if previous_frame is None:
            return "Video start"
            
        current_analysis = current_frame.get("analysis", {})
        previous_analysis = previous_frame.get("analysis", {})
        
        current_type = current_analysis.get("content_type", "unknown")
        previous_type = previous_analysis.get("content_type", "unknown")
        
        if current_type != previous_type:
            return f"Transition from {previous_type} to {current_type}"
        elif current_analysis.get("has_text", False) and not previous_analysis.get("has_text", False):
            return "Text content appeared"
        elif not current_analysis.get("has_text", False) and previous_analysis.get("has_text", False):
            return "Text content disappeared"
        else:
            confidence = current_frame.get("scene_confidence", 0)
            if confidence > 0.7:
                return "Major scene transition"
            elif confidence > 0.5:
                return "Moderate scene change"
            else:
                return "Minor content change"
                
    def _generate_analysis_summary(self, frames_data: List[Dict[str, Any]], 
                                 interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary"""
        summary = {
            "total_frames_analyzed": len(frames_data),
            "scene_changes": len([f for f in frames_data if f["is_scene_change"]]),
            "interactions_detected": len(interactions),
            "content_types": {},
            "timeline_segments": [],
            "key_moments": [],
            "processing_stats": {
                "avg_scene_confidence": 0,
                "frames_with_text": 0,
                "significant_changes": 0
            }
        }
        
        # Analyze content types distribution
        for frame in frames_data:
            content_type = frame.get("analysis", {}).get("content_type", "unknown")
            summary["content_types"][content_type] = summary["content_types"].get(content_type, 0) + 1
            
        # Calculate processing stats
        scene_confidences = [f["scene_confidence"] for f in frames_data if f["scene_confidence"] > 0]
        if scene_confidences:
            summary["processing_stats"]["avg_scene_confidence"] = float(np.mean(scene_confidences))
            
        summary["processing_stats"]["frames_with_text"] = sum(
            1 for f in frames_data if f.get("analysis", {}).get("has_text", False)
        )
        
        summary["processing_stats"]["significant_changes"] = sum(
            1 for f in frames_data if f.get("analysis", {}).get("significant_change", False)
        )
        
        # Generate timeline segments
        summary["timeline_segments"] = self._create_timeline_segments(frames_data)
        
        # Identify key moments (high confidence scene changes)
        summary["key_moments"] = [
            {
                "timestamp": interaction["timestamp"],
                "timestamp_formatted": interaction["timestamp_formatted"],
                "description": interaction["description"],
                "confidence": interaction["confidence"]
            }
            for interaction in interactions
            if interaction["confidence"] > 0.6
        ][:10]  # Top 10 key moments
        
        return summary
        
    def _create_timeline_segments(self, frames_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create timeline segments based on scene changes"""
        if not frames_data:
            return []
            
        segments = []
        current_segment_start = 0
        
        for i, frame in enumerate(frames_data):
            if frame["is_scene_change"] or i == len(frames_data) - 1:
                # End current segment
                if i > 0 or not frame["is_scene_change"]:
                    end_timestamp = frame["timestamp"] if i == len(frames_data) - 1 else frames_data[i-1]["timestamp"]
                    segment = {
                        "start_time": current_segment_start,
                        "end_time": end_timestamp,
                        "start_time_formatted": self._format_timestamp(current_segment_start),
                        "end_time_formatted": self._format_timestamp(end_timestamp),
                        "duration": end_timestamp - current_segment_start,
                        "content_type": self._get_segment_content_type(frames_data, 
                                                                     max(0, i-5), i),
                        "description": f"Segment {len(segments) + 1}"
                    }
                    segments.append(segment)
                    
                # Start new segment
                if frame["is_scene_change"]:
                    current_segment_start = frame["timestamp"]
                    
        return segments
        
    def _get_segment_content_type(self, frames_data: List[Dict[str, Any]], 
                                start_idx: int, end_idx: int) -> str:
        """Determine the most common content type in a segment"""
        content_types = []
        for i in range(start_idx, min(end_idx, len(frames_data))):
            content_type = frames_data[i].get("analysis", {}).get("content_type", "unknown")
            content_types.append(content_type)
            
        if content_types:
            return max(set(content_types), key=content_types.count)
        return "unknown"
        
    def _format_timestamp(self, secs: float) -> str:
        """Format timestamp as HH:MM:SS.mmm"""
        td = timedelta(seconds=secs)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds_part = total_seconds % 60
        milliseconds = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"
        
    def cleanup_temp_files(self):
        """Clean up temporary frame files"""
        try:
            frames_dir = self.config_manager.get_temp_dir() / "frames"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
                self.logger.info("Cleaned up temporary frame files")
        except Exception as e:
            self.logger.warning(f"Error cleaning up temp files: {e}")