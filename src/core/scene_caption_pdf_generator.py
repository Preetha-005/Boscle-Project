"""
Scene Caption PDF Generator for Meeting Video Captioning Program
Generates a PDF with screenshots at scene changes paired with corresponding captions/transcripts
"""

import cv2
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np


class SceneCaptionPDFGenerator:
    """
    Generates a PDF document containing screenshots captured at scene changes
    along with the corresponding captions/transcripts spoken at those moments.
    """
    
    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger
        
        # Configuration for scene detection
        self.scene_threshold = config_manager.get("video_processing.scene_detection_threshold", 0.3)
        self.min_scene_duration = 2.0  # Minimum seconds between scene captures
        self.max_screenshots = 50  # Maximum screenshots to include in PDF
        
    def generate_scene_caption_pdf(self, video_path: str, transcript: Dict[str, Any], 
                                   output_dir: str) -> Optional[str]:
        """
        Generate a PDF with scene change screenshots and corresponding captions.
        
        Args:
            video_path: Path to the video file
            transcript: Transcript data with segments containing timestamps and text
            output_dir: Directory to save the output PDF
            
        Returns:
            Path to the generated PDF file, or None if generation fails
        """
        try:
            self.logger.info(f"Starting scene caption PDF generation for: {video_path}")
            
            # Extract scene changes with screenshots
            scenes = self._extract_scene_changes(video_path)
            self.logger.info(f"Detected {len(scenes)} scene changes")
            
            if not scenes:
                self.logger.warning("No scene changes detected, capturing key frames instead")
                scenes = self._extract_key_frames(video_path)
            
            # Match scenes with transcript segments
            scenes_with_captions = self._match_scenes_with_captions(scenes, transcript)
            self.logger.info(f"Matched {len(scenes_with_captions)} scenes with captions")
            
            # Generate PDF
            pdf_path = self._generate_pdf(scenes_with_captions, output_dir, video_path)
            self.logger.info(f"Generated scene caption PDF: {pdf_path}")
            
            # Cleanup temporary frame files
            self._cleanup_temp_frames(scenes)
            
            return pdf_path
            
        except Exception as e:
            self.logger.error(f"Scene caption PDF generation failed: {e}", exc_info=True)
            return None
    
    def _extract_scene_changes(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract screenshots at scene change points in the video.
        Uses histogram comparison for scene detection.
        """
        scenes = []
        cap = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            self.logger.info(f"Video: {fps:.2f} fps, {total_frames} frames, {duration:.2f}s duration")
            
            previous_frame = None
            previous_hist = None
            last_scene_time = -self.min_scene_duration
            frame_number = 0
            
            # Create temp directory for frames
            temp_dir = self.config_manager.get_temp_dir() / "scene_frames"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Always capture the first frame
            ret, frame = cap.read()
            if ret:
                timestamp = 0.0
                frame_path = self._save_frame(frame, timestamp, temp_dir)
                scenes.append({
                    "timestamp": timestamp,
                    "timestamp_formatted": self._format_timestamp(timestamp),
                    "frame_path": frame_path,
                    "description": "Video start",
                    "scene_confidence": 1.0
                })
                last_scene_time = 0
                previous_frame = frame.copy()
                previous_hist = self._calculate_histogram(frame)
            
            frame_number = 1
            
            # Process remaining frames (sample every 0.5 seconds for efficiency)
            sample_interval = max(1, int(fps * 0.5))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % sample_interval == 0:
                    timestamp = frame_number / fps
                    
                    # Check for scene change
                    if previous_frame is not None:
                        is_scene_change, confidence = self._detect_scene_change(
                            previous_hist, frame
                        )
                        
                        # Only capture if it's a scene change and enough time has passed
                        if is_scene_change and (timestamp - last_scene_time) >= self.min_scene_duration:
                            if len(scenes) < self.max_screenshots:
                                frame_path = self._save_frame(frame, timestamp, temp_dir)
                                scenes.append({
                                    "timestamp": timestamp,
                                    "timestamp_formatted": self._format_timestamp(timestamp),
                                    "frame_path": frame_path,
                                    "description": f"Scene change (confidence: {confidence:.2f})",
                                    "scene_confidence": confidence
                                })
                                last_scene_time = timestamp
                    
                    previous_frame = frame.copy()
                    previous_hist = self._calculate_histogram(frame)
                
                frame_number += 1
            
            cap.release()
            
            # If we have very few scenes, distribute evenly across video
            if len(scenes) < 5 and duration > 30:
                scenes = self._extract_evenly_distributed_frames(video_path, min(10, int(duration / 30)))
            
            return scenes
            
        except Exception as e:
            if cap is not None:
                cap.release()
            self.logger.error(f"Scene extraction failed: {e}", exc_info=True)
            return []
    
    def _extract_key_frames(self, video_path: str, num_frames: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key frames evenly distributed across the video.
        Fallback when scene detection finds too few changes.
        """
        return self._extract_evenly_distributed_frames(video_path, num_frames)
    
    def _extract_evenly_distributed_frames(self, video_path: str, num_frames: int = 10) -> List[Dict[str, Any]]:
        """Extract frames evenly distributed across video duration."""
        frames = []
        cap = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            temp_dir = self.config_manager.get_temp_dir() / "scene_frames"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate frame intervals
            interval = duration / (num_frames + 1)
            
            for i in range(1, num_frames + 1):
                timestamp = i * interval
                frame_number = int(timestamp * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    frame_path = self._save_frame(frame, timestamp, temp_dir)
                    frames.append({
                        "timestamp": timestamp,
                        "timestamp_formatted": self._format_timestamp(timestamp),
                        "frame_path": frame_path,
                        "description": f"Key frame at {self._format_timestamp(timestamp)}",
                        "scene_confidence": 0.5
                    })
            
            cap.release()
            return frames
            
        except Exception as e:
            if cap is not None:
                cap.release()
            self.logger.error(f"Key frame extraction failed: {e}")
            return []
    
    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate normalized color histogram for a frame."""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return hist
    
    def _detect_scene_change(self, previous_hist: np.ndarray, current_frame: np.ndarray) -> tuple:
        """
        Detect if there's a scene change between the previous histogram and current frame.
        Returns (is_scene_change, confidence_score)
        """
        try:
            current_hist = self._calculate_histogram(current_frame)
            
            # Compare histograms using correlation
            correlation = cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL)
            
            # Calculate difference score (1 - correlation)
            difference_score = 1 - correlation
            
            # Scene change if difference exceeds threshold
            is_scene_change = difference_score > self.scene_threshold
            
            return is_scene_change, difference_score
            
        except Exception as e:
            self.logger.warning(f"Scene detection error: {e}")
            return False, 0.0
    
    def _save_frame(self, frame: np.ndarray, timestamp: float, output_dir: Path) -> str:
        """Save a frame to disk and return the file path."""
        try:
            timestamp_str = f"{timestamp:.2f}".replace(".", "_")
            filename = f"scene_{timestamp_str}.jpg"
            file_path = output_dir / filename
            
            # Save with good quality
            cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            return str(file_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to save frame: {e}")
            return ""
    
    def _match_scenes_with_captions(self, scenes: List[Dict[str, Any]], 
                                    transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Match each scene with captions by estimating sentence-level timing.
        Shows only sentences spoken during each scene's specific time window.
        """
        import re
        
        segments = transcript.get("segments", [])
        
        if not segments:
            self.logger.warning("No transcript segments available")
            return [dict(scene, caption="[No caption available]") for scene in scenes]
        
        # Build a list of all sentences with estimated timestamps
        sentence_list = []
        
        for segment in segments:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            seg_duration = seg_end - seg_start
            text = segment.get("text", "").strip()
            
            if not text:
                continue
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                continue
            
            # Estimate timestamp for each sentence within the segment
            num_sentences = len(sentences)
            time_per_sentence = seg_duration / num_sentences
            
            for idx, sentence in enumerate(sentences):
                # Estimate when this sentence was spoken
                estimated_start = seg_start + (idx * time_per_sentence)
                estimated_end = seg_start + ((idx + 1) * time_per_sentence)
                
                sentence_list.append({
                    'text': sentence,
                    'start': estimated_start,
                    'end': estimated_end,
                    'segment_start': seg_start,
                    'segment_end': seg_end
                })
        
        matched_scenes = []
        
        # Sort scenes by timestamp to determine time windows
        sorted_scenes = sorted(scenes, key=lambda s: s["timestamp"])
        
        for i, scene in enumerate(sorted_scenes):
            scene_time = scene["timestamp"]
            
            # Define time window for this scene
            if i < len(sorted_scenes) - 1:
                next_scene_time = sorted_scenes[i + 1]["timestamp"]
                window_end = next_scene_time
            else:
                window_end = scene_time + 8.0  # 8 seconds after last scene
            
            window_start = scene_time  # Start from scene timestamp
            
            # Collect sentences that fall within this time window
            relevant_sentences = []
            earliest_start = None
            latest_end = None
            
            for sentence_data in sentence_list:
                sent_start = sentence_data['start']
                sent_end = sentence_data['end']
                sent_mid = (sent_start + sent_end) / 2
                
                # Include sentence if its midpoint is within the time window
                if window_start <= sent_mid <= window_end:
                    relevant_sentences.append(sentence_data)
                    
                    if earliest_start is None or sent_start < earliest_start:
                        earliest_start = sent_start
                    if latest_end is None or sent_end > latest_end:
                        latest_end = sent_end
            
            # Build caption text from relevant sentences
            if relevant_sentences:
                caption_parts = [s['text'] for s in relevant_sentences]
                caption_text = " ".join(caption_parts)
                caption_time = f"[{self._format_timestamp(earliest_start)} - {self._format_timestamp(latest_end)}]"
            else:
                # Fallback: find closest sentence
                closest_sentence = None
                min_distance = float('inf')
                
                for sentence_data in sentence_list:
                    sent_mid = (sentence_data['start'] + sentence_data['end']) / 2
                    distance = abs(scene_time - sent_mid)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_sentence = sentence_data
                
                if closest_sentence and min_distance < 5.0:
                    caption_text = closest_sentence['text']
                    caption_time = f"[{self._format_timestamp(closest_sentence['start'])} - {self._format_timestamp(closest_sentence['end'])}]"
                else:
                    caption_text = "[No caption available for this scene]"
                    caption_time = ""
            
            matched_scene = dict(scene)
            matched_scene["caption"] = caption_text
            matched_scene["caption_time"] = caption_time
            matched_scenes.append(matched_scene)
        
        # Sort back to original order
        scene_to_index = {id(s): idx for idx, s in enumerate(scenes)}
        matched_scenes.sort(key=lambda s: scene_to_index.get(id(s), 0))
        
        return matched_scenes
    
    def _generate_pdf(self, scenes: List[Dict[str, Any]], output_dir: str, 
                      video_path: str) -> str:
        """Generate the PDF document with scene screenshots and captions in tabular format."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, 
                                           Image, Table, TableStyle, PageBreak)
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create output file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"scene_captions_{timestamp}.pdf"
            pdf_path = output_path / pdf_filename
            
            # Create document
            doc = SimpleDocTemplate(
                str(pdf_path), 
                pagesize=letter,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch,
                leftMargin=0.5*inch,
                rightMargin=0.5*inch
            )
            
            # Create custom styles
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=22,
                spaceAfter=20,
                textColor=colors.HexColor('#1a365d'),
                alignment=TA_CENTER
            )
            
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=15,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
            
            scene_header_style = ParagraphStyle(
                'SceneHeader',
                parent=styles['Heading3'],
                fontSize=12,
                spaceBefore=5,
                spaceAfter=5,
                textColor=colors.HexColor('#2c5282'),
                alignment=TA_LEFT
            )
            
            timestamp_style = ParagraphStyle(
                'Timestamp',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#718096'),
                spaceAfter=5
            )
            
            bullet_style = ParagraphStyle(
                'BulletCaption',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                textColor=colors.HexColor('#2d3748'),
                leftIndent=10,
                bulletIndent=0,
                spaceBefore=3,
                spaceAfter=3
            )
            
            intro_style = ParagraphStyle(
                'Intro',
                parent=styles['Normal'],
                fontSize=10,
                leading=16,
                textColor=colors.HexColor('#4a5568'),
                alignment=TA_CENTER,
                spaceBefore=10,
                spaceAfter=20
            )
            
            # Build story
            story = []
            
            # Title
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Scene Captions Report", title_style))
            
            video_name = Path(video_path).stem
            story.append(Paragraph(f"Video: {video_name}", subtitle_style))

            
            story.append(Spacer(1, 10))
            
            # Table header
            header_style = ParagraphStyle(
                'TableHeader',
                parent=styles['Heading4'],
                fontSize=11,
                textColor=colors.white,
                alignment=TA_CENTER
            )
            
            # Process each scene in tabular format
            for i, scene in enumerate(scenes, 1):
                # Create table data for this scene
                table_data = []
                
                # Left column: Screenshot
                left_content = []
                
                # Scene number and timestamp header
                left_content.append(Paragraph(
                    f"<b>Scene {i}</b>", 
                    scene_header_style
                ))
                left_content.append(Paragraph(
                    f"⏱ {scene['timestamp_formatted']}", 
                    timestamp_style
                ))
                
                # Add screenshot
                if scene.get("frame_path") and Path(scene["frame_path"]).exists():
                    try:
                        img = Image(scene["frame_path"], width=3.2*inch, height=2.2*inch)
                        left_content.append(img)
                    except Exception as e:
                        self.logger.warning(f"Could not add image: {e}")
                        left_content.append(Paragraph(
                            "[Screenshot not available]", 
                            styles['Normal']
                        ))
                else:
                    left_content.append(Paragraph(
                        "[Screenshot not available]", 
                        styles['Normal']
                    ))
                
                # Right column: Captions as bullet points
                right_content = []
                
                # Caption header
                right_content.append(Paragraph(
                    "<b>Caption:</b>", 
                    scene_header_style
                ))
                
                # Caption timestamp
                if scene.get("caption_time"):
                    right_content.append(Paragraph(
                        f"<i>{scene['caption_time']}</i>", 
                        timestamp_style
                    ))
                
                right_content.append(Spacer(1, 5))
                
                # Caption text as bullet points
                caption = scene.get("caption", "[No caption available]")
                if caption:
                    # Escape any special characters for ReportLab
                    caption = caption.replace("&", "&amp;")
                    caption = caption.replace("<", "&lt;")
                    caption = caption.replace(">", "&gt;")
                    
                    # Split caption into sentences for bullet points
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', caption)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    
                    if sentences:
                        for sentence in sentences[:8]:  # Limit to 8 bullet points
                            # Add bullet point
                            right_content.append(Paragraph(
                                f"• {sentence}", 
                                bullet_style
                            ))
                    else:
                        right_content.append(Paragraph(
                            f"• {caption}", 
                            bullet_style
                        ))
                else:
                    right_content.append(Paragraph(
                        "• [No caption available for this scene]", 
                        bullet_style
                    ))
                
                # Create table with two columns
                table_data.append([left_content, right_content])
                
                # Create table with styling
                scene_table = Table(
                    table_data, 
                    colWidths=[3.5*inch, 3.5*inch],
                    hAlign='CENTER'
                )
                
                scene_table.setStyle(TableStyle([
                    # Alignment
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # Left column centered
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),    # Right column left-aligned
                    
                    # Padding
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    
                    # Borders and background
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e0')),
                    ('LINEAFTER', (0, 0), (0, -1), 1, colors.HexColor('#e2e8f0')),
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
                    ('BACKGROUND', (1, 0), (1, -1), colors.white),
                ]))
                
                story.append(scene_table)
                story.append(Spacer(1, 15))
                
                # Add page break every 2 scenes
                if i % 2 == 0 and i < len(scenes):
                    story.append(PageBreak())
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"PDF generated successfully: {pdf_path}")
            return str(pdf_path)
            
        except ImportError as e:
            self.logger.error(f"ReportLab not available: {e}")
            return self._generate_text_fallback(scenes, output_dir, video_path)
        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}", exc_info=True)
            return self._generate_text_fallback(scenes, output_dir, video_path)
    
    def _generate_text_fallback(self, scenes: List[Dict[str, Any]], output_dir: str,
                                video_path: str) -> str:
        """Generate a text file as fallback when PDF generation fails."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            txt_filename = f"scene_captions_{timestamp}.txt"
            txt_path = output_path / txt_filename
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SCENE CAPTIONS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Video: {Path(video_path).name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Scenes: {len(scenes)}\n")
                f.write("\n" + "-" * 80 + "\n\n")
                
                for i, scene in enumerate(scenes, 1):
                    f.write(f"SCENE {i} - {scene['timestamp_formatted']}\n")
                    f.write("-" * 40 + "\n")
                    if scene.get("caption_time"):
                        f.write(f"Caption Time: {scene['caption_time']}\n")
                    f.write(f"Caption: {scene.get('caption', '[No caption]')}\n")
                    f.write(f"Screenshot: {scene.get('frame_path', 'Not available')}\n")
                    f.write("\n" + "-" * 80 + "\n\n")
            
            self.logger.info(f"Text fallback generated: {txt_path}")
            return str(txt_path)
            
        except Exception as e:
            self.logger.error(f"Text fallback generation failed: {e}")
            return ""
    
    def _cleanup_temp_frames(self, scenes: List[Dict[str, Any]]):
        """Clean up temporary frame files after PDF generation."""
        for scene in scenes:
            frame_path = scene.get("frame_path")
            if frame_path and Path(frame_path).exists():
                try:
                    Path(frame_path).unlink()
                except Exception as e:
                    self.logger.warning(f"Could not delete temp frame {frame_path}: {e}")
        
        # Try to remove the temp directory if empty
        try:
            temp_dir = self.config_manager.get_temp_dir() / "scene_frames"
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception as e:
            self.logger.warning(f"Could not remove temp directory: {e}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
