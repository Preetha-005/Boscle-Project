"""
Audio transcription module for the Meeting Video Captioning Program
Handles speech-to-text conversion with multiple service providers
Cross-platform compatible (Windows/Linux/macOS)
"""

import os
import time
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import timedelta
import wave


class AudioTranscriber:
    """Handles audio transcription using multiple speech-to-text services"""
    
    SYSTEM = platform.system()
    IS_WINDOWS = SYSTEM == "Windows"
    
    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger
        
        # Configuration
        self.model_type = config_manager.get("audio_transcription.model", "whisper")
        self.language = config_manager.get("audio_transcription.language", "en")
        self.chunk_duration = config_manager.get("audio_transcription.chunk_duration", 30)
        self.confidence_threshold = config_manager.get("audio_transcription.confidence_threshold", 0.8)
        self.enable_speaker_detection = config_manager.get("audio_transcription.enable_speaker_detection", True)
        
        # API credentials - secure from environment variables
        self.google_api_key = config_manager.get_secure("audio_transcription.google_api_key")
        self.azure_api_key = config_manager.get_secure("audio_transcription.azure_api_key")
        self.azure_region = config_manager.get_secure("audio_transcription.azure_region", default="eastus")
        self.openai_api_key = config_manager.get_secure("report_generation.openai_api_key")
        
        # Supported models
        self.supported_models = ["whisper", "google", "azure", "local"]
        
        # Pre-load Whisper model if using whisper
        self.whisper_model = None
        if self.model_type == "whisper":
            try:
                import whisper
                self.logger.info("Loading Whisper model (this may take a moment)...")
                self.whisper_model = whisper.load_model("base")
                self.logger.info("Whisper model loaded successfully")
            except ImportError:
                self.logger.warning("Whisper not available, will use API fallback")
            except Exception as e:
                self.logger.warning(f"Failed to load Whisper model: {e}")
        
    def transcribe_video(self, video_path: str) -> Dict[str, Any]:
        """
        Main transcription function
        Returns comprehensive transcript with timestamps and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting audio transcription using {self.model_type}")
        
        try:
            # Extract audio from video
            self.logger.log_processing_step("Extracting audio from video")
            audio_path = self._extract_audio(video_path)
            
            # Get audio information
            audio_info = self._get_audio_info(audio_path)
            self.logger.info(f"Audio info: {audio_info}")
            
            # Split audio into chunks for processing
            self.logger.log_processing_step("Splitting audio into chunks")
            audio_chunks = self._split_audio(audio_path, self.chunk_duration)
            
            # Transcribe each chunk
            self.logger.log_processing_step("Transcribing audio chunks")
            transcripts = []
            
            for i, chunk_path in enumerate(audio_chunks):
                chunk_start_time = i * self.chunk_duration
                
                try:
                    chunk_transcript = self._transcribe_chunk(chunk_path, chunk_start_time)
                    if chunk_transcript:
                        transcripts.extend(chunk_transcript)
                        
                    progress = ((i + 1) / len(audio_chunks)) * 30 + 50  # 50-80% progress
                    self.logger.log_processing_step(f"Transcribed chunk {i+1}/{len(audio_chunks)}", progress)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to transcribe chunk {i+1}: {e}")
                    
            # Post-process transcripts
            self.logger.log_processing_step("Post-processing transcripts")
            final_transcript = self._post_process_transcripts(transcripts, audio_info)
            
            # Clean up temporary files
            self._cleanup_temp_files([audio_path] + audio_chunks)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Transcription completed in {processing_time:.2f}s")
            
            return final_transcript
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            raise
            
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio track from video using FFmpeg"""
        try:
            # Create temporary audio file
            temp_dir = self.config_manager.get_temp_dir()
            audio_filename = f"audio_{int(time.time())}.wav"
            audio_path = temp_dir / audio_filename
            
            # FFmpeg command to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # Disable video
                '-acodec', 'pcm_s16le',  # Use PCM 16-bit little-endian
                '-ac', '1',  # Mono channel
                '-ar', '16000',  # 16kHz sample rate (good for speech recognition)
                '-y',  # Overwrite output file
                str(audio_path)
            ]
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
                
            if not audio_path.exists():
                raise Exception("Audio extraction failed - output file not created")
                
            self.logger.info(f"Audio extracted to: {audio_path}")
            return str(audio_path)
            
        except subprocess.TimeoutExpired:
            raise Exception("Audio extraction timed out")
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in PATH")
        except Exception as e:
            raise Exception(f"Audio extraction failed: {str(e)}")
            
    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get audio file information"""
        try:
            with wave.open(audio_path, 'rb') as audio_file:
                frames = audio_file.getnframes()
                sample_rate = audio_file.getframerate()
                duration = frames / sample_rate
                channels = audio_file.getnchannels()
                sample_width = audio_file.getsampwidth()
                
            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "total_frames": frames,
                "file_size": Path(audio_path).stat().st_size
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get audio info: {e}")
            return {"duration": 0, "sample_rate": 16000, "channels": 1}
            
    def _split_audio(self, audio_path: str, chunk_duration: int) -> List[str]:
        """Split audio into smaller chunks for processing"""
        try:
            audio_info = self._get_audio_info(audio_path)
            total_duration = audio_info["duration"]
            
            # Process videos under 10 minutes as a single chunk
            if total_duration <= 600:  # 10 minutes
                self.logger.info(f"Audio duration ({total_duration:.1f}s) is under 10 minutes, processing as single chunk")
                return [audio_path]
                
            chunks = []
            temp_dir = self.config_manager.get_temp_dir()
            
            num_chunks = int(total_duration / chunk_duration) + 1
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                
                if start_time >= total_duration:
                    break
                    
                chunk_filename = f"chunk_{i:03d}_{int(time.time())}.wav"
                chunk_path = temp_dir / chunk_filename
                
                # Use FFmpeg to extract chunk
                cmd = [
                    'ffmpeg',
                    '-i', audio_path,
                    '-ss', str(start_time),
                    '-t', str(chunk_duration),
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    str(chunk_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and chunk_path.exists():
                    chunks.append(str(chunk_path))
                else:
                    self.logger.warning(f"Failed to create chunk {i}: {result.stderr}")
                    
            self.logger.info(f"Split audio into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.warning(f"Audio splitting failed: {e}")
            return [audio_path]
            
    def _transcribe_chunk(self, chunk_path: str, start_time_offset: float) -> List[Dict[str, Any]]:
        """Transcribe a single audio chunk"""
        if self.model_type == "whisper":
            return self._transcribe_with_whisper(chunk_path, start_time_offset)
        elif self.model_type == "google":
            return self._transcribe_with_google(chunk_path, start_time_offset)
        elif self.model_type == "azure":
            return self._transcribe_with_azure(chunk_path, start_time_offset)
        elif self.model_type == "local":
            return self._transcribe_with_local(chunk_path, start_time_offset)
        else:
            raise ValueError(f"Unsupported transcription model: {self.model_type}")
            
    def _transcribe_with_whisper(self, audio_path: str, start_time_offset: float) -> List[Dict[str, Any]]:
        """Transcribe using OpenAI Whisper (local or API)"""
        try:
            if self.whisper_model is not None:
                result = self.whisper_model.transcribe(
                    audio_path,
                    language=self.language if self.language != "auto" else None,
                    word_timestamps=True,
                    verbose=False
                )
                
                segments = []
                for segment in result["segments"]:
                    segment_start = segment["start"] + start_time_offset
                    segment_end = segment["end"] + start_time_offset
                    
                    segment_data = {
                        "text": segment["text"].strip(),
                        "start": segment_start,
                        "end": segment_end,
                        "start_formatted": self._format_timestamp(segment_start),
                        "end_formatted": self._format_timestamp(segment_end),
                        "confidence": 0.9,
                        "words": []
                    }
                    
                    if "words" in segment:
                        for word_info in segment["words"]:
                            word_data = {
                                "word": word_info["word"].strip(),
                                "start": word_info["start"] + start_time_offset,
                                "end": word_info["end"] + start_time_offset,
                                "confidence": 0.9
                            }
                            segment_data["words"].append(word_data)
                            
                    segments.append(segment_data)
                    
                return segments
            else:
                return self._transcribe_with_openai_api(audio_path, start_time_offset)
                
        except Exception as e:
            self.logger.warning(f"Whisper transcription failed: {e}")
            return []
            
    def _transcribe_with_openai_api(self, audio_path: str, start_time_offset: float) -> List[Dict[str, Any]]:
        """Transcribe using OpenAI Whisper API"""
        try:
            if not self.openai_api_key or "YOUR_" in self.openai_api_key:
                raise Exception("OpenAI API key not configured")
                
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_api_key)
            
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=self.language if self.language != "auto" else None,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
                
            segments = []
            for segment in response.segments:
                segment_start = segment["start"] + start_time_offset
                segment_end = segment["end"] + start_time_offset
                
                segment_data = {
                    "text": segment["text"].strip(),
                    "start": segment_start,
                    "end": segment_end,
                    "start_formatted": self._format_timestamp(segment_start),
                    "end_formatted": self._format_timestamp(segment_end),
                    "confidence": 0.9,
                    "words": []
                }
                segments.append(segment_data)
                
            return segments
            
        except ImportError:
            self.logger.warning("OpenAI library not available")
            return []
        except Exception as e:
            self.logger.warning(f"OpenAI API transcription failed: {e}")
            return []
            
    def _transcribe_with_google(self, audio_path: str, start_time_offset: float) -> List[Dict[str, Any]]:
        """Transcribe using Google Cloud Speech-to-Text"""
        try:
            if not self.google_api_key or "YOUR_" in self.google_api_key:
                raise Exception("Google API key not configured")
                
            from google.cloud import speech
            import io
            
            client = speech.SpeechClient()
            
            with io.open(audio_path, "rb") as audio_file:
                content = audio_file.read()
                
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.language,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            )
            
            response = client.recognize(config=config, audio=audio)
            
            segments = []
            for result in response.results:
                if result.alternatives:
                    alternative = result.alternatives[0]
                    
                    words_data = []
                    segment_start = None
                    segment_end = None
                    
                    for word_info in alternative.words:
                        word_start = word_info.start_time.total_seconds() + start_time_offset
                        word_end = word_info.end_time.total_seconds() + start_time_offset
                        
                        if segment_start is None:
                            segment_start = word_start
                        segment_end = word_end
                        
                        words_data.append({
                            "word": word_info.word,
                            "start": word_start,
                            "end": word_end,
                            "confidence": alternative.confidence
                        })
                        
                    if segment_start is not None:
                        segment_data = {
                            "text": alternative.transcript.strip(),
                            "start": segment_start,
                            "end": segment_end,
                            "start_formatted": self._format_timestamp(segment_start),
                            "end_formatted": self._format_timestamp(segment_end),
                            "confidence": alternative.confidence,
                            "words": words_data
                        }
                        segments.append(segment_data)
                        
            return segments
            
        except ImportError:
            self.logger.warning("Google Cloud Speech library not installed")
            return []
        except Exception as e:
            self.logger.warning(f"Google transcription failed: {e}")
            return []
            
    def _transcribe_with_azure(self, audio_path: str, start_time_offset: float) -> List[Dict[str, Any]]:
        """Transcribe using Azure Speech Services"""
        try:
            if not self.azure_api_key or "YOUR_" in self.azure_api_key:
                raise Exception("Azure API key not configured")
                
            import azure.cognitiveservices.speech as speechsdk
            
            speech_config = speechsdk.SpeechConfig(
                subscription=self.azure_api_key,
                region=self.azure_region
            )
            speech_config.speech_recognition_language = self.language
            
            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
            
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            result = speech_recognizer.recognize_once()
            
            segments = []
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                audio_info = self._get_audio_info(audio_path)
                segment_data = {
                    "text": result.text.strip(),
                    "start": start_time_offset,
                    "end": start_time_offset + audio_info["duration"],
                    "start_formatted": self._format_timestamp(start_time_offset),
                    "end_formatted": self._format_timestamp(start_time_offset + audio_info["duration"]),
                    "confidence": 0.8,
                    "words": []
                }
                segments.append(segment_data)
                
            return segments
            
        except ImportError:
            self.logger.warning("Azure Speech Services library not installed")
            return []
        except Exception as e:
            self.logger.warning(f"Azure transcription failed: {e}")
            return []
            
    def _transcribe_with_local(self, audio_path: str, start_time_offset: float) -> List[Dict[str, Any]]:
        """Transcribe using local/fallback method"""
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                
            text = recognizer.recognize_google(audio_data, language=self.language)
            
            audio_info = self._get_audio_info(audio_path)
            segment_data = {
                "text": text.strip(),
                "start": start_time_offset,
                "end": start_time_offset + audio_info["duration"],
                "start_formatted": self._format_timestamp(start_time_offset),
                "end_formatted": self._format_timestamp(start_time_offset + audio_info["duration"]),
                "confidence": 0.7,
                "words": []
            }
            
            return [segment_data]
            
        except ImportError:
            self.logger.warning("speech_recognition library not installed")
            return []
        except Exception as e:
            self.logger.warning(f"Local transcription failed: {e}")
            return []
            
    def _post_process_transcripts(self, transcripts: List[Dict[str, Any]], 
                                audio_info: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process and combine transcript segments"""
        
        transcripts.sort(key=lambda x: x["start"])
        
        merged_segments = self._merge_segments(transcripts)
        
        full_text = " ".join(segment["text"] for segment in merged_segments)
        
        total_duration = audio_info.get("duration", 0)
        total_words = len(full_text.split())
        avg_confidence = sum(s["confidence"] for s in merged_segments) / len(merged_segments) if merged_segments else 0
        
        summary = self._generate_transcript_summary(merged_segments, full_text)
        
        return {
            "segments": merged_segments,
            "full_text": full_text,
            "statistics": {
                "total_duration": total_duration,
                "total_segments": len(merged_segments),
                "total_words": total_words,
                "average_confidence": avg_confidence,
                "words_per_minute": (total_words / total_duration * 60) if total_duration > 0 else 0
            },
            "summary": summary,
            "audio_info": audio_info,
            "processing_info": {
                "model_used": self.model_type,
                "language": self.language,
                "chunk_duration": self.chunk_duration,
                "timestamp": time.time()
            }
        }
        
    def _merge_segments(self, segments: List[Dict[str, Any]], 
                       gap_threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Merge segments that are close together"""
        if not segments:
            return []
            
        merged = []
        current = segments[0].copy()
        
        for next_segment in segments[1:]:
            gap = next_segment["start"] - current["end"]
            
            if gap <= gap_threshold:
                current["text"] += " " + next_segment["text"]
                current["end"] = next_segment["end"]
                current["end_formatted"] = next_segment["end_formatted"]
                
                if "words" in current and "words" in next_segment:
                    current["words"].extend(next_segment["words"])
                    
                current["confidence"] = (current["confidence"] + next_segment["confidence"]) / 2
            else:
                merged.append(current)
                current = next_segment.copy()
                
        merged.append(current)
        
        return merged
        
    def _generate_transcript_summary(self, segments: List[Dict[str, Any]], 
                                   full_text: str) -> Dict[str, Any]:
        """Generate summary of transcript content"""
        
        word_count = len(full_text.split())
        sentence_count = len([s for s in full_text.split('.') if s.strip()])
        
        low_confidence_segments = [
            s for s in segments 
            if s["confidence"] < self.confidence_threshold
        ]
        
        words = full_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "low_confidence_segments": len(low_confidence_segments),
            "top_keywords": [word for word, count in top_keywords],
            "estimated_speakers": 1,
            "quality_indicators": {
                "avg_confidence": sum(s["confidence"] for s in segments) / len(segments) if segments else 0,
                "confidence_variance": self._calculate_confidence_variance(segments),
                "speech_rate_wpm": (word_count / (segments[-1]["end"] if segments else 1)) * 60
            }
        }
        
    def _calculate_confidence_variance(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate variance in confidence scores"""
        if not segments:
            return 0
            
        confidences = [s["confidence"] for s in segments]
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        return variance
        
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds_part = total_seconds % 60
        milliseconds = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"
        
    def _cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary audio files"""
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(f"Could not delete temp file {file_path}: {e}")