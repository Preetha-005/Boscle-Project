#!/usr/bin/env python3

import argparse
import sys
import os
import platform
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.video_processor import VideoProcessor
from src.core.audio_transcriber import AudioTranscriber
from src.core.caption_generator import CaptionGenerator
from src.core.report_generator import ReportGenerator
from src.input_handlers.video_input_handler import VideoInputHandler
from src.utils.config_manager import ConfigManager
from src.utils.logger import Logger
from src.utils.validator import InputValidator
from src.utils.video_converter import ensure_compatible_format


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Meeting Video Captioning & Documentation Program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process local video file
  python cli.py --input "meeting.mp4" --output "./output"
  
  # Process YouTube video
  python cli.py --input "https://youtube.com/watch?v=..." --type youtube --output "./output"
  
  # Generate only captions (no report)
  python cli.py --input "video.mp4" --output "./output" --no-report
  
  # Generate only report (no captions)
  python cli.py --input "video.mp4" --output "./output" --no-captions
  
  # Specify transcription service
  python cli.py --input "video.mp4" --output "./output" --transcription-model whisper
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file path or URL"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory path"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["local", "youtube", "web"],
        default="local",
        help="Input type (default: local)"
    )
    
    # Processing options
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip caption generation"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    
    parser.add_argument(
        "--transcription-model",
        choices=["whisper", "google", "azure", "local"],
        help="Transcription service to use (default: from config)"
    )
    
    parser.add_argument(
        "--report-format",
        choices=["pdf", "docx", "txt"],
        help="Report format (default: from config)"
    )
    
    parser.add_argument(
        "--language",
        help="Audio language code (e.g., 'en', 'es', 'fr')"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable console output"
    )
    
    # Advanced options
    parser.add_argument(
        "--scene-threshold",
        type=float,
        help="Scene detection threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "--frame-interval",
        type=float,
        help="Frame extraction interval in seconds"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary files after processing"
    )
    
    return parser


def validate_arguments(args, validator):
    """Validate command line arguments"""
    errors = []
    
    # Validate input
    if args.type == "local":
        is_valid, error = validator.validate_local_file(args.input)
        if not is_valid:
            errors.append(f"Input file error: {error}")
    elif args.type == "youtube":
        is_valid, error = validator.validate_youtube_url(args.input)
        if not is_valid:
            errors.append(f"YouTube URL error: {error}")
    elif args.type == "web":
        is_valid, error = validator.validate_web_url(args.input)
        if not is_valid:
            errors.append(f"Web URL error: {error}")
    
    # Validate output directory
    is_valid, error = validator.validate_output_directory(args.output)
    if not is_valid:
        errors.append(f"Output directory error: {error}")
    
    # Check that at least one output is enabled
    if args.no_captions and args.no_report:
        errors.append("Cannot disable both captions and report generation")
    
    return errors


def setup_configuration(args):
    """Setup configuration based on command line arguments"""
    config_manager = ConfigManager(args.config if args.config else None)
    
    # Override configuration with command line arguments
    if args.transcription_model:
        config_manager.set("audio_transcription.model", args.transcription_model)
    
    if args.report_format:
        config_manager.set("report_generation.format", args.report_format)
    
    if args.language:
        config_manager.set("audio_transcription.language", args.language)
    
    if args.scene_threshold is not None:
        config_manager.set("video_processing.scene_detection_threshold", args.scene_threshold)
    
    if args.frame_interval is not None:
        config_manager.set("video_processing.frame_extraction_interval", args.frame_interval)
    
    # Logging level
    if args.verbose:
        config_manager.set("logging.level", "DEBUG")
    elif args.quiet:
        config_manager.set("logging.console_output", False)
    
    return config_manager


def process_video_cli(args, config_manager, logger):
    """Main video processing function for CLI"""
    try:
        # Initialize components
        video_processor = VideoProcessor(config_manager, logger)
        audio_transcriber = AudioTranscriber(config_manager, logger)
        caption_generator = CaptionGenerator(config_manager, logger)
        report_generator = ReportGenerator(config_manager, logger)
        input_handler = VideoInputHandler(config_manager, logger)
        validator = InputValidator()
        
        logger.info("Starting video processing...")
        
        # Step 1: Handle input
        logger.info(f"Processing input: {args.input} (type: {args.type})")
        
        if args.type == "local":
            video_path = args.input
        elif args.type == "youtube":
            logger.info("Downloading YouTube video...")
            video_path = input_handler.download_youtube_video(args.input, args.output)
        elif args.type == "web":
            logger.info("Downloading web video...")
            video_path = input_handler.download_web_video(args.input, args.output)
        else:
            raise ValueError(f"Unknown input type: {args.type}")
        
        # Step 1.5: Convert video to compatible format if needed (avi, webm, mov, etc.)
        logger.info("Checking video format compatibility...")
        original_video_path = video_path
        video_path = ensure_compatible_format(video_path, config_manager, logger, args.output)
        if video_path != original_video_path:
            logger.info(f"Video converted for compatibility: {Path(original_video_path).name} -> {Path(video_path).name}")
        
        # Step 2: Process video
        logger.info("Analyzing video content...")
        video_analysis = video_processor.process_video(video_path)
        
        # Step 3: Transcribe audio
        logger.info("Transcribing audio...")
        transcript = audio_transcriber.transcribe_video(video_path)
        
        output_files = []
        
        # Step 4: Generate captions (if enabled)
        if not args.no_captions:
            logger.info("Generating captioned video...")
            captioned_video_path = caption_generator.generate_captioned_video(
                video_path, transcript, args.output
            )
            output_files.append(captioned_video_path)
            
            # Also generate subtitle files
            subtitle_files = caption_generator.generate_subtitle_files_only(
                transcript, args.output
            )
            output_files.extend(subtitle_files.values())
        
        # Step 5: Generate report (if enabled)
        if not args.no_report:
            logger.info("Generating comprehensive report...")
            report_path = report_generator.generate_report(
                video_analysis, transcript, args.output
            )
            output_files.append(report_path)
        
        # Cleanup temporary files (unless disabled)
        if not args.no_cleanup:
            logger.info("Cleaning up temporary files...")
            video_processor.cleanup_temp_files()
            if args.type in ["youtube", "web"] and video_path != args.input:
                # Clean up downloaded video if it was temporary
                try:
                    Path(video_path).unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(f"Could not remove temp file: {e}")
        
        logger.info("Processing completed successfully!")
        logger.info("Output files:")
        for file_path in output_files:
            logger.info(f"  - {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return False


def main():
    """Main CLI entry point"""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup configuration
    try:
        config_manager = setup_configuration(args)
        logger = Logger("MeetingCaptioning_CLI", config_manager)
    except Exception as e:
        print(f"Configuration error: {e}")
        return 1
    
    # Validate arguments
    validator = InputValidator()
    validation_errors = validate_arguments(args, validator)
    
    if validation_errors:
        print("Validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1
    
    # Process video
    logger.info("Meeting Video Captioning Program - CLI Mode")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info("=" * 50)
    
    success = process_video_cli(args, config_manager, logger)
    
    if success:
        logger.info("All processing completed successfully!")
        return 0
    else:
        logger.error("Processing failed with errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())