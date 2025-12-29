#!/usr/bin/env python3
"""
Web Server for Meeting Captioning Application
Serves the frontend and handles video processing requests
Cross-platform compatible (Windows/Linux/macOS)
"""

import os
import sys
import uuid
import platform
import subprocess
import textwrap
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.video_processor import VideoProcessor
from src.core.audio_transcriber import AudioTranscriber
from src.core.caption_generator import CaptionGenerator
from src.core.report_generator import ReportGenerator
from src.core.scene_caption_pdf_generator import SceneCaptionPDFGenerator
from src.input_handlers.video_input_handler import VideoInputHandler
from src.utils.config_manager import ConfigManager
from src.utils.logger import Logger
from src.utils.validator import InputValidator
from src.utils.video_converter import VideoConverter, ensure_compatible_format

# Platform detection
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_LINUX = SYSTEM == "Linux"
IS_MACOS = SYSTEM == "Darwin"

app = Flask(__name__, 
            static_folder='.',
            template_folder='.')
CORS(app)

# Initialize components
config_manager = ConfigManager()
logger = Logger()
validator = InputValidator()

video_processor = VideoProcessor(config_manager, logger)
audio_transcriber = AudioTranscriber(config_manager, logger)
caption_generator = CaptionGenerator(config_manager, logger)
report_generator = ReportGenerator(config_manager, logger)
scene_caption_generator = SceneCaptionPDFGenerator(config_manager, logger)
input_handler = VideoInputHandler(config_manager, logger)
video_converter = VideoConverter(config_manager, logger)

# Store processing status
processing_status = {}


def get_default_output_dir() -> Path:
    """Get cross-platform default output directory"""
    return config_manager.get_output_dir()


def get_upload_dir() -> Path:
    """Get cross-platform upload directory"""
    upload_dir = config_manager.get_output_dir() / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('.', path)


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get configuration for frontend"""
    return jsonify({
        'outputDir': str(get_default_output_dir()),
        'platform': SYSTEM,
        'pathSeparator': '\\' if IS_WINDOWS else '/'
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create uploads directory
        upload_dir = get_upload_dir()
        
        # Save file with unique name
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{file.filename}"
        filepath = upload_dir / filename
        
        file.save(str(filepath))
        
        logger.info(f"File uploaded: {filepath}")
        
        return jsonify({
            'success': True,
            'filepath': str(filepath),
            'filename': file.filename
        }), 200
        
    except Exception as e:
        logger.error(f"File upload failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_video():
    """Handle video processing request"""
    try:
        data = request.get_json()
        
        input_type = data.get('inputType', 'local')
        video_path_or_url = data.get('videoPath')
        options = data.get('options', {})
        
        if not video_path_or_url:
            return jsonify({'error': 'No video path or URL provided'}), 400
        
        # Generate processing ID
        process_id = str(uuid.uuid4())
        
        # Initialize status
        processing_status[process_id] = {
            'status': 'processing',
            'progress': 0,
            'step': 'Initializing',
            'results': None,
            'error': None
        }
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_video_background,
            args=(process_id, input_type, video_path_or_url, options),
            daemon=True
        )
        thread.start()
        
        return jsonify({'processId': process_id}), 202
        
    except Exception as e:
        logger.error(f"Error starting video processing: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<process_id>', methods=['GET'])
def get_status(process_id):
    """Get processing status"""
    if process_id not in processing_status:
        return jsonify({'error': 'Process not found'}), 404
    
    return jsonify(processing_status[process_id])


@app.route('/api/download', methods=['GET'])
def download_file():
    """Download a processed file"""
    try:
        filename = request.args.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Security: Only allow downloading from the output directory
        output_dir = get_default_output_dir()
        file_path = output_dir / filename
        
        # Resolve to prevent path traversal attacks
        file_path = file_path.resolve()
        
        # Ensure the file is within the output directory
        if not str(file_path).startswith(str(output_dir.resolve())):
            logger.warning(f"Path traversal attempt blocked: {filename}")
            return jsonify({'error': 'Invalid file path'}), 400
        
        if not file_path.exists():
            logger.error(f"File not found for download: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        logger.info(f"Serving download: {file_path}")
        
        # Send file for download
        return send_from_directory(
            str(output_dir),
            filename,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/open-file', methods=['POST'])
def open_file():
    """Open a file with the default system application (local development only)"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        
        logger.info(f"Received open file request for: {filepath}")
        
        if not filepath:
            return jsonify({'error': 'No filepath provided'}), 400
        
        file_path = Path(filepath)
        logger.info(f"Resolved path: {file_path}")
        logger.info(f"File exists: {file_path.exists()}")
        
        if not file_path.exists():
            logger.error(f"File not found at: {file_path}")
            return jsonify({'error': f'File not found: {filepath}'}), 404
        
        # Open file with default application based on OS
        logger.info(f"Opening file on {SYSTEM}")
        
        if IS_WINDOWS:
            os.startfile(str(file_path))
        elif IS_MACOS:
            subprocess.run(['open', str(file_path)], check=True)
        else:  # Linux
            subprocess.run(['xdg-open', str(file_path)], check=True)
        
        logger.info(f"Successfully opened file: {file_path}")
        return jsonify({'success': True, 'message': f'Opened {file_path.name}'}), 200
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error opening file: {e}", exc_info=True)
        return jsonify({'error': f'Failed to open file: {e}'}), 500
    except Exception as e:
        logger.error(f"Error opening file: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def create_password_protected_zip(output_dir, password, video_path, report_path, transcript_path, scene_caption_pdf_path, logger):
    """
    Create a password-protected zip file containing all output files.
    Uses pyzipper for AES-256 encryption if available, otherwise falls back to standard zip.
    After creating the zip, deletes the original unprotected files.
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"protected_meeting_files_{timestamp}.zip"
        zip_path = output_dir / zip_filename
        
        files_to_add = []
        
        # Collect all files that exist
        if video_path and Path(video_path).exists():
            files_to_add.append(('captioned_video' + Path(video_path).suffix, video_path))
        
        if report_path and Path(report_path).exists():
            files_to_add.append(('meeting_report.pdf', report_path))
        
        if transcript_path and Path(transcript_path).exists():
            files_to_add.append(('transcript.txt', transcript_path))
        
        if scene_caption_pdf_path and Path(scene_caption_pdf_path).exists():
            files_to_add.append(('scene_captions.pdf', scene_caption_pdf_path))
        
        if not files_to_add:
            logger.warning("No files to add to protected archive")
            return None
        
        # Try to use pyzipper for AES encryption (more secure)
        try:
            import pyzipper
            
            with pyzipper.AESZipFile(str(zip_path), 'w', 
                                      compression=pyzipper.ZIP_DEFLATED,
                                      encryption=pyzipper.WZ_AES) as zf:
                zf.setpassword(password.encode('utf-8'))
                
                for arcname, filepath in files_to_add:
                    zf.write(filepath, arcname)
                
                # Add a README file
                readme_content = """
=================================================================
PASSWORD PROTECTED MEETING FILES
=================================================================

This archive contains your processed meeting files:
- captioned_video: Your video with burned-in captions
- meeting_report.pdf: Comprehensive meeting analysis report
- transcript.txt: Full timestamped transcript
- scene_captions.pdf: Screenshots at scene changes with captions

IMPORTANT: These files are encrypted with AES-256 encryption.
You will need the password you set (or received) to extract them.

To extract on Windows:
  - Use 7-Zip or WinRAR (both support AES encryption)
  - Enter your password when prompted

To extract on macOS/Linux:
  - Use: unzip -P your_password protected_meeting_files.zip
  - Or use a GUI tool that supports AES encryption

=================================================================
Generated by Meeting Captioning Studio
=================================================================
""".strip()
                zf.writestr('README.txt', readme_content.encode('utf-8'))
            
            logger.info(f"Created AES-encrypted archive: {zip_path}")
            
            # Delete original unprotected files after successful zip creation
            for arcname, filepath in files_to_add:
                try:
                    Path(filepath).unlink()
                    logger.info(f"Deleted original file: {filepath}")
                except Exception as del_err:
                    logger.warning(f"Could not delete original file {filepath}: {del_err}")
            
            return str(zip_path)
            
        except ImportError:
            # Fallback to standard zipfile (less secure but works everywhere)
            logger.warning("pyzipper not available, using standard zip (less secure)")
            
            with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
                for arcname, filepath in files_to_add:
                    zf.write(filepath, arcname)
                
                # Add password info file since standard zip can't encrypt
                password_note = f"""
=================================================================
PASSWORD PROTECTED MEETING FILES
=================================================================

NOTE: Standard ZIP encryption was used.
Your password for these files is: {password}

For stronger encryption, install pyzipper:
  pip install pyzipper

=================================================================
Generated by Meeting Captioning Studio
=================================================================
""".strip()
                zf.writestr('PASSWORD_INFO.txt', password_note.encode('utf-8'))
            
            logger.info(f"Created standard archive with password info: {zip_path}")
            
            # Delete original unprotected files after successful zip creation
            for arcname, filepath in files_to_add:
                try:
                    Path(filepath).unlink()
                    logger.info(f"Deleted original file: {filepath}")
                except Exception as del_err:
                    logger.warning(f"Could not delete original file {filepath}: {del_err}")
            
            return str(zip_path)
            
    except Exception as e:
        logger.error(f"Failed to create password-protected archive: {e}", exc_info=True)
        return None


def create_regular_zip(output_dir, video_path, report_path, transcript_path, scene_caption_pdf_path, logger):
    """
    Create a regular (non-password) zip file containing all output files.
    After creating the zip, deletes the original files.
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"meeting_files_{timestamp}.zip"
        zip_path = output_dir / zip_filename
        
        files_to_add = []
        
        # Collect all files that exist
        if video_path and Path(video_path).exists():
            files_to_add.append(('captioned_video' + Path(video_path).suffix, video_path))
        
        if report_path and Path(report_path).exists():
            files_to_add.append(('meeting_report.pdf', report_path))
        
        if transcript_path and Path(transcript_path).exists():
            files_to_add.append(('transcript.txt', transcript_path))
        
        if scene_caption_pdf_path and Path(scene_caption_pdf_path).exists():
            files_to_add.append(('scene_captions.pdf', scene_caption_pdf_path))
        
        if not files_to_add:
            logger.warning("No files to add to archive")
            return None
        
        with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
            for arcname, filepath in files_to_add:
                zf.write(filepath, arcname)
            
            # Add a README file
            readme_content = """
=================================================================
MEETING CAPTIONING STUDIO - OUTPUT FILES
=================================================================

This archive contains your processed meeting files:

📹 captioned_video: Your video with burned-in captions
📄 meeting_report.pdf: Comprehensive meeting analysis report
📝 transcript.txt: Full timestamped transcript
🎬 scene_captions.pdf: Screenshots at scene changes with captions

=================================================================
Thank you for using Meeting Captioning Studio!
=================================================================
""".strip()
            zf.writestr('README.txt', readme_content.encode('utf-8'))
        
        logger.info(f"Created archive: {zip_path}")
        
        # Delete original files after successful zip creation
        for arcname, filepath in files_to_add:
            try:
                Path(filepath).unlink()
                logger.info(f"Deleted original file: {filepath}")
            except Exception as del_err:
                logger.warning(f"Could not delete original file {filepath}: {del_err}")
        
        return str(zip_path)
        
    except Exception as e:
        logger.error(f"Failed to create archive: {e}", exc_info=True)
        return None



def process_video_background(process_id, input_type, video_path_or_url, options):
    """Process video in background"""
    try:
        # Always use server default directory for web app
        # Users will download the results, not access server filesystem
        output_dir = get_default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get password protection settings
        password_protection = options.get('passwordProtection', False)
        file_password = options.get('filePassword', None)
        
        logger.info(f"Output directory: {output_dir}")
        if password_protection:
            logger.info("Password protection enabled")
        
        # Step 1: Handle input
        update_status(process_id, 10, 'Processing input')
        
        if input_type == 'local':
            is_valid, error = validator.validate_local_file(video_path_or_url)
            if not is_valid:
                raise ValueError(f'Invalid local video file: {error}')
            video_path = video_path_or_url
        elif input_type == 'youtube':
            is_valid, error = validator.validate_youtube_url(video_path_or_url)
            if not is_valid:
                raise ValueError(f'Invalid YouTube URL: {error}')
            video_path = input_handler.download_youtube_video(video_path_or_url, str(output_dir))
        elif input_type == 'url':
            is_valid, error = validator.validate_web_url(video_path_or_url)
            if not is_valid:
                raise ValueError(f'Invalid web URL: {error}')
            video_path = input_handler.download_web_video(video_path_or_url, str(output_dir))
        else:
            raise ValueError('Unknown input type')
        
        # Step 1.5: Convert video to compatible format if needed (avi, webm, mov, etc.)
        update_status(process_id, 15, 'Checking video format')
        original_video_path = video_path
        video_path = ensure_compatible_format(video_path, config_manager, logger, str(output_dir))
        if video_path != original_video_path:
            logger.info(f"Video converted for compatibility: {original_video_path} -> {video_path}")
        
        # Step 2: Extract frames and analyze
        update_status(process_id, 30, 'Extracting audio')
        video_analysis = video_processor.process_video(video_path)
        
        # Step 3: Transcribe audio
        update_status(process_id, 60, 'Transcribing')
        transcript = audio_transcriber.transcribe_video(video_path)
        
        # Step 4: Generate captions (if enabled)
        captioned_video_path = None
        if options.get('burnCaptions', True):
            update_status(process_id, 80, 'Generating captions')
            captioned_video_path = caption_generator.generate_captioned_video(
                video_path, transcript, str(output_dir))
        
        # Step 5: Generate report (if enabled)
        report_path = None
        if options.get('generateReport', True):
            update_status(process_id, 88, 'Creating report')
            report_path = report_generator.generate_report(
                video_analysis, transcript, str(output_dir))
        
        # Step 6: Generate scene caption PDF (screenshots with captions at scene changes)
        scene_caption_pdf_path = None
        if options.get('generateSceneCaptions', True):
            update_status(process_id, 92, 'Creating scene captions PDF')
            scene_caption_pdf_path = scene_caption_generator.generate_scene_caption_pdf(
                video_path, transcript, str(output_dir))
            if scene_caption_pdf_path:
                logger.info(f"Generated scene captions PDF: {scene_caption_pdf_path}")
        
        # Save transcript to text file
        transcript_path = None
        try:
            transcript_file = output_dir / 'transcript.txt'
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write("=" * 120 + "\n")
                f.write("MEETING TRANSCRIPT\n")
                f.write("=" * 120 + "\n\n")
                
                if 'full_text' in transcript:
                    f.write("FULL TEXT SUMMARY\n")
                    f.write("-" * 120 + "\n\n")
                    
                    full_text = transcript['full_text']
                    wrapped_text = textwrap.fill(full_text, width=120)
                    f.write(wrapped_text)
                    f.write("\n\n")
                
                f.write("=" * 120 + "\n")
                f.write("TIMESTAMPED TRANSCRIPT\n")
                f.write("=" * 120 + "\n\n")
                
                for segment in transcript.get('segments', []):
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    
                    if text:
                        duration = end_time - start_time
                        
                        # Split into sentences like the PDF report
                        import re
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        sentences = [s.strip() for s in sentences if s.strip()]
                        
                        if len(sentences) <= 3:
                            # Short segment - display as is
                            start_str = str(timedelta(seconds=int(start_time))).split('.')[0]
                            end_str = str(timedelta(seconds=int(end_time))).split('.')[0]
                            
                            f.write(f"[{start_str} - {end_str}]\n")
                            wrapped_segment = textwrap.fill(text, width=116, initial_indent="    ", 
                                                            subsequent_indent="    ")
                            f.write(wrapped_segment)
                            f.write("\n\n")
                        else:
                            # Long segment - split into chunks of 2-3 sentences
                            total_sentences = len(sentences)
                            sentences_per_chunk = 3
                            
                            for i in range(0, total_sentences, sentences_per_chunk):
                                chunk_sentences = sentences[i:i + sentences_per_chunk]
                                chunk_text = ' '.join(chunk_sentences)
                                
                                # Interpolate timestamp
                                chunk_start_time = start_time + (duration * i / total_sentences)
                                chunk_end_time = start_time + (duration * min(i + sentences_per_chunk, total_sentences) / total_sentences)
                                
                                start_str = str(timedelta(seconds=int(chunk_start_time))).split('.')[0]
                                end_str = str(timedelta(seconds=int(chunk_end_time))).split('.')[0]
                                
                                f.write(f"[{start_str} - {end_str}]\n")
                                wrapped_segment = textwrap.fill(chunk_text, width=116, initial_indent="    ", 
                                                                subsequent_indent="    ")
                                f.write(wrapped_segment)
                                f.write("\n\n")
            
            transcript_path = str(transcript_file)
            logger.info(f"Saved transcript to: {transcript_path}")
        except Exception as e:
            logger.error(f"Failed to save transcript file: {e}")
        
        # Always create a zip file for download
        zip_path = None
        update_status(process_id, 95, 'Preparing files for download')
        
        if password_protection and file_password:
            # Create password-protected zip
            zip_path = create_password_protected_zip(
                output_dir, 
                file_password,
                captioned_video_path,
                report_path,
                transcript_path,
                scene_caption_pdf_path,
                logger
            )
        else:
            # Create regular zip without password
            zip_path = create_regular_zip(
                output_dir,
                captioned_video_path,
                report_path,
                transcript_path,
                scene_caption_pdf_path,
                logger
            )
        
        # Clean up uploaded source file if it exists in uploads folder
        if zip_path and input_type == 'local':
            try:
                source_path = Path(video_path)
                if source_path.exists() and 'uploads' in str(source_path):
                    source_path.unlink()
                    logger.info(f"Deleted source file: {source_path}")
                    
                    # Remove uploads folder if empty
                    uploads_dir = source_path.parent
                    if uploads_dir.exists() and uploads_dir.name == 'uploads':
                        remaining_files = list(uploads_dir.iterdir())
                        if not remaining_files:
                            uploads_dir.rmdir()
                            logger.info(f"Removed empty uploads folder: {uploads_dir}")
            except Exception as cleanup_err:
                logger.warning(f"Could not clean up uploads: {cleanup_err}")
        
        # Complete - return the zip file path
        results = {}
        
        if zip_path and Path(zip_path).exists():
            results['zipFile'] = zip_path
            results['zipFilename'] = Path(zip_path).name
            results['passwordProtected'] = password_protection
        else:
            # Fallback if zip creation failed
            if captioned_video_path and Path(captioned_video_path).exists():
                results['captionedVideo'] = captioned_video_path
            if report_path and Path(report_path).exists():
                results['report'] = report_path
            if transcript_path and Path(transcript_path).exists():
                results['transcript'] = transcript_path
        
        processing_status[process_id].update({
            'status': 'completed',
            'progress': 100,
            'step': 'Complete',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        processing_status[process_id].update({
            'status': 'error',
            'error': str(e)
        })


def update_status(process_id, progress, step):
    """Update processing status"""
    processing_status[process_id].update({
        'progress': progress,
        'step': step
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200


if __name__ == '__main__':
    # Get port from environment (Railway sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    default_output = get_default_output_dir()
    
    print("=" * 70)
    print("Meeting Captioning Studio - Web Server")
    print("=" * 70)
    print(f"\n🚀 Server starting...")
    print(f"📂 Serving from: {Path(__file__).parent}")
    print(f"💻 Platform: {SYSTEM}")
    print(f"🌐 Port: {port}")
    print(f"\n📁 Default Output Directory:")
    print(f"   {default_output}")
    print(f"   (Processed videos will be saved here)")
    print(f"\n💡 Tip: You can customize the output directory in the web interface")
    print(f"\n⚡ Press Ctrl+C to stop the server\n")
    print("=" * 70)
    
    # Railway deployment configuration
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    # Fix for Windows: Use stat reloader instead of watchdog
    # This prevents the server from restarting when Whisper modifies files
    # in site-packages during audio processing
    if debug_mode and IS_WINDOWS:
        print("\n🔧 Using stat reloader on Windows (more stable with Whisper)")
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True, reloader_type='stat')
    else:
        app.run(host='0.0.0.0', port=port, debug=debug_mode)