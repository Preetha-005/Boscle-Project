#!/usr/bin/env python3

import sys
import os
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
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


class MeetingCaptioningApp:
    """Main application class with GUI interface"""
    
    # Platform detection
    SYSTEM = platform.system()
    IS_WINDOWS = SYSTEM == "Windows"
    IS_LINUX = SYSTEM == "Linux"
    IS_MACOS = SYSTEM == "Darwin"
    
    def __init__(self, root):
        self.root = root
        self.root.title("Meeting Video Captioning & Documentation Program")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.logger = Logger()
        self.validator = InputValidator()
        
        self.video_processor = VideoProcessor(self.config_manager, self.logger)
        self.audio_transcriber = AudioTranscriber(self.config_manager, self.logger)
        self.caption_generator = CaptionGenerator(self.config_manager, self.logger)
        self.report_generator = ReportGenerator(self.config_manager, self.logger)
        self.input_handler = VideoInputHandler(self.config_manager, self.logger)
        
        self.current_video_path = None
        self.processing = False
        
        self.setup_ui()
        
    def _get_default_font(self) -> str:
        """Get platform-specific default font"""
        if self.IS_WINDOWS:
            return "Arial"
        elif self.IS_MACOS:
            return "Helvetica"
        else:  # Linux
            return "DejaVu Sans"
        
    def _get_default_output_dir(self) -> str:
        """Get platform-specific default output directory"""
        return str(self.config_manager.get_output_dir())
        
    def setup_ui(self):
        """Setup the user interface"""
        # Get platform-specific font
        default_font = self._get_default_font()
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title - use platform-specific font
        title_label = ttk.Label(main_frame, text="Meeting Video Captioning & Documentation", 
                               font=(default_font, 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Video input section
        input_frame = ttk.LabelFrame(main_frame, text="Video Input", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Input type selection
        ttk.Label(input_frame, text="Input Type:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.input_type_var = tk.StringVar(value="local")
        input_type_combo = ttk.Combobox(input_frame, textvariable=self.input_type_var,
                                       values=["local", "youtube", "web_url"], state="readonly")
        input_type_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        input_type_combo.bind("<<ComboboxSelected>>", self.on_input_type_change)
        
        # File path/URL entry
        ttk.Label(input_frame, text="Video Path/URL:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.video_path_var = tk.StringVar()
        self.video_path_entry = ttk.Entry(input_frame, textvariable=self.video_path_var)
        self.video_path_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        
        self.browse_button = ttk.Button(input_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=1, column=2, pady=(10, 0))
        
        # Options section
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure(1, weight=1)
        
        # Output directory - use cross-platform default
        ttk.Label(options_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.output_dir_var = tk.StringVar(value=self._get_default_output_dir())
        output_dir_entry = ttk.Entry(options_frame, textvariable=self.output_dir_var)
        output_dir_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(options_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2)
        
        # Processing options
        self.burn_captions_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Burn captions into video", 
                       variable=self.burn_captions_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        self.generate_report_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Generate detailed report", 
                       variable=self.generate_report_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Start Processing", 
                                        command=self.start_processing, style="Accent.TButton")
        self.process_button.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Log output
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=15, state='disabled')
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        main_frame.rowconfigure(4, weight=1)
        
    def on_input_type_change(self, event=None):
        """Handle input type selection change"""
        input_type = self.input_type_var.get()
        if input_type == "local":
            self.browse_button.configure(state="normal")
            self.video_path_entry.configure(state="normal")
        else:
            self.browse_button.configure(state="disabled")
            self.video_path_entry.configure(state="normal")
            
    def browse_file(self):
        """Browse for local video file"""
        filetypes = [
            ("Video files", "*.mp4 *.mov *.avi *.mkv *.wmv *.flv *.webm"),
            ("MP4 files", "*.mp4"),
            ("MOV files", "*.mov"),
            ("AVI files", "*.avi"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if filename:
            self.video_path_var.set(filename)
            
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(
            title="Select Output Directory"
        )
        
        if directory:
            self.output_dir_var.set(directory)
            
    def log_message(self, message):
        """Add message to log output"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_progress(self, value, message=""):
        """Update progress bar and message"""
        self.progress_var.set(value)
        if message:
            self.log_message(message)
            
    def start_processing(self):
        """Start video processing in a separate thread"""
        if self.processing:
            return
            
        # Validate inputs
        video_path_or_url = self.video_path_var.get().strip()
        if not video_path_or_url:
            messagebox.showerror("Error", "Please provide a video file or URL")
            return
            
        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return
            
        # Start processing thread
        self.processing = True
        self.process_button.configure(text="Processing...", state="disabled")
        
        thread = threading.Thread(target=self.process_video, 
                                 args=(video_path_or_url, output_dir), daemon=True)
        thread.start()
        
    def process_video(self, video_path_or_url, output_dir):
        """Process video in background thread"""
        try:
            self.log_message("Starting video processing...")
            self.update_progress(0, "Initializing...")
            
            # Step 1: Handle input and get local video path
            self.update_progress(10, "Processing input...")
            input_type = self.input_type_var.get()
            
            if input_type == "local":
                is_valid, error = self.validator.validate_local_file(video_path_or_url)
                if not is_valid:
                    raise ValueError(f"Invalid local video file: {error}")
                video_path = video_path_or_url
            elif input_type == "youtube":
                is_valid, error = self.validator.validate_youtube_url(video_path_or_url)
                if not is_valid:
                    raise ValueError(f"Invalid YouTube URL: {error}")
                video_path = self.input_handler.download_youtube_video(video_path_or_url, output_dir)
            elif input_type == "web_url":
                is_valid, error = self.validator.validate_web_url(video_path_or_url)
                if not is_valid:
                    raise ValueError(f"Invalid web URL: {error}")
                video_path = self.input_handler.download_web_video(video_path_or_url, output_dir)
            else:
                raise ValueError("Unknown input type")
                
            self.current_video_path = video_path
            
            # Step 1.5: Convert video to compatible format if needed (avi, webm, mov, etc.)
            self.update_progress(15, "Checking video format compatibility...")
            original_video_path = video_path
            video_path = ensure_compatible_format(video_path, self.config_manager, self.logger, output_dir)
            if video_path != original_video_path:
                self.log_message(f"Video converted for compatibility: {Path(original_video_path).name} -> {Path(video_path).name}")
                self.current_video_path = video_path
            
            # Step 2: Extract frames and analyze video
            self.update_progress(25, "Extracting frames and analyzing video...")
            video_analysis = self.video_processor.process_video(video_path)
            
            # Step 3: Transcribe audio
            self.update_progress(50, "Transcribing audio...")
            transcript = self.audio_transcriber.transcribe_video(video_path)
            
            # Step 4: Generate captions (if enabled)
            captioned_video_path = None
            if self.burn_captions_var.get():
                self.update_progress(70, "Generating captioned video...")
                captioned_video_path = self.caption_generator.generate_captioned_video(
                    video_path, transcript, output_dir)
            
            # Step 5: Generate report (if enabled)
            report_path = None
            if self.generate_report_var.get():
                self.update_progress(85, "Generating report...")
                report_path = self.report_generator.generate_report(
                    video_analysis, transcript, output_dir)
            
            # Completion
            self.update_progress(100, "Processing completed successfully!")
            
            # Show completion message
            completion_msg = "Video processing completed!\n\n"
            if captioned_video_path:
                completion_msg += f"Captioned video: {captioned_video_path}\n"
            if report_path:
                completion_msg += f"Report: {report_path}\n"
                
            self.root.after(0, lambda: messagebox.showinfo("Success", completion_msg))
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            self.log_message(error_msg)
            self.logger.error(error_msg, exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            
        finally:
            # Reset UI state
            self.processing = False
            self.root.after(0, lambda: self.process_button.configure(
                text="Start Processing", state="normal"))


def main():
    """Main application entry point"""
    # Create and configure the root window
    root = tk.Tk()
    
    # Set up styling
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    
    # Create and run the application
    app = MeetingCaptioningApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()