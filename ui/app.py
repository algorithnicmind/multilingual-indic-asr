"""
Tkinter-based User Interface for Multilingual Indic ASR.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import sys
import sounddevice as sd
import numpy as np
import soundfile as sf
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.inference import ASRPipeline
except ImportError:
    ASRPipeline = None


class ASRApp:
    """
    GUI Application for Multilingual ASR.
    
    Features:
    - Audio file upload
    - Language detection display
    - Transcription output
    - Progress indication
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the ASR application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Multilingual Indic ASR")
        self.root.geometry("700x500")
        self.root.resizable(True, True)
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.bg_color = "#1e1e2e"
        self.fg_color = "#cdd6f4"
        self.accent_color = "#89b4fa"
        
        self.root.configure(bg=self.bg_color)
        
        # Initialize pipeline
        self.pipeline = None
        self.current_file = None
        self.is_recording = False
        self.recording_data = []
        self.stream = None
        
        # Build UI
        self._create_widgets()
        
        # Load models in background
        self._load_models_async()
    
    def _create_widgets(self):
        """Create UI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🎙️ Multilingual Indic ASR",
            font=("Helvetica", 24, "bold"),
            bg=self.bg_color,
            fg=self.accent_color
        )
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = tk.Label(
            main_frame,
            text="Speech-to-Text for English, Hindi, and Odia",
            font=("Helvetica", 12),
            bg=self.bg_color,
            fg=self.fg_color
        )
        subtitle_label.pack(pady=(0, 20))
        
        # File selection frame
        file_frame = tk.Frame(main_frame, bg=self.bg_color)
        file_frame.pack(fill=tk.X, pady=10)
        
        self.file_label = tk.Label(
            file_frame,
            text="No file selected",
            font=("Helvetica", 11),
            bg=self.bg_color,
            fg=self.fg_color,
            anchor="w"
        )
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.browse_btn = tk.Button(
            file_frame,
            text="📁 Browse",
            font=("Helvetica", 11),
            command=self._browse_file,
            bg=self.accent_color,
            fg="#1e1e2e",
            activebackground="#a6d1fa",
            cursor="hand2",
            relief=tk.FLAT,
            padx=15,
            pady=5
        )
        self.browse_btn.pack(side=tk.RIGHT)
        
        self.browse_btn.pack(side=tk.RIGHT)
        
        # Recording frame
        rec_frame = tk.Frame(main_frame, bg=self.bg_color)
        rec_frame.pack(pady=10)
        
        self.record_btn = tk.Button(
            rec_frame,
            text="🔴 Record (Hold)",
            font=("Helvetica", 12),
            command=self._toggle_recording,
            bg="#f38ba8",
            fg="#1e1e2e",
            activebackground="#eba0ac",
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=8
        )
        self.record_btn.pack()
        
        # Transcribe button
        self.transcribe_btn = tk.Button(
            main_frame,
            text="🎤 Transcribe",
            font=("Helvetica", 14, "bold"),
            command=self._transcribe,
            bg="#a6e3a1",
            fg="#1e1e2e",
            activebackground="#b4f8a8",
            cursor="hand2",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            state=tk.DISABLED
        )
        self.transcribe_btn.pack(pady=20)
        
        # Status frame
        status_frame = tk.Frame(main_frame, bg=self.bg_color)
        status_frame.pack(fill=tk.X, pady=10)
        
        # Language display
        lang_frame = tk.Frame(status_frame, bg=self.bg_color)
        lang_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            lang_frame,
            text="Language:",
            font=("Helvetica", 11, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        ).pack(side=tk.LEFT)
        
        self.lang_label = tk.Label(
            lang_frame,
            text="—",
            font=("Helvetica", 11),
            bg=self.bg_color,
            fg=self.accent_color
        )
        self.lang_label.pack(side=tk.LEFT, padx=5)
        
        # Confidence display
        conf_frame = tk.Frame(status_frame, bg=self.bg_color)
        conf_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Label(
            conf_frame,
            text="Confidence:",
            font=("Helvetica", 11, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        ).pack(side=tk.LEFT)
        
        self.conf_label = tk.Label(
            conf_frame,
            text="—",
            font=("Helvetica", 11),
            bg=self.bg_color,
            fg=self.accent_color
        )
        self.conf_label.pack(side=tk.LEFT, padx=5)
        
        # Transcription output
        tk.Label(
            main_frame,
            text="Transcription:",
            font=("Helvetica", 12, "bold"),
            bg=self.bg_color,
            fg=self.fg_color,
            anchor="w"
        ).pack(fill=tk.X, pady=(20, 5))
        
        text_frame = tk.Frame(main_frame, bg="#313244", relief=tk.SUNKEN, bd=2)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = tk.Text(
            text_frame,
            font=("Consolas", 12),
            bg="#313244",
            fg=self.fg_color,
            insertbackground=self.fg_color,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            bg=self.bg_color,
            fg="#6c7086",
            anchor="w"
        )
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def _browse_file(self):
        """Open file browser dialog."""
        filetypes = [
            ("WAV files", "*.wav"),
            ("All audio files", "*.wav *.mp3 *.flac"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if filepath:
            self.current_file = filepath
            filename = Path(filepath).name
            self.file_label.config(text=f"📄 {filename}")
            self.transcribe_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Selected: {filename}")
            
    def _toggle_recording(self):
        """Toggle recording state."""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_btn.config(text="⏹️ Stop Recording", bg="#fab387")
            self.status_var.set("Recording... Speak now")
            self.recording_data = []
            
            # Start stream
            self.stream = sd.InputStream(callback=self._audio_callback, channels=1, samplerate=16000)
            self.stream.start()
            
        else:
            # Stop recording
            self.is_recording = False
            self.record_btn.config(text="🔴 Record", bg="#f38ba8")
            self.status_var.set("Recording stopped")
            
            if self.stream:
                self.stream.stop()
                self.stream.close()
            
            # Save recorded audio
            if len(self.recording_data) > 0:
                audio_data = np.concatenate(self.recording_data, axis=0)
                output_path = "temp_recording.wav"
                sf.write(output_path, audio_data, 16000)
                
                self.current_file = output_path
                self.file_label.config(text="📄 temp_recording.wav")
                self.transcribe_btn.config(state=tk.NORMAL)
                self.status_var.set("Recording saved. Ready to transcribe.")
            else:
                messagebox.showwarning("Warning", "No audio recorded")
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream."""
        if status:
            print(status)
        self.recording_data.append(indata.copy())
    
    def _transcribe(self):
        """Start transcription in background thread."""
        if not self.current_file:
            messagebox.showwarning("No File", "Please select an audio file first.")
            return
        
        self.status_var.set("Transcribing...")
        self.transcribe_btn.config(state=tk.DISABLED)
        
        # Run in background thread
        thread = threading.Thread(target=self._do_transcribe)
        thread.daemon = True
        thread.start()
    
    def _do_transcribe(self):
        """Perform transcription (run in thread)."""
        try:
            if self.pipeline:
                result = self.pipeline.transcribe(self.current_file)
            else:
                # Demo mode without models
                result = {
                    'language': 'english',
                    'language_code': 'en',
                    'text': '[Models not loaded - demo mode]',
                    'confidence': 0.0
                }
            
            # Update UI in main thread
            self.root.after(0, lambda: self._update_result(result))
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
    
    def _update_result(self, result: dict):
        """Update UI with transcription result."""
        # Update language
        lang_display = {
            'english': '🇬🇧 English',
            'hindi': '🇮🇳 Hindi',
            'odia': '🇮🇳 Odia'
        }
        self.lang_label.config(text=lang_display.get(result['language'], result['language']))
        
        # Update confidence
        conf = result.get('confidence', 0)
        self.conf_label.config(text=f"{conf*100:.1f}%")
        
        # Update transcription
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, result.get('text', ''))
        self.output_text.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set("Transcription complete")
        self.transcribe_btn.config(state=tk.NORMAL)
    
    def _show_error(self, message: str):
        """Show error message."""
        messagebox.showerror("Error", f"Transcription failed: {message}")
        self.status_var.set("Error occurred")
        self.transcribe_btn.config(state=tk.NORMAL)
    
    def _load_models_async(self):
        """Load models in background."""
        def load():
            try:
                if ASRPipeline:
                    self.pipeline = ASRPipeline()
                    self.pipeline.load_models()
                    self.root.after(0, lambda: self.status_var.set("Models loaded"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Demo mode (no models)"))
        
        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()


def run_app():
    """Run the ASR application."""
    print("Initializing UI...", flush=True)
    try:
        root = tk.Tk()
        print("Tkinter root created.", flush=True)
        app = ASRApp(root)
        print("App initialized, starting mainloop...", flush=True)
        root.mainloop()
    except Exception as e:
        print(f"CRITICAL UI ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting app...", flush=True)
    run_app()
