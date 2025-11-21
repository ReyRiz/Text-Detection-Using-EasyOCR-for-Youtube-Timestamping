"""
detect_gui_multithread.py

Flow:
1) Pilih video, atur interval skip (mis: 30)
2) Ekstrak semua frame tiap skip -> simpan ke folder temp (nama: frame_{index}_{HH-MM-SS}.jpg)
3) Setelah ekstraksi selesai -> lakukan OCR multithread (EasyOCR) pada semua gambar
   - tiap worker thread akan membuat reader sendiri (thread-local)
   - progres ditampilkan di terminal (tqdm) dan GUI (ttk.Progressbar + label)
4) Hasil (timestamps) dikondensasi menjadi rentang dan disimpan ke file hasil_deteksi.txt
5) Opsi: hapus folder frame setelah selesai

Requirements:
- pip install easyocr opencv-python tqdm
- GPU: set opsi "Use GPU" jika environment mendukung (CUDA + torch)
"""

import os
import cv2
import shutil
import math
import threading
import queue
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import easyocr
import numpy as np
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', message='.*RNN module.*')

# ---------------------------
# Utilities
# ---------------------------
def seconds_to_hhmmss(s):
    h = int(s) // 3600
    m = (int(s) % 3600) // 60
    sec = int(s) % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def hhmmss_to_seconds(hhmmss):
    # accepts "HH:MM:SS"
    parts = hhmmss.split(":")
    if len(parts) != 3:
        return 0
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def condense_seconds_to_ranges(int_seconds, max_gap=1):
    """Given sorted list of unique integer seconds, condense to ranges using max_gap."""
    if not int_seconds:
        return []
    s = sorted(set(int_seconds))
    ranges = []
    start = prev = s[0]
    for x in s[1:]:
        if x <= prev + max_gap:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    return ranges

def select_roi(video_path):
    """Open video and let user select ROI for OCR detection.
    Returns (x, y, w, h) tuple or None if cancelled."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Resize frame for display if too large
    display_frame = frame.copy()
    height, width = display_frame.shape[:2]
    max_display_height = 720
    
    scale = 1.0
    if height > max_display_height:
        scale = max_display_height / height
        new_width = int(width * scale)
        display_frame = cv2.resize(display_frame, (new_width, max_display_height))
    
    # Instructions
    instructions = "Pilih area untuk OCR (drag mouse untuk membuat kotak)"
    instructions2 = "Tekan ENTER untuk konfirmasi, ESC untuk batal, R untuk reset"
    cv2.putText(display_frame, instructions, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_frame, instructions2, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Use cv2.selectROI for interactive selection
    window_name = "Pilih Area OCR - Drag untuk membuat kotak, ENTER konfirmasi, ESC batal"
    roi = cv2.selectROI(window_name, display_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    # roi is (x, y, w, h) in display coordinates
    x, y, w, h = roi
    
    # If cancelled (w or h is 0)
    if w == 0 or h == 0:
        return None
    
    # Scale back to original frame coordinates
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)
    
    return (x, y, w, h)

# ---------------------------
# Frame extraction + immediate OCR with GPU optimization
# ---------------------------
def extract_and_detect_frames(video_path, target_text, skip=30, jpg_quality=70, 
                               use_gpu=False, min_confidence=0.4, 
                               batch_size=8, progress_callback=None, log_callback=None, roi=None):
    """
    Extract frames every `skip` frames and run OCR in batches for GPU optimization.
    progress_callback(processed_frames, total_frames, detected_count) is optional for GUI updates.
    log_callback(message) is optional for logging progress messages.
    roi: (x, y, w, h) tuple to crop frame before OCR, or None for full frame
    Returns list of detected seconds
    """
    # Initialize EasyOCR reader once with GPU optimization
    if log_callback:
        log_callback("Menginisialisasi EasyOCR reader dengan GPU optimization...")
    
    # Set CUDA optimization if GPU is enabled
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                # Aggressive CUDA optimization untuk RTX 3050
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.deterministic = False
                torch.set_num_threads(1)  # Minimize CPU overhead
                
                # Set GPU memory allocation strategy
                torch.cuda.empty_cache()
                
                # Enable TF32 for Ampere GPUs (RTX 30 series)
                if hasattr(torch.cuda, 'allow_tf32_cublas'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                if log_callback:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    log_callback(f"GPU terdeteksi: {gpu_name} ({gpu_memory:.1f} GB)")
                    log_callback(f"Batch size: {batch_size} frames (parallel processing)")
                    log_callback(f"CUDA optimization: ACTIVE")
        except ImportError:
            if log_callback:
                log_callback("PyTorch tidak terdeteksi, GPU optimization terbatas")
    
    reader = easyocr.Reader(["id", "en"], gpu=use_gpu, verbose=False, 
                           quantize=False, model_storage_directory=None)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Gagal membuka video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration_sec = total_frames / fps
    total_frames_to_process = total_frames // skip
    
    if log_callback:
        log_callback(f"Video FPS: {fps:.2f}, Total frames: {total_frames}")
        log_callback(f"Durasi video: {seconds_to_hhmmss(video_duration_sec)}")
        log_callback(f"Akan memproses {total_frames_to_process} frame (setiap {skip} frame)")
        if roi:
            log_callback(f"ROI Area: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    
    frame_idx = 0
    processed = 0
    detected_seconds = []
    detected_timestamps = []  # Store formatted timestamps untuk display
    start_time = time.time()
    last_log_time = start_time
    last_display_update = start_time
    current_position = "00:00:00"
    
    # Batch processing untuk GPU optimization
    frame_batch = []
    timestamp_batch = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining batch
            if frame_batch:
                process_batch_optimized(reader, frame_batch, timestamp_batch, target_text, 
                                       min_confidence, detected_seconds, detected_timestamps, log_callback, use_gpu)
                processed += len(frame_batch)
            break
            
        if frame_idx % skip == 0:
            # Dapatkan timestamp aktual dari video (dalam milliseconds)
            ts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts_seconds = ts_msec / 1000.0  # Convert ke seconds
            
            # Crop frame to ROI if specified
            processed_frame = frame
            if roi:
                x, y, w, h = roi
                processed_frame = frame[y:y+h, x:x+w]
            
            # Add to batch - hindari copy jika memungkinkan
            frame_batch.append(processed_frame if use_gpu else processed_frame.copy())
            timestamp_batch.append(ts_seconds)
            
            # Process batch when full
            if len(frame_batch) >= batch_size:
                process_batch_optimized(reader, frame_batch, timestamp_batch, target_text,
                                       min_confidence, detected_seconds, detected_timestamps, log_callback, use_gpu)
                
                processed += len(frame_batch)
                frame_batch = []
                timestamp_batch = []
                
                # Update progress with ETA
                if progress_callback:
                    try:
                        progress_callback(frame_idx, total_frames, len(detected_seconds))
                    except Exception:
                        pass
                
                # Log progress setiap 30 detik
                current_time = time.time()
                if log_callback and (current_time - last_log_time) >= 30:
                    log_callback(f"Progress: {processed}/{total_frames_to_process} frames")
                    last_log_time = current_time
        
        frame_idx += 1
    
    cap.release()
    
    # Final progress callback
    if progress_callback:
        try:
            progress_callback(frame_idx, total_frames, len(detected_seconds))
        except Exception:
            pass
    
    if log_callback:
        total_elapsed = time.time() - start_time
        elapsed_str = seconds_to_hhmmss(total_elapsed)
        log_callback(f"\n{'='*50}")
        log_callback(f"Selesai! Total waktu: {elapsed_str}")
        log_callback(f"Total deteksi: {len(detected_seconds)}")
    
    return detected_seconds

def process_batch_optimized(reader, frame_batch, timestamp_batch, target_text, 
                           min_confidence, detected_seconds, detected_timestamps, log_callback, use_gpu):
    """Optimized batch processing - process multiple frames efficiently"""
    try:
        # Pre-compile target text untuk speed
        target_lower = target_text.lower()
        
        # Process semua frames dalam batch
        for i, frame in enumerate(frame_batch):
            ts_seconds = timestamp_batch[i]
            ts_str = seconds_to_hhmmss(ts_seconds)
            
            # Resize frame untuk OCR lebih cepat (optimal: 720p untuk speed)
            height, width = frame.shape[:2]
            if height > 720:
                scale = 720 / height
                new_width = int(width * scale)
                frame = cv2.resize(frame, (new_width, 720), interpolation=cv2.INTER_AREA)
            
            try:
                # OCR dengan optimasi agresif untuk kecepatan maksimal
                ocr_out = reader.readtext(
                    frame,
                    paragraph=False,      # Lebih cepat untuk single line text
                    min_size=15,          # Skip text kecil (naik dari 10)
                    text_threshold=0.5,   # Threshold lebih tinggi = lebih cepat
                    low_text=0.4,         # Naikkan untuk skip kandidat lemah
                    link_threshold=0.4,   # Lebih tinggi = lebih cepat
                    canvas_size=1920,     # Turunkan dari 2560 untuk speed
                    mag_ratio=1.0,        # Magnification ratio
                    width_ths=0.7,        # Width threshold untuk text grouping
                    add_margin=0.05,      # Margin lebih kecil = lebih cepat
                    contrast_ths=0.1,     # Contrast threshold
                    adjust_contrast=0.5   # Reduce contrast adjustment overhead
                )
                
                # Quick scan untuk target text
                found = False
                for bbox, text, conf in ocr_out:
                    if conf and conf >= min_confidence and target_lower in text.lower():
                        detected_seconds.append(int(ts_seconds))
                        detected_timestamps.append(ts_str)
                        found = True
                        if log_callback:
                            log_callback(f"Terdeteksi : {ts_str}")
                        break
                
                # Jika tidak ditemukan, log juga
                if not found and log_callback:
                    log_callback(f"Belum Terdeteksi : {ts_str}")
                        
            except Exception as e:
                if log_callback:
                    log_callback(f"Error pada {ts_str}: {e}")
                    
    except Exception as e:
        if log_callback:
            log_callback(f"Error batch processing: {e}")

# ---------------------------
# Save condensed ranges to file
# ---------------------------
def save_condensed_ranges(seconds_list, out_file="hasil_deteksi.txt"):
    # filter None and unique ints
    secs = sorted({int(s) for s in seconds_list if s is not None})
    ranges = condense_seconds_to_ranges(secs, max_gap=1)

    with open(out_file, "w", encoding="utf-8") as f:
        if not ranges:
            f.write("Teks tidak ditemukan.\n")
            return
        for a, b in ranges:
            if a == b:
                f.write(f"{seconds_to_hhmmss(a)}\n")
            else:
                f.write(f"{seconds_to_hhmmss(a)}-{seconds_to_hhmmss(b)}\n")
    return ranges

# ---------------------------
# GUI (Tkinter)
# ---------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Real-time Timestamp Detection")

        self.video_path = tk.StringVar(value="")
        self.skip_var = tk.IntVar(value=30)
        self.target_var = tk.StringVar(value="baru saja memberikan mental bajanya")
        self.use_gpu_var = tk.BooleanVar(value=True)  # Default GPU ON untuk RTX 3050
        self.min_conf_var = tk.DoubleVar(value=0.35)
        self.batch_size_var = tk.IntVar(value=12)  # Batch size lebih besar untuk speed
        self.output_path = tk.StringVar(value="hasil_deteksi.txt")
        self.roi = None  # Store ROI coordinates (x, y, w, h)
        self.roi_label_text = tk.StringVar(value="ROI: Full Frame")

        # layout
        frm = ttk.Frame(root, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(frm, text="Video:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frm, textvariable=self.video_path, width=60).grid(row=row, column=1, sticky=tk.W)
        ttk.Button(frm, text="Pilih...", command=self.browse_video).grid(row=row, column=2, padx=4)
        row += 1
        
        # ROI selection
        ttk.Label(frm, text="Area OCR:").grid(row=row, column=0, sticky=tk.W)
        ttk.Label(frm, textvariable=self.roi_label_text).grid(row=row, column=1, sticky=tk.W)
        ttk.Button(frm, text="Pilih ROI", command=self.select_roi_area).grid(row=row, column=2, padx=4)
        row += 1

        ttk.Label(frm, text="Target teks:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frm, textvariable=self.target_var, width=60).grid(row=row, column=1, columnspan=2, sticky=tk.W)
        row += 1

        ttk.Label(frm, text="Simpan hasil ke:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(frm, textvariable=self.output_path, width=60).grid(row=row, column=1, sticky=tk.W)
        ttk.Button(frm, text="Browse...", command=self.browse_output).grid(row=row, column=2, padx=4)
        row += 1

        ttk.Label(frm, text="Skip frame (N):").grid(row=row, column=0, sticky=tk.W)
        ttk.Spinbox(frm, from_=1, to=1000, textvariable=self.skip_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(frm, text="Use GPU (EasyOCR):").grid(row=row, column=0, sticky=tk.W)
        ttk.Checkbutton(frm, variable=self.use_gpu_var, onvalue=True, offvalue=False).grid(row=row, column=1, sticky=tk.W)
        ttk.Label(frm, text="Batch size (GPU):").grid(row=row, column=1, padx=(120,0))
        ttk.Spinbox(frm, from_=1, to=32, textvariable=self.batch_size_var, width=6).grid(row=row, column=1, sticky=tk.E)
        row += 1

        ttk.Label(frm, text="OCR min confidence:").grid(row=row, column=0, sticky=tk.W)
        ttk.Spinbox(frm, from_=0.0, to=1.0, increment=0.05, textvariable=self.min_conf_var, width=6).grid(row=row, column=1, sticky=tk.W)
        row += 1

        # Progress and control
        self.progress = ttk.Progressbar(frm, orient="horizontal", length=540, mode="determinate")
        self.progress.grid(row=row, column=0, columnspan=3, pady=(10,4))
        row += 1

        status_frame = ttk.Frame(frm)
        status_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Idle")
        self.status_label.pack(side=tk.LEFT)
        self.detected_label = ttk.Label(status_frame, text="Detected: 0")
        self.detected_label.pack(side=tk.LEFT, padx=16)
        row += 1

        # Detection Results text widget (Terdeteksi/Belum Terdeteksi logs)
        detection_frame = ttk.LabelFrame(frm, text="Detection Results", padding=4)
        detection_frame.grid(row=row, column=0, columnspan=3, pady=(8,0), sticky=(tk.W, tk.E, tk.N, tk.S))
        row += 1
        
        self.detection_text = tk.Text(detection_frame, height=8, width=70, wrap=tk.WORD)
        detection_scroll = ttk.Scrollbar(detection_frame, orient=tk.VERTICAL, command=self.detection_text.yview)
        self.detection_text.configure(yscrollcommand=detection_scroll.set)
        self.detection_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        detection_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        detection_frame.columnconfigure(0, weight=1)
        detection_frame.rowconfigure(0, weight=1)
        
        # Log text widget (General logs)
        log_frame = ttk.LabelFrame(frm, text="System Log", padding=4)
        log_frame.grid(row=row, column=0, columnspan=3, pady=(8,0), sticky=(tk.W, tk.E, tk.N, tk.S))
        row += 1
        
        self.log_text = tk.Text(log_frame, height=6, width=70, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=(8,0))
        self.start_btn = ttk.Button(btn_frame, text="Start Detection", command=self.start_pipeline)
        self.start_btn.pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="Quit", command=root.quit).pack(side=tk.LEFT, padx=8)

        # internal queue for background thread to update GUI
        self.gui_queue = queue.Queue()
        # poll queue periodically
        self.root.after(200, self._process_gui_queue)

        # store last result
        self.last_ranges = None

    def browse_video(self):
        p = filedialog.askopenfilename(title="Pilih file video", filetypes=[("Video files", "*.mp4;*.avi;*.mkv;*.mov"), ("All files", "*")])
        if p:
            self.video_path.set(p)
            # Auto-set output path berdasarkan nama video
            video_name = os.path.splitext(os.path.basename(p))[0]  # Nama tanpa ekstensi
            output_dir = r"E:\Kerja\Timestamps TXT"
            output_file = os.path.join(output_dir, f"{video_name}.txt")
            self.output_path.set(output_file)
    
    def browse_output(self):
        p = filedialog.asksaveasfilename(
            title="Simpan hasil deteksi",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="hasil_deteksi.txt"
        )
        if p:
            self.output_path.set(p)
    
    def select_roi_area(self):
        """Open ROI selection window"""
        video = self.video_path.get().strip()
        if not video or not os.path.exists(video):
            messagebox.showerror("Error", "Pilih file video terlebih dahulu!")
            return
        
        self.log_message("Membuka jendela pemilihan ROI...")
        roi = select_roi(video)
        
        if roi:
            self.roi = roi
            x, y, w, h = roi
            self.roi_label_text.set(f"ROI: x={x}, y={y}, w={w}, h={h}")
            self.log_message(f"ROI dipilih: x={x}, y={y}, w={w}, h={h}")
            messagebox.showinfo("ROI Selected", f"Area OCR berhasil dipilih!\n\nKoordinat:\nx={x}, y={y}\nLebar={w}, Tinggi={h}")
        else:
            self.log_message("Pemilihan ROI dibatalkan.")

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.detection_text.delete(1.0, tk.END)
    
    def log_message(self, msg):
        """Add message to log text widget"""
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()
    
    def update_detection_display(self, msg):
        """Add message to detection display (append mode)"""
        self.detection_text.insert(tk.END, msg + "\n")
        self.detection_text.see(tk.END)
        self.detection_text.update_idletasks()

    def _process_gui_queue(self):
        # read gui_queue for updates from background worker
        q = self.gui_queue
        try:
            while True:
                item = q.get_nowait()
                if not item:
                    continue
                tag = item[0]
                if tag == "progress":
                    processed, total, detected = item[1], item[2], item[3]
                    # show percent
                    pct = min(100, int((processed / total) * 100)) if total else 0
                    self.progress['value'] = pct
                    self.status_label['text'] = f"Processing: {processed}/{total}"
                    self.detected_label['text'] = f"Detected: {detected}"
                elif tag == "detection_display":
                    # Update detection display (append message)
                    content = item[1]
                    self.update_detection_display(content)
                elif tag == "detection_clear":
                    # Clear detection display
                    self.detection_text.delete(1.0, tk.END)
                elif tag == "log":
                    msg = item[1]
                    self.log_message(msg)
                elif tag == "done":
                    ranges = item[1]
                    out_file = item[2]
                    self.progress['value'] = 100
                    self.status_label['text'] = f"Done. Results saved: {out_file}"
                    self.log_message(f"\n{'='*50}")
                    self.log_message(f"SELESAI! Hasil disimpan ke: {out_file}")
                    if ranges:
                        pretty = ", ".join([f"{seconds_to_hhmmss(a)}" if a==b else f"{seconds_to_hhmmss(a)}-{seconds_to_hhmmss(b)}" for a,b in ranges])
                        self.log_message(f"Rentang waktu: {pretty}")
                        messagebox.showinfo("Selesai", f"Hasil: {pretty}\n\nFile: {out_file}")
                    else:
                        self.log_message("Teks tidak ditemukan dalam video.")
                        messagebox.showinfo("Selesai", f"Teks tidak ditemukan.\nFile: {out_file}")
                elif tag == "error":
                    msg = item[1]
                    self.log_message(f"ERROR: {msg}")
                    messagebox.showerror("Error", msg)
                # else ignore unknown tags
        except Exception:
            # queue empty or other
            pass
        finally:
            self.root.after(200, self._process_gui_queue)

    def start_pipeline(self):
        video = self.video_path.get().strip()
        if not video or not os.path.exists(video):
            messagebox.showerror("Error", "Pilih file video yang valid.")
            return

        # disable start button while running
        self.start_btn.config(state=tk.DISABLED)
        # launch background thread to run pipeline
        t = threading.Thread(target=self._pipeline_thread, daemon=True)
        t.start()

    def _pipeline_thread(self):
        video = self.video_path.get().strip()
        skip = max(1, int(self.skip_var.get()))
        target_text = self.target_var.get().strip()
        use_gpu = bool(self.use_gpu_var.get())
        min_conf = float(self.min_conf_var.get())
        batch_size = int(self.batch_size_var.get())
        out_file = self.output_path.get().strip() or "hasil_deteksi.txt"

        try:
            # Pastikan folder output ada
            output_dir = os.path.dirname(out_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            self.gui_queue.put(("detection_clear", None))
            self.gui_queue.put(("log", f"Memulai deteksi pada video: {os.path.basename(video)}"))
            self.gui_queue.put(("log", f"Target teks: '{target_text}'"))
            self.gui_queue.put(("log", f"Skip frame: {skip}, GPU: {use_gpu}, Batch size: {batch_size}, Min confidence: {min_conf}"))
            self.gui_queue.put(("log", f"Hasil akan disimpan ke: {out_file}"))
            self.gui_queue.put(("log", "="*50))
            
            # Progress callback
            def _progress_cb(processed, total, detected):
                try:
                    self.gui_queue.put(("progress", processed, total, detected))
                except Exception:
                    pass
            
            # Log callback
            def _log_cb(msg):
                try:
                    self.gui_queue.put(("log", msg))
                except Exception:
                    pass
            
            # Extract and detect simultaneously with batch processing
            secs_found = extract_and_detect_frames(
                video, target_text, skip=skip, 
                use_gpu=use_gpu, min_confidence=min_conf,
                batch_size=batch_size,
                progress_callback=_progress_cb,
                log_callback=_log_cb,
                roi=self.roi
            )

            # Condense and save
            ranges = save_condensed_ranges(secs_found, out_file=out_file)

            # send done
            final_display = "\n" + "=" * 40 + "\n"
            final_display += "✓ Deteksi Selesai!\n"
            final_display += "=" * 40
            
            self.gui_queue.put(("detection_display", final_display))
            self.gui_queue.put(("done", ranges, out_file))
        except Exception as e:
            import traceback
            error_msg = f"Pipeline error: {e}\n{traceback.format_exc()}"
            self.gui_queue.put(("detection_display", f"\n✗ Error: {e}"))
            self.gui_queue.put(("error", error_msg))
        finally:
            # re-enable start button (from GUI thread)
            self.root.after(100, lambda: self.start_btn.config(state=tk.NORMAL))


# ---------------------------
# Launcher
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
