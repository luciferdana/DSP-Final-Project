from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import time
import pyqtgraph as pg

from src.video.camera import Camera
from src.video.processor import detect_face, get_forehead_roi, get_chest_roi
from src.signal.respiration import RespirationSignalProcessor
from src.signal.rppg import RPPGSignalProcessor

class MainWindow(QMainWindow):
    """Jendela utama aplikasi."""
    
    def __init__(self):
        """Inisialisasi jendela utama."""
        super().__init__()
        
        # Atur properti jendela
        self.setWindowTitle("Tugas Besar Pengolahan Sinyal Digital")
        self.setMinimumSize(1000, 600)
        
        # Inisialisasi kamera
        self.camera = Camera()
        
        # Inisialisasi processor sinyal
        self.resp_processor = RespirationSignalProcessor()
        self.rppg_processor = RPPGSignalProcessor()
        
        # Timestamp awal
        self.start_time = None
        
        # Setup UI
        self.setup_ui()
        
        # Timer untuk memperbarui frame video
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def setup_ui(self):
        """Menyiapkan antarmuka pengguna."""
        # Widget utama
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout utama
        main_layout = QHBoxLayout(central_widget)
        
        # Panel kiri - Tampilan video
        video_group = QGroupBox("Tampilan Video")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        
        # Tombol kontrol
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Mulai")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton("Berhenti")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        
        # Tombol simpan data
        self.save_button = QPushButton("Simpan Data")
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.save_button)
        video_layout.addLayout(control_layout)
        
        # Panel kanan - Tampilan sinyal
        signal_group = QGroupBox("Tampilan Sinyal")
        signal_layout = QVBoxLayout(signal_group)
        
        # Plot sinyal respirasi dengan PyQtGraph
        self.resp_plot = pg.PlotWidget()
        self.resp_plot.setBackground('#f0f0f0')
        self.resp_plot.setLabel('left', 'Amplitudo')
        self.resp_plot.setLabel('bottom', 'Waktu (s)')
        self.resp_plot.setTitle('Sinyal Respirasi')
        self.resp_curve = self.resp_plot.plot(pen=pg.mkPen(color='#2E86C1', width=2))
        
        # Plot sinyal rPPG dengan PyQtGraph
        self.rppg_plot = pg.PlotWidget()
        self.rppg_plot.setBackground('#f0f0f0')
        self.rppg_plot.setLabel('left', 'Amplitudo')
        self.rppg_plot.setLabel('bottom', 'Waktu (s)')
        self.rppg_plot.setTitle('Sinyal rPPG')
        self.rppg_curve = self.rppg_plot.plot(pen=pg.mkPen(color='#C0392B', width=2))
        
        # Label untuk menampilkan laju pernapasan dan denyut jantung
        self.resp_rate_label = QLabel("Laju Pernapasan: --")
        self.heart_rate_label = QLabel("Denyut Jantung: --")
        
        # Tambahkan widgets ke layout
        signal_layout.addWidget(QLabel("Sinyal Respirasi:"))
        signal_layout.addWidget(self.resp_plot)
        signal_layout.addWidget(self.resp_rate_label)
        signal_layout.addWidget(QLabel("Sinyal rPPG:"))
        signal_layout.addWidget(self.rppg_plot)
        signal_layout.addWidget(self.heart_rate_label)
        
        # Tambahkan panel ke layout utama
        main_layout.addWidget(video_group, 1)
        main_layout.addWidget(signal_group, 1)
    
    def reset_processors(self):
        """Reset processor sinyal."""
        self.resp_processor.reset()
        self.rppg_processor.reset()
        self.start_time = None
    
    def start_camera(self):
        """Mulai kamera dan pemrosesan video."""
        if self.camera.start():
            self.reset_processors()  # Reset processor
            self.timer.start(30)  # Perbarui setiap 30ms (~33 FPS)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(True)
    
    def stop_camera(self):
        """Hentikan kamera dan pemrosesan video."""
        self.timer.stop()
        self.camera.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        # Bersihkan tampilan video
        self.video_label.clear()
    
    def update_frame(self):
        """Perbarui frame video yang ditampilkan dan proses sinyal."""
        # Inisialisasi timestamp jika belum ada
        if self.start_time is None:
            self.start_time = time.time()
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        frame = self.camera.read_frame()
        if frame is not None:
            # Deteksi wajah
            face_rect = detect_face(frame)
            
            # Salin frame untuk ditampilkan (dengan overlay ROI)
            display_frame = frame.copy()
            
            if face_rect is not None:
                x, y, w, h = face_rect
                
                # Gambar kotak di sekitar wajah
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Dapatkan ROI untuk rPPG (dahi)
                forehead_result = get_forehead_roi(face_rect, frame)
                if forehead_result is not None:
                    forehead_roi, (fx, fy, fw, fh) = forehead_result
                    
                    # Gambar kotak di sekitar ROI dahi
                    cv2.rectangle(display_frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
                    
                    # Proses ROI dahi untuk sinyal rPPG
                    self.rppg_processor.process_roi(forehead_roi, elapsed_time)
                    
                    # Dapatkan dan tampilkan sinyal rPPG
                    rppg_time, rppg_signal = self.rppg_processor.get_filtered_signal()
                    if len(rppg_signal) > 5:  # Pastikan ada cukup data
                        self.rppg_curve.setData(rppg_time, rppg_signal)
                        
                        # Estimasi denyut jantung
                        heart_rate = self.rppg_processor.estimate_heart_rate()
                        if heart_rate is not None:
                            self.heart_rate_label.setText(f"Denyut Jantung: {heart_rate:.1f} BPM")
                
                # Dapatkan ROI untuk respirasi (dada)
                chest_result = get_chest_roi(face_rect, frame)
                if chest_result is not None:
                    chest_roi, (cx, cy, cw, ch) = chest_result
                    
                    # Gambar kotak di sekitar ROI dada
                    cv2.rectangle(display_frame, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                    
                    # Proses ROI dada untuk sinyal respirasi
                    self.resp_processor.process_roi(chest_roi, elapsed_time)
                    
                    # Dapatkan dan tampilkan sinyal respirasi
                    resp_time, resp_signal = self.resp_processor.get_filtered_signal()
                    resp_time, resp_signal = self.resp_processor.get_filtered_signal()
                    if len(resp_signal) > 5:  # Pastikan ada cukup data
                        self.resp_curve.setData(resp_time, resp_signal)
                        
                        # Estimasi laju pernapasan
                        resp_rate = self.resp_processor.estimate_respiration_rate()
                        if resp_rate is not None:
                            # validasi sudah dilakukan dalam processor
                            self.resp_rate_label.setText(f"Laju Pernapasan: {resp_rate:.1f} napas/menit")
                        else:
                            # Tampilkan indikator sedang mengukur
                            if len(resp_signal) < self.resp_processor.sampling_rate * 5:
                                self.resp_rate_label.setText("Laju Pernapasan: mengukur...")
                            else:
                                self.resp_rate_label.setText("Laju Pernapasan: --")
                            
            # Konversi frame ke RGB untuk ditampilkan
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Konversi ke QImage
            h, w, ch = frame_rgb.shape
            img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            
            # Tampilkan gambar
            self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def save_data(self):
        """Simpan data sinyal ke file CSV."""
        try:
            from src.utils.helpers import save_data_to_csv, ensure_directory_exists
            import os
            from datetime import datetime
            
            # Buat direktori jika belum ada
            data_dir = "data"
            ensure_directory_exists(data_dir)
            
            # Nama file dengan timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Simpan data respirasi
            resp_time, resp_signal = self.resp_processor.get_filtered_signal()
            resp_file = os.path.join(data_dir, f"respirasi_{timestamp}.csv")
            save_data_to_csv(resp_time, resp_signal, resp_file)
            
            # Simpan data rPPG
            rppg_time, rppg_signal = self.rppg_processor.get_filtered_signal()
            rppg_file = os.path.join(data_dir, f"rppg_{timestamp}.csv")
            save_data_to_csv(rppg_time, rppg_signal, rppg_file)
            
            # Tampilkan pesan konfirmasi
            QMessageBox.information(self, "Simpan Data", 
                                  f"Data berhasil disimpan ke:\n{resp_file}\n{rppg_file}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Gagal menyimpan data: {str(e)}")
    
    def closeEvent(self, event):
        """Tangani event penutupan jendela untuk membersihkan sumber daya."""
        self.stop_camera()
        event.accept()