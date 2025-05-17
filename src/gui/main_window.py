from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import pyqtgraph as pg
from src.signal.respiration import RespirationSignalProcessor
from src.signal.rppg import RPPGSignalProcessor
from src.video.processor import detect_face, get_forehead_roi, get_chest_roi
import time

from src.video.camera import Camera

class MainWindow(QMainWindow):
    """Jendela utama aplikasi."""
    
    def __init__(self):
        """Inisialisasi jendela utama."""
        super().__init__()
        
        # Atur properti jendela
        self.setWindowTitle("Pemrosesan Sinyal Respirasi dan rPPG")
        self.setMinimumSize(1000, 600)
        
        # Inisialisasi kamera
        self.camera = Camera()
        
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
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        video_layout.addLayout(control_layout)
        
        # Panel kanan - Tampilan sinyal
        signal_group = QGroupBox("Tampilan Sinyal")
        signal_layout = QVBoxLayout(signal_group)
        
        # Placeholder untuk plot sinyal (akan diganti nanti)
        resp_label = QLabel("Placeholder Sinyal Respirasi")
        resp_label.setAlignment(Qt.AlignCenter)
        resp_label.setStyleSheet("background-color: #f0f0f0;")
        resp_label.setMinimumHeight(200)
        
        rppg_label = QLabel("Placeholder Sinyal rPPG")
        rppg_label.setAlignment(Qt.AlignCenter)
        rppg_label.setStyleSheet("background-color: #f0f0f0;")
        rppg_label.setMinimumHeight(200)
        
        signal_layout.addWidget(QLabel("Sinyal Respirasi:"))
        signal_layout.addWidget(resp_label)
        signal_layout.addWidget(QLabel("Sinyal rPPG:"))
        signal_layout.addWidget(rppg_label)
        
        # Tambahkan panel ke layout utama
        main_layout.addWidget(video_group, 1)
        main_layout.addWidget(signal_group, 1)
    
    def start_camera(self):
        """Mulai kamera dan pemrosesan video."""
        if self.camera.start():
            self.timer.start(30)  # Perbarui setiap 30ms (~33 FPS)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
    
    def stop_camera(self):
        """Hentikan kamera dan pemrosesan video."""
        self.timer.stop()
        self.camera.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        # Bersihkan tampilan video
        self.video_label.clear()
    
    def update_frame(self):
        """Perbarui frame video yang ditampilkan."""
        frame = self.camera.read_frame()
        if frame is not None:
            # Konversi frame ke format RGB (dari BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Konversi ke QImage
            h, w, ch = frame_rgb.shape
            img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            
            # Tampilkan gambar
            self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Di sini Anda akan memproses frame untuk mengekstrak sinyal
            # Untuk saat ini, kita hanya menampilkan frame mentah
    
    def closeEvent(self, event):
        """Tangani event penutupan jendela untuk membersihkan sumber daya."""
        self.stop_camera()
        event.accept()