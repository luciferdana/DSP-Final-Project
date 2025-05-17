"""
Modul untuk mengakses dan memproses video dari webcam.
"""

import cv2
import numpy as np

class Camera:
    """Kelas untuk mengakses webcam dan mengambil frame video."""
    
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """
        Inisialisasi objek kamera.
        
        Parameter
        ----------
        camera_id : int, opsional
            ID kamera yang diakses, default 0 (webcam utama)
        width : int, opsional
            Lebar frame yang diinginkan, default 640
        height : int, opsional
            Tinggi frame yang diinginkan, default 480
        fps : int, opsional
            Frame per detik yang diinginkan, default 30
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
    
    def start(self):
        """Memulai pengambilan video dari webcam."""
        if self.cap is not None:
            self.stop()
            
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            raise RuntimeError("Tidak dapat mengakses kamera.")
            
        self.is_running = True
        return True
    
    def read_frame(self):
        """
        Membaca satu frame dari webcam.
        
        Returns
        -------
        numpy.ndarray atau None
            Frame video dalam format BGR jika berhasil, None jika gagal
        """
        if not self.is_running or self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return frame
    
    def stop(self):
        """Menghentikan pengambilan video dari webcam."""
        if self.cap is not None:
            self.is_running = False
            self.cap.release()
            self.cap = None
            
    def __del__(self):
        """Destruktor kelas untuk memastikan kamera dilepaskan."""
        self.stop()