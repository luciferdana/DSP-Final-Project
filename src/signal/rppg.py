"""
Modul untuk ekstraksi dan pemrosesan sinyal remote-photoplethysmography (rPPG).
"""

import numpy as np
import cv2
from src.signal.filters import bandpass_filter, moving_average, detrend

class RPPGSignalProcessor:
    """Kelas untuk memproses dan mengekstrak sinyal rPPG."""
    
    def __init__(self, buffer_size=300, sampling_rate=30):
        """
        Inisialisasi processor sinyal rPPG.
        
        Parameter
        ----------
        buffer_size : int, opsional
            Ukuran buffer untuk menyimpan nilai sinyal, default 300
        sampling_rate : int, opsional
            Laju sampling dalam Hz, default 30
        """
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        
        # Buffer terpisah untuk kanal RGB
        self.r_buffer = np.zeros(buffer_size)
        self.g_buffer = np.zeros(buffer_size)
        self.b_buffer = np.zeros(buffer_size)
        self.time_buffer = np.zeros(buffer_size)
        
        self.current_idx = 0
        self.start_time = None
    
    def reset(self):
        """Reset buffer sinyal."""
        self.r_buffer = np.zeros(self.buffer_size)
        self.g_buffer = np.zeros(self.buffer_size)
        self.b_buffer = np.zeros(self.buffer_size)
        self.time_buffer = np.zeros(self.buffer_size)
        self.current_idx = 0
        self.start_time = None
    
    def process_roi(self, roi, timestamp):
        """
        Proses ROI untuk mendapatkan sinyal rPPG.
        
        Parameter
        ----------
        roi : numpy.ndarray
            Region of Interest dari frame video
        timestamp : float
            Waktu pengambilan frame dalam detik
            
        Returns
        -------
        float
            Nilai sinyal yang diekstrak
        """
        if roi is None or roi.size == 0:
            return None
        
        # Inisialisasi waktu mulai jika belum ada
        if self.start_time is None:
            self.start_time = timestamp
        
        # Ekstrak nilai rata-rata RGB dari ROI
        mean_b = np.mean(roi[:, :, 0])
        mean_g = np.mean(roi[:, :, 1])
        mean_r = np.mean(roi[:, :, 2])
        
        # Simpan nilai dan waktu ke buffer
        self.r_buffer[self.current_idx] = mean_r
        self.g_buffer[self.current_idx] = mean_g
        self.b_buffer[self.current_idx] = mean_b
        self.time_buffer[self.current_idx] = timestamp - self.start_time
        
        # Perbarui indeks, reset jika mencapai akhir buffer
        self.current_idx = (self.current_idx + 1) % self.buffer_size
        
        # Nilai rPPG yang belum difilter - menggunakan kanal hijau
        # Kanal hijau sering digunakan karena memiliki sensitivitas terbaik untuk deteksi denyut jantung
        return mean_g
    
    def get_filtered_signal(self):
        """
        Dapatkan sinyal rPPG yang telah difilter.
        
        Returns
        -------
        tuple
            (time_array, signal_array) dari buffer saat ini
        """
        # Reorder buffer berdasarkan indeks terbaru
        if self.current_idx == 0:
            time_array = self.time_buffer.copy()
            r_array = self.r_buffer.copy()
            g_array = self.g_buffer.copy()
            b_array = self.b_buffer.copy()
        else:
            time_array = np.concatenate((self.time_buffer[self.current_idx:], 
                                         self.time_buffer[:self.current_idx]))
            r_array = np.concatenate((self.r_buffer[self.current_idx:], 
                                      self.r_buffer[:self.current_idx]))
            g_array = np.concatenate((self.g_buffer[self.current_idx:], 
                                      self.g_buffer[:self.current_idx]))
            b_array = np.concatenate((self.b_buffer[self.current_idx:], 
                                      self.b_buffer[:self.current_idx]))
        
        # Setidaknya butuh 3 detik data untuk proses yang berguna
        min_samples = self.sampling_rate * 3
        if len(g_array) < min_samples:
            return time_array, g_array
        
        # Filter sinyal
        # Gunakan metode rPPG yang lebih canggih (misalnya, algoritma POS - Plane-Orthogonal-to-Skin)
        # Ini adalah implementasi sederhana
        
        # Normalisasi nilai RGB
        r_n = r_array / np.mean(r_array)
        g_n = g_array / np.mean(g_array)
        b_n = b_array / np.mean(b_array)
        
        # Metode sederhana: gunakan kanal hijau setelah normalisasi
        # Metode yang lebih canggih: POS, ChromMinMax, dll.
        signal_array = g_n
        
        # Detrend sinyal (hilangkan komponen DC)
        signal_array = detrend(signal_array)
        
        # Bandpass filter untuk sinyal denyut jantung (0.7-3.5 Hz ~ 42-210 BPM)
        signal_array = bandpass_filter(signal_array, 0.7, 3.5, self.sampling_rate)
        
        # Moving average untuk memperhalus sinyal
        window_size = min(5, len(signal_array) // 10)
        if window_size > 2:
            signal_array = moving_average(signal_array, window_size)
            time_array = time_array[:len(signal_array)]  # Sesuaikan panjang waktu
        
        return time_array, signal_array
    
    def estimate_heart_rate(self):
        """
        Estimasi denyut jantung dalam BPM.
        
        Returns
        -------
        float
            Perkiraan denyut jantung dalam BPM, atau None jika data tidak cukup
        """
        # Dapatkan sinyal yang telah difilter
        _, signal = self.get_filtered_signal()
        
        # Butuh setidaknya 5 detik data untuk estimasi yang berguna
        min_samples = self.sampling_rate * 5
        if len(signal) < min_samples:
            return None
        
        # Gunakan FFT untuk mendapatkan komponen frekuensi
        n = len(signal)
        fft_data = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, 1 / self.sampling_rate)
        
        # Batasi kisaran frekuensi untuk denyut jantung (40-180 BPM ~ 0.67-3 Hz)
        valid_idx = np.where((freqs >= 0.67) & (freqs <= 3))[0]
        if len(valid_idx) == 0:
            return None
        
        valid_fft = fft_data[valid_idx]
        valid_freqs = freqs[valid_idx]
        
        # Cari frekuensi dengan amplitudo tertinggi
        max_idx = np.argmax(valid_fft)
        dominant_freq = valid_freqs[max_idx]
        
        # Konversi ke BPM
        heart_rate = dominant_freq * 60
        
        return heart_rate