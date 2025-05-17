import numpy as np
import cv2
from src.signal.filters import bandpass_filter, moving_average, detrend

class RespirationSignalProcessor:
    """Kelas untuk memproses dan mengekstrak sinyal respirasi."""
    
    def __init__(self, buffer_size=100, sampling_rate=30):
        """
        Inisialisasi processor sinyal respirasi.
        
        Parameter
        ----------
        buffer_size : int, opsional
            Ukuran buffer untuk menyimpan nilai sinyal, default 100
        sampling_rate : int, opsional
            Laju sampling dalam Hz, default 30
        """
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        self.signal_buffer = np.zeros(buffer_size)
        self.time_buffer = np.zeros(buffer_size)
        self.current_idx = 0
        self.start_time = None
    
    def reset(self):
        """Reset buffer sinyal."""
        self.signal_buffer = np.zeros(self.buffer_size)
        self.time_buffer = np.zeros(self.buffer_size)
        self.current_idx = 0
        self.start_time = None
    
    def process_roi(self, roi, timestamp):
        """
        Proses ROI untuk mendapatkan sinyal respirasi.
        
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
        
        # Ekstrak nilai dari ROI (misalnya, rata-rata pergerakan vertikal)
        # Ini adalah contoh sederhana, dapat diganti dengan metode yang lebih canggih
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_roi)
        
        # Simpan nilai dan waktu ke buffer
        self.signal_buffer[self.current_idx] = avg_brightness
        self.time_buffer[self.current_idx] = timestamp - self.start_time
        
        # Perbarui indeks, reset jika mencapai akhir buffer
        self.current_idx = (self.current_idx + 1) % self.buffer_size
        
        return avg_brightness
    
    def get_filtered_signal(self):
        """
        Dapatkan sinyal respirasi yang telah difilter.
        
        Returns
        -------
        tuple
            (time_array, signal_array) dari buffer saat ini
        """
        # Reorder buffer berdasarkan indeks terbaru
        if self.current_idx == 0:
            time_array = self.time_buffer.copy()
            signal_array = self.signal_buffer.copy()
        else:
            time_array = np.concatenate((self.time_buffer[self.current_idx:], 
                                         self.time_buffer[:self.current_idx]))
            signal_array = np.concatenate((self.signal_buffer[self.current_idx:], 
                                           self.signal_buffer[:self.current_idx]))
        
        # Filter sinyal
        if len(signal_array) > 10:  # Pastikan cukup data untuk difilter
            # Detrend sinyal (hilangkan komponen DC)
            signal_array = detrend(signal_array)
            
            # Bandpass filter untuk sinyal respirasi (0.1-0.5 Hz)
            signal_array = bandpass_filter(signal_array, 0.1, 0.5, self.sampling_rate)
            
            # Moving average untuk memperhalus sinyal
            window_size = min(15, len(signal_array) // 3)
            if window_size > 2:
                signal_array = moving_average(signal_array, window_size)
                time_array = time_array[:len(signal_array)]  # Sesuaikan panjang waktu
        
        return time_array, signal_array
    
    def estimate_respiration_rate(self):
        """
        Estimasi laju pernapasan dalam napas per menit.
        
        Returns
        -------
        float
            Perkiraan laju pernapasan dalam napas per menit, atau None jika data tidak cukup
        """
        # Dapatkan sinyal yang telah difilter
        _, signal = self.get_filtered_signal()
        
        if len(signal) < self.sampling_rate:  # Butuh minimal 1 detik data
            return None
        
        # Hitung zero-crossings untuk estimasi frekuensi
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        
        if len(zero_crossings) < 4:  # Butuh minimal 2 siklus (4 persimpangan nol)
            return None
        
        # Perkirakan periode dari zero-crossings
        # Setiap siklus penuh memiliki 2 persimpangan nol
        cycles = len(zero_crossings) / 2
        time_span = len(signal) / self.sampling_rate  # durasi dalam detik
        
        # Hitung napas per menit
        breaths_per_second = cycles / time_span
        breaths_per_minute = breaths_per_second * 60
        
        return breaths_per_minute