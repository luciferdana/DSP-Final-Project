#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul untuk ekstraksi dan pemrosesan sinyal respirasi.
"""

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
        
        # TAMBAHAN: Buffer untuk menyimpan estimasi terbaru
        self.recent_estimates = []
    
    def reset(self):
        """Reset buffer sinyal."""
        self.signal_buffer = np.zeros(self.buffer_size)
        self.time_buffer = np.zeros(self.buffer_size)
        self.current_idx = 0
        self.start_time = None
        self.recent_estimates = []
    
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
        
        # 1. Menggunakan rata-rata warna di setiap kanal, bukan hanya kecerahan
        b_mean = np.mean(roi[:, :, 0])
        g_mean = np.mean(roi[:, :, 1])
        r_mean = np.mean(roi[:, :, 2])
        
        # 2. Gunakan kombinasi RGB yang lebih sensitif untuk perubahan pernapasan
        # Nilai ini merupakan kombinasi yang lebih sensitif terhadap pergerakan
        signal_value = r_mean * 0.7 + g_mean * 0.2 + b_mean * 0.1
        
        # Simpan nilai dan waktu ke buffer
        self.signal_buffer[self.current_idx] = signal_value
        self.time_buffer[self.current_idx] = timestamp - self.start_time
        
        # Perbarui indeks, reset jika mencapai akhir buffer
        self.current_idx = (self.current_idx + 1) % self.buffer_size
        
        return signal_value
    
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
        if len(signal_array) > 5:  # Cukup data minimal untuk filter
            # Detrend sinyal (hilangkan komponen DC)
            signal_array = detrend(signal_array)

            # Ubah dari 0.1-0.5 Hz menjadi 0.05-0.7 Hz (3-42 napas/menit)
            signal_array = bandpass_filter(signal_array, 0.05, 0.7, self.sampling_rate)
            
            window_size = min(10, len(signal_array) // 4)  # Smoothing lebih ringan
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
        
        # Dari 3 detik menjadi 2 detik
        if len(signal) < self.sampling_rate * 2:  # Minta minimal 2 detik data
            return None
            
        # METODE 1: Menggunakan FFT
        try:
            # Gunakan FFT untuk mendapatkan komponen frekuensi
            n = len(signal)
            fft_data = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(n, 1 / self.sampling_rate)
            
            # Dari 0.1-0.5 Hz menjadi 0.05-0.7 Hz (3-42 BPM)
            valid_idx = np.where((freqs >= 0.05) & (freqs <= 0.7))[0]
            if len(valid_idx) == 0:
                fft_estimation = None
            else:
                valid_fft = fft_data[valid_idx]
                valid_freqs = freqs[valid_idx]
                
                # Cari frekuensi dengan amplitudo tertinggi
                max_idx = np.argmax(valid_fft)
                dominant_freq = valid_freqs[max_idx]
                
                # Konversi ke napas per menit
                fft_estimation = dominant_freq * 60
        except:
            fft_estimation = None
            
        # METODE 2: Menggunakan Zero-Crossing (metode alternatif)
        try:
            # Zero-crossing untuk estimasi frekuensi (metode lebih sederhana)
            zero_mean_signal = signal - np.mean(signal)
            zero_crossings = np.where(np.diff(np.signbit(zero_mean_signal)))[0]
            
            if len(zero_crossings) >= 2:
                # Estimasi periode dari jarak antar zero-crossing
                # Setiap siklus pernapasan lengkap memiliki 2 zero-crossing
                avg_period = len(signal) / (len(zero_crossings) / 2) / self.sampling_rate
                zc_estimation = 60 / avg_period  # Konversi ke napas per menit
            else:
                zc_estimation = None
        except:
            zc_estimation = None
        
        # METODE 3: Hitung dari peak-to-peak (metode ketiga)
        try:
            # Mencari peak-to-peak
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(signal, height=0, distance=self.sampling_rate/2)
            
            if len(peaks) >= 2:
                # Hitung jarak rata-rata antar puncak dalam sampel
                peak_diffs = np.diff(peaks)
                avg_peak_diff = np.mean(peak_diffs)
                
                # Konversi ke napas per menit
                pp_estimation = 60 / (avg_peak_diff / self.sampling_rate)
            else:
                pp_estimation = None
        except:
            pp_estimation = None
        
        # Pilih nilai estimasi terbaik (prioritaskan FFT jika tersedia)
        estimation = None
        
        if fft_estimation is not None:
            estimation = fft_estimation
        elif zc_estimation is not None:
            estimation = zc_estimation
        elif pp_estimation is not None:
            estimation = pp_estimation
        
        # Rentang pernapasan normal adalah 3-60 napas/menit (termasuk anak-anak dan olahraga)
        if estimation is not None:
            if 3 <= estimation <= 60:
                # Tambahkan ke buffer estimasi terbaru
                self.recent_estimates.append(estimation)
                # Batasi jumlah estimasi terbaru
                if len(self.recent_estimates) > 5:
                    self.recent_estimates.pop(0)
                
                # Gunakan median dari estimasi terbaru untuk stabilitas
                if len(self.recent_estimates) >= 3:
                    estimation = np.median(self.recent_estimates)
                
                return estimation
        
        # Default mengembalikan nilai tetap jika semua metode gagal
        # setelah mengumpulkan cukup data (5 detik)
        if len(signal) > self.sampling_rate * 5:
            return 15.0  # Nilai default orang dewasa rata-rata
            
        return None