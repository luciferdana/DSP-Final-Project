#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul untuk ekstraksi dan pemrosesan sinyal rPPG (remote photoplethysmography).
Algoritma dioptimasi dengan validasi untuk menangani NaN dan Inf values.
"""

import numpy as np
import cv2
from src.signal.filters import bandpass_filter, moving_average, detrend

class RPPGSignalProcessor:
    """Kelas untuk memproses dan mengekstrak sinyal rPPG dengan algoritma yang dioptimasi dan robust."""
    
    def __init__(self, buffer_size=None, sampling_rate=None):
        """
        Inisialisasi processor sinyal rPPG.
        
        Parameter
        ----------
        buffer_size : int, opsional
            Ukuran buffer untuk menyimpan nilai sinyal, ambil dari config jika None  
        sampling_rate : int, opsional
            Laju sampling dalam Hz, ambil dari config jika None
        """
        # Import konfigurasi
        from src.utils.utils import RPPG_CONFIG
        
        # Gunakan nilai konfigurasi sebagai default
        self.buffer_size = buffer_size or RPPG_CONFIG['buffer_size']
        self.sampling_rate = sampling_rate or RPPG_CONFIG['sampling_rate']
        
        # Buffer terpisah untuk kanal RGB
        self.r_buffer = np.zeros(self.buffer_size)
        self.g_buffer = np.zeros(self.buffer_size)
        self.b_buffer = np.zeros(self.buffer_size)
        self.time_buffer = np.zeros(self.buffer_size)
        
        self.current_idx = 0
        self.start_time = None
        
        # Buffer untuk estimasi heart rate yang stabil
        self.recent_hr_estimates = []
        self.max_recent_estimates = 5
    
    def reset(self):
        """Reset buffer sinyal."""
        self.r_buffer = np.zeros(self.buffer_size)
        self.g_buffer = np.zeros(self.buffer_size)
        self.b_buffer = np.zeros(self.buffer_size)
        self.time_buffer = np.zeros(self.buffer_size)
        self.current_idx = 0
        self.start_time = None
        self.recent_hr_estimates = []
    
    def _validate_rgb_values(self, r, g, b):
        """
        Validasi nilai RGB untuk menghindari nilai ekstrem.
        
        Parameter
        ----------
        r, g, b : float
            Nilai RGB
            
        Returns
        -------
        tuple
            (r, g, b) yang sudah divalidasi, atau None jika tidak valid
        """
        # Check untuk NaN atau Inf
        if not (np.isfinite(r) and np.isfinite(g) and np.isfinite(b)):
            return None
        
        # Check untuk nilai negatif atau nol yang bisa menyebabkan masalah
        if r <= 0 or g <= 0 or b <= 0:
            return None
        
        # Check untuk nilai terlalu ekstrem
        if r < 10 or g < 10 or b < 10:  # Terlalu gelap
            return None
        if r > 250 or g > 250 or b > 250:  # Terlalu terang
            return None
        
        return (r, g, b)
    
    def process_roi(self, roi, timestamp):
        """
        Proses ROI untuk mendapatkan sinyal rPPG dengan preprocessing yang lebih baik.
        
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
        
        try:
            # Preprocessing ROI untuk kualitas sinyal yang lebih baik
            # Gunakan gaussian blur untuk mengurangi noise spasial
            roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            
            # Ekstrak nilai rata-rata RGB dari ROI yang sudah di-blur
            mean_b = np.mean(roi_blurred[:, :, 0])
            mean_g = np.mean(roi_blurred[:, :, 1])
            mean_r = np.mean(roi_blurred[:, :, 2])
            
            # Validasi nilai RGB
            validated_rgb = self._validate_rgb_values(mean_r, mean_g, mean_b)
            if validated_rgb is None:
                return None
            
            mean_r, mean_g, mean_b = validated_rgb
            
            # Simpan nilai dan waktu ke buffer
            self.r_buffer[self.current_idx] = mean_r
            self.g_buffer[self.current_idx] = mean_g
            self.b_buffer[self.current_idx] = mean_b
            self.time_buffer[self.current_idx] = timestamp - self.start_time
            
            # Perbarui indeks, reset jika mencapai akhir buffer
            self.current_idx = (self.current_idx + 1) % self.buffer_size
            
            # Nilai rPPG yang belum difilter - menggunakan kanal hijau
            return mean_g
            
        except Exception as e:
            print(f"Warning: Error dalam process_roi rPPG: {e}")
            return None
    
    def get_filtered_signal(self):
        """
        Dapatkan sinyal rPPG yang telah difilter dengan algoritma yang dioptimasi.
        
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
        
        try:
            # Filter data yang valid (tidak nol)
            valid_mask = (r_array > 0) & (g_array > 0) & (b_array > 0)
            if np.sum(valid_mask) < min_samples:
                # Tidak cukup data valid, gunakan simple green channel
                return time_array, g_array
            
            # Gunakan hanya data yang valid
            r_valid = r_array[valid_mask]
            g_valid = g_array[valid_mask]
            b_valid = b_array[valid_mask]
            
            # Normalisasi RGB yang robust
            try:
                # Gunakan percentile untuk normalisasi yang lebih robust
                r_norm_val = np.percentile(r_valid, 50)  # Median
                g_norm_val = np.percentile(g_valid, 50)
                b_norm_val = np.percentile(b_valid, 50)
                
                # Pastikan nilai normalisasi tidak nol
                if r_norm_val == 0 or g_norm_val == 0 or b_norm_val == 0:
                    # Fallback ke mean
                    r_norm_val = np.mean(r_valid)
                    g_norm_val = np.mean(g_valid)
                    b_norm_val = np.mean(b_valid)
                
                if r_norm_val > 0 and g_norm_val > 0 and b_norm_val > 0:
                    r_n = r_array / r_norm_val
                    g_n = g_array / g_norm_val
                    b_n = b_array / b_norm_val
                else:
                    # Jika normalisasi gagal, gunakan raw green channel
                    signal_array = g_array
                    return time_array, signal_array
                    
            except Exception as e:
                print(f"Warning: Normalisasi RGB gagal: {e}")
                # Fallback ke raw green channel
                return time_array, g_array
            
            # Kombinasi kanal yang optimal untuk rPPG
            # Menggunakan kombinasi Green-Red yang lebih sensitif
            signal_array = g_n - 0.5 * r_n
            
            # Validasi sinyal sebelum filtering
            if not np.all(np.isfinite(signal_array)):
                print("Warning: Signal mengandung NaN/Inf sebelum filtering")
                # Clean up signal
                signal_array = np.nan_to_num(signal_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Detrend sinyal (hilangkan komponen DC)
            signal_array = detrend(signal_array)
            
            # Jika detrend menghasilkan array kosong, return original
            if len(signal_array) == 0:
                return time_array, g_array
            
            # Bandpass filter untuk sinyal denyut jantung
            signal_array = bandpass_filter(signal_array, 0.8, 3.0, self.sampling_rate)
            
            # Jika filter menghasilkan array kosong, return simple green
            if len(signal_array) == 0:
                return time_array, g_array
            
            # Moving average untuk memperhalus sinyal
            window_size = max(3, min(7, len(signal_array) // 15))
            if window_size > 2 and len(signal_array) > window_size:
                signal_array = moving_average(signal_array, window_size)
                if len(signal_array) > 0:
                    time_array = time_array[:len(signal_array)]
                else:
                    return time_array, g_array
            
            # Final validation
            if len(signal_array) > 0 and np.all(np.isfinite(signal_array)):
                return time_array, signal_array
            else:
                # Return simple green channel jika semua filtering gagal
                return time_array, g_array
                
        except Exception as e:
            print(f"Warning: Error dalam filtering rPPG: {e}")
            # Return simple green channel sebagai fallback
            return time_array, g_array
    
    def estimate_heart_rate(self):
        """
        Estimasi denyut jantung dalam BPM dengan multi-method validation.
        
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
        
        # Validasi sinyal
        if not np.all(np.isfinite(signal)):
            return None
        
        try:
            # Multi-method estimation untuk akurasi yang lebih baik
            heart_rates = []
            
            # Method 1: FFT-based (primary method)
            fft_hr = self._estimate_fft_heart_rate(signal)
            if fft_hr is not None:
                heart_rates.append(fft_hr)
            
            # Method 2: Peak detection (secondary validation)
            peak_hr = self._estimate_peak_heart_rate(signal)
            if peak_hr is not None:
                heart_rates.append(peak_hr)
            
            # Consensus dari multiple methods
            if len(heart_rates) > 0:
                # Gunakan median untuk stabilitas
                final_hr = np.median(heart_rates)
                
                # Validasi range fisiologis
                if 40 <= final_hr <= 150:
                    # Tambahkan ke buffer estimasi terbaru untuk smoothing
                    self.recent_hr_estimates.append(final_hr)
                    if len(self.recent_hr_estimates) > self.max_recent_estimates:
                        self.recent_hr_estimates.pop(0)
                    
                    # Gunakan moving average dari estimasi terbaru
                    if len(self.recent_hr_estimates) >= 3:
                        stable_hr = np.median(self.recent_hr_estimates)
                        if 40 <= stable_hr <= 150:
                            return stable_hr
                    
                    return final_hr
            
            return None
            
        except Exception as e:
            print(f"Warning: Error dalam estimasi heart rate: {e}")
            return None
    
    def _estimate_fft_heart_rate(self, signal):
        """Estimasi heart rate menggunakan FFT."""
        try:
            n = len(signal)
            fft_data = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(n, 1 / self.sampling_rate)
            
            # Range frekuensi heart rate (50-130 BPM)
            valid_idx = np.where((freqs >= 0.83) & (freqs <= 2.17))[0]
            if len(valid_idx) == 0:
                return None
            
            valid_fft = fft_data[valid_idx]
            valid_freqs = freqs[valid_idx]
            
            # Cari frekuensi dengan amplitude tertinggi
            max_idx = np.argmax(valid_fft)
            dominant_freq = valid_freqs[max_idx]
            
            return dominant_freq * 60
        except:
            return None
    
    def _estimate_peak_heart_rate(self, signal):
        """Estimasi heart rate menggunakan peak detection."""
        try:
            from scipy.signal import find_peaks
            
            # Deteksi peaks dengan parameter yang dioptimasi
            peaks, _ = find_peaks(signal, 
                                distance=self.sampling_rate//3,  # Min 200ms between peaks
                                height=np.std(signal)*0.3)       # Threshold berdasarkan std
            
            if len(peaks) >= 3:
                # Gunakan median interval untuk robustness
                peak_intervals = np.diff(peaks) / self.sampling_rate
                
                # Filter interval yang masuk akal
                valid_intervals = peak_intervals[(peak_intervals >= 0.46) & 
                                               (peak_intervals <= 1.5)]  # 40-130 BPM
                
                if len(valid_intervals) >= 2:
                    avg_interval = np.median(valid_intervals)
                    return 60 / avg_interval
            
            return None
        except:
            return None
    
    def get_signal_quality(self):
        """
        Evaluasi kualitas sinyal rPPG berdasarkan SNR dan stabilitas.
        
        Returns
        -------
        str
            Rating kualitas: 'Excellent', 'Good', 'Fair', 'Poor'
        """
        _, signal = self.get_filtered_signal()
        
        if len(signal) < self.sampling_rate * 3:
            return 'Poor'
        
        try:
            # Check jika signal valid
            if not np.all(np.isfinite(signal)):
                return 'Poor'
            
            # Hitung SNR sederhana
            signal_power = np.var(signal)
            if signal_power == 0:
                return 'Poor'
            
            noise_estimate = np.var(np.diff(signal))  # High-frequency noise estimate
            
            if noise_estimate == 0:
                return 'Excellent'
            
            snr = signal_power / noise_estimate
            
            # Evaluasi stabilitas heart rate
            hr_stability = 1.0
            if len(self.recent_hr_estimates) >= 3:
                hr_std = np.std(self.recent_hr_estimates)
                hr_stability = max(0.1, 1.0 - hr_std / 20.0)
            
            # Kombinasi SNR dan stabilitas
            quality_score = snr * hr_stability
            
            if quality_score > 8:
                return 'Excellent'
            elif quality_score > 4:
                return 'Good'
            elif quality_score > 2:
                return 'Fair'
            else:
                return 'Poor'
                
        except:
            return 'Poor'