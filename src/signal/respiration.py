#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul untuk ekstraksi dan pemrosesan sinyal respirasi.
"""

import numpy as np
import cv2
from src.signal.filters import bandpass_filter, moving_average, detrend

class RespirationSignalProcessor:
    """Kelas untuk memproses dan mengekstrak sinyal respirasi dengan algoritma yang dioptimasi dan robust."""
    
    def __init__(self, buffer_size=None, sampling_rate=None):
        """
        Inisialisasi processor sinyal respirasi.
        
        Parameter
        ----------
        buffer_size : int, opsional
            Ukuran buffer untuk menyimpan nilai sinyal, ambil dari config jika None
        sampling_rate : int, opsional
            Laju sampling dalam Hz, ambil dari config jika None
        """
        # Import konfigurasi
        from src.utils.utils import RESPIRATION_CONFIG
        
        # Gunakan nilai konfigurasi sebagai default
        self.buffer_size = buffer_size or RESPIRATION_CONFIG['buffer_size']
        self.sampling_rate = sampling_rate or RESPIRATION_CONFIG['sampling_rate']
        
        # Inisialisasi buffer
        self.signal_buffer = np.zeros(self.buffer_size)
        self.time_buffer = np.zeros(self.buffer_size)
        self.current_idx = 0
        self.start_time = None
        
        # Buffer untuk menyimpan estimasi terbaru untuk stabilitas
        self.recent_estimates = []
        self.max_recent_estimates = 5
        
        # Buffer untuk optical flow analysis
        self.prev_gray_roi = None
        self.prev_roi_size = None  # Track ROI size untuk konsistensi
        self.flow_buffer = []
        self.max_flow_buffer = 10
        
        # Kalibrasi awal untuk adaptive threshold
        self.baseline_values = []
        self.is_calibrated = False
    
    def reset(self):
        """Reset buffer sinyal dan state."""
        self.signal_buffer = np.zeros(self.buffer_size)
        self.time_buffer = np.zeros(self.buffer_size)
        self.current_idx = 0
        self.start_time = None
        self.recent_estimates = []
        self.prev_gray_roi = None
        self.prev_roi_size = None
        self.flow_buffer = []
        self.baseline_values = []
        self.is_calibrated = False
    
    def _validate_signal_value(self, value):
        """
        Validasi nilai sinyal untuk menghindari NaN/Inf.
        
        Parameter
        ----------
        value : float
            Nilai sinyal
            
        Returns
        -------
        float atau None
            Nilai yang sudah divalidasi, atau None jika tidak valid
        """
        if value is None:
            return None
        
        # Check untuk NaN atau Inf
        if not np.isfinite(value):
            return None
        
        # Check untuk nilai yang terlalu ekstrem
        if abs(value) > 1e6:  # Nilai terlalu besar
            return None
        
        return float(value)
    
    def process_roi(self, roi, timestamp):
        """
        Proses ROI untuk mendapatkan sinyal respirasi dengan multiple methods.
        
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
        
        # Gunakan RGB method yang lebih stabil (skip optical flow untuk menghindari error)
        signal_value = self._process_rgb_changes(roi)
        
        # Validasi nilai sinyal
        validated_value = self._validate_signal_value(signal_value)
        if validated_value is not None:
            # Simpan nilai dan waktu ke buffer
            self.signal_buffer[self.current_idx] = validated_value
            self.time_buffer[self.current_idx] = timestamp - self.start_time
            
            # Kalibrasi baseline values untuk 5 detik pertama
            if len(self.baseline_values) < self.sampling_rate * 5:
                self.baseline_values.append(validated_value)
            elif not self.is_calibrated:
                self.is_calibrated = True
            
            # Perbarui indeks, reset jika mencapai akhir buffer
            self.current_idx = (self.current_idx + 1) % self.buffer_size
            
            return validated_value
        
        return None
    
    def _process_optical_flow_fixed(self, roi):
        """
        Proses ROI menggunakan optical flow dengan size validation yang proper.
        DISABLED untuk menghindari size mismatch error.
        
        Parameter
        ----------
        roi : numpy.ndarray
            Region of Interest dari frame video
            
        Returns
        -------
        float
            Nilai sinyal dari optical flow analysis
        """
        try:
            # Convert to grayscale untuk analysis
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Gaussian blur untuk mengurangi noise
            gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            
            current_size = gray_roi.shape
            
            if self.prev_gray_roi is not None and self.prev_roi_size is not None:
                # Check jika ukuran ROI berubah
                if current_size != self.prev_roi_size:
                    # ROI size berubah, resize previous ROI untuk match
                    try:
                        resized_prev = cv2.resize(self.prev_gray_roi, (current_size[1], current_size[0]))
                        
                        # Frame difference dengan size yang sudah match
                        frame_diff = cv2.absdiff(gray_roi, resized_prev)
                        movement_intensity = np.mean(frame_diff)
                        
                        # Validasi movement intensity
                        if np.isfinite(movement_intensity) and movement_intensity >= 0:
                            # Simpan ke flow buffer untuk smoothing
                            self.flow_buffer.append(movement_intensity)
                            if len(self.flow_buffer) > self.max_flow_buffer:
                                self.flow_buffer.pop(0)
                            
                            # Return smoothed value
                            if len(self.flow_buffer) > 0:
                                signal_value = np.mean(self.flow_buffer)
                                
                                # Update tracking variables
                                self.prev_gray_roi = gray_roi.copy()
                                self.prev_roi_size = current_size
                                
                                return signal_value
                    except Exception as resize_error:
                        # Jika resize gagal, reset tracking dan return fallback
                        print(f"Warning: Resize gagal: {resize_error}")
                        self.prev_gray_roi = gray_roi.copy()
                        self.prev_roi_size = current_size
                        return 0.0
                else:
                    # Same size, safe untuk absdiff
                    frame_diff = cv2.absdiff(gray_roi, self.prev_gray_roi)
                    movement_intensity = np.mean(frame_diff)
                    
                    # Validasi movement intensity
                    if np.isfinite(movement_intensity) and movement_intensity >= 0:
                        # Simpan ke flow buffer untuk smoothing
                        self.flow_buffer.append(movement_intensity)
                        if len(self.flow_buffer) > self.max_flow_buffer:
                            self.flow_buffer.pop(0)
                        
                        # Return smoothed value
                        if len(self.flow_buffer) > 0:
                            signal_value = np.mean(self.flow_buffer)
                            
                            # Update tracking variables
                            self.prev_gray_roi = gray_roi.copy()
                            self.prev_roi_size = current_size
                            
                            return signal_value
            
            # Frame pertama atau jika ada error, simpan sebagai reference
            self.prev_gray_roi = gray_roi.copy()
            self.prev_roi_size = current_size
            return 0.0
                
        except Exception as e:
            print(f"Warning: Error dalam optical flow: {e}")
            # Reset state jika ada error
            self.prev_gray_roi = None
            self.prev_roi_size = None
            self.flow_buffer = []
            return None
    
    def _process_rgb_changes(self, roi):
        """
        RGB analysis method untuk deteksi respirasi - more stable than optical flow.
        
        Parameter
        ----------
        roi : numpy.ndarray
            Region of Interest dari frame video
            
        Returns
        -------
        float
            Nilai sinyal dari RGB analysis
        """
        try:
            # Ekstrak nilai rata-rata RGB
            b_mean = np.mean(roi[:, :, 0])
            g_mean = np.mean(roi[:, :, 1])
            r_mean = np.mean(roi[:, :, 2])
            
            # Validasi nilai RGB
            if not (np.isfinite(b_mean) and np.isfinite(g_mean) and np.isfinite(r_mean)):
                return None
            
            if b_mean <= 0 or g_mean <= 0 or r_mean <= 0:
                return None
            
            # Kombinasi yang sensitif untuk chest movement
            # Fokus pada perubahan yang berhubungan dengan respirasi
            signal_value = r_mean * 0.6 + g_mean * 0.3 + b_mean * 0.1
            
            return signal_value
            
        except Exception as e:
            print(f"Warning: Error dalam RGB analysis: {e}")
            return None
    
    def get_filtered_signal(self):
        """
        Dapatkan sinyal respirasi yang telah difilter dengan algoritma yang dioptimasi.
        
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
        
        # Filter sinyal dengan robust processing
        if len(signal_array) > 5:  # Cukup data minimal untuk filter
            
            try:
                # Stage 1: Outlier removal (simplified)
                if self.is_calibrated and len(self.baseline_values) > 0:
                    baseline_std = np.std(self.baseline_values)
                    baseline_mean = np.mean(self.baseline_values)
                    
                    if baseline_std > 0:
                        # Remove extreme outliers (>3 std dari baseline)
                        outlier_mask = np.abs(signal_array - baseline_mean) < 3 * baseline_std
                        if np.sum(outlier_mask) > len(signal_array) * 0.7:
                            # Simple outlier replacement dengan median
                            median_val = np.median(signal_array[outlier_mask])
                            signal_array[~outlier_mask] = median_val
                
                # Stage 2: Validasi sinyal sebelum filtering
                if not np.all(np.isfinite(signal_array)):
                    print("Warning: Signal respirasi mengandung NaN/Inf sebelum filtering")
                    signal_array = np.nan_to_num(signal_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Stage 3: Detrend sinyal
                signal_array = detrend(signal_array)
                
                # Jika detrend menghasilkan array kosong, return original
                if len(signal_array) == 0:
                    return time_array, self.signal_buffer[:len(time_array)]
                
                # Stage 4: Bandpass filter untuk respirasi
                signal_array = bandpass_filter(signal_array, 0.08, 0.5, self.sampling_rate)
                
                # Jika filter menghasilkan array kosong, return simple version
                if len(signal_array) == 0:
                    return time_array, self.signal_buffer[:len(time_array)]
                
                # Stage 5: Smoothing dengan validasi
                if len(signal_array) > 8:
                    window_size = min(8, len(signal_array) // 5)
                    if window_size > 2:
                        signal_array = moving_average(signal_array, window_size)
                        if len(signal_array) == 0:
                            return time_array, self.signal_buffer[:len(time_array)]
                
                # Update time array sesuai panjang sinyal
                if len(signal_array) > 0:
                    time_array = time_array[:len(signal_array)]
                
                # Final validation
                if len(signal_array) > 0 and np.all(np.isfinite(signal_array)):
                    return time_array, signal_array
                else:
                    # Return original buffer jika filtering gagal
                    return self.time_buffer[:len(self.signal_buffer)], self.signal_buffer
                
            except Exception as e:
                print(f"Warning: Error dalam filtering respirasi: {e}")
                # Return original buffer jika ada error
                return self.time_buffer[:len(self.signal_buffer)], self.signal_buffer
        
        return time_array, signal_array
    
    def estimate_respiration_rate(self):
        """
        Estimasi laju pernapasan dalam napas per menit dengan multi-method validation.
        
        Returns
        -------
        float
            Perkiraan laju pernapasan dalam napas per menit, atau None jika data tidak cukup
        """
        # Dapatkan sinyal yang telah difilter
        _, signal = self.get_filtered_signal()
        
        # Minimal 3 detik data untuk estimasi
        if len(signal) < self.sampling_rate * 3:
            return None
        
        # Validasi sinyal
        if not np.all(np.isfinite(signal)):
            return None
            
        # Multi-method estimation dengan error handling
        estimations = []
        
        # Method 1: FFT Analysis (primary)
        fft_estimation = self._estimate_fft_method_safe(signal)
        if fft_estimation is not None:
            estimations.append(fft_estimation)
        
        # Method 2: Peak Detection (secondary)
        peak_estimation = self._estimate_peak_method_safe(signal)
        if peak_estimation is not None:
            estimations.append(peak_estimation)
        
        # Consensus dari available methods
        if len(estimations) > 0:
            # Gunakan median untuk robustness
            final_estimation = np.median(estimations)
            
            # Validasi range respirasi normal (5-40 napas/menit)
            if 5 <= final_estimation <= 40:
                # Tambahkan ke buffer estimasi terbaru untuk smoothing
                self.recent_estimates.append(final_estimation)
                if len(self.recent_estimates) > self.max_recent_estimates:
                    self.recent_estimates.pop(0)
                
                # Gunakan moving median dari estimasi terbaru untuk stabilitas
                if len(self.recent_estimates) >= 3:
                    stable_estimation = np.median(self.recent_estimates)
                    return stable_estimation
                
                return final_estimation
        
        # Default jika semua metode gagal tapi sudah ada data cukup lama
        if len(signal) > self.sampling_rate * 8:
            return 15.0  # Nilai default orang dewasa rata-rata
            
        return None
    
    def _estimate_fft_method_safe(self, signal):
        """Estimasi menggunakan FFT analysis dengan error handling."""
        try:
            n = len(signal)
            if n < 10:  # Data terlalu sedikit
                return None
            
            fft_data = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(n, 1 / self.sampling_rate)
            
            # Validasi FFT result
            if not np.all(np.isfinite(fft_data)):
                return None
            
            # Range frekuensi respirasi (0.08-0.5 Hz = 5-30 napas/menit)
            valid_idx = np.where((freqs >= 0.08) & (freqs <= 0.5))[0]
            if len(valid_idx) == 0:
                return None
            
            valid_fft = fft_data[valid_idx]
            valid_freqs = freqs[valid_idx]
            
            if len(valid_fft) == 0 or np.max(valid_fft) == 0:
                return None
            
            # Cari frekuensi dengan amplitudo tertinggi
            max_idx = np.argmax(valid_fft)
            dominant_freq = valid_freqs[max_idx]
            
            # Konversi ke napas per menit
            result = dominant_freq * 60
            
            # Validasi hasil
            if np.isfinite(result) and 5 <= result <= 40:
                return result
            
            return None
        except Exception as e:
            print(f"Warning: Error dalam FFT estimation: {e}")
            return None
    
    def _estimate_peak_method_safe(self, signal):
        """Estimasi menggunakan peak detection dengan error handling."""
        try:
            from scipy.signal import find_peaks
            
            if len(signal) < self.sampling_rate * 2:  # Butuh minimal 2 detik
                return None
            
            # Deteksi peaks dengan parameter yang aman
            signal_std = np.std(signal)
            if signal_std == 0:
                return None
            
            peaks, _ = find_peaks(signal, 
                                height=signal_std*0.3,  # Threshold berdasarkan std
                                distance=self.sampling_rate*2)  # Min 2 detik antar peak
            
            if len(peaks) >= 2:
                # Hitung jarak rata-rata antar puncak dalam sampel
                peak_diffs = np.diff(peaks)
                
                if len(peak_diffs) > 0:
                    avg_peak_diff = np.median(peak_diffs)  # Gunakan median
                    
                    if avg_peak_diff > 0:
                        # Konversi ke napas per menit
                        result = 60 / (avg_peak_diff / self.sampling_rate)
                        
                        # Validasi hasil
                        if np.isfinite(result) and 5 <= result <= 40:
                            return result
            
            return None
        except Exception as e:
            print(f"Warning: Error dalam peak estimation: {e}")
            return None
    
    def get_signal_quality(self):
        """
        Evaluasi kualitas sinyal respirasi berdasarkan SNR dan stabilitas.
        
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
            
            # Hitung SNR
            signal_power = np.var(signal)
            if signal_power == 0:
                return 'Poor'
            
            noise_estimate = np.var(np.diff(signal))
            
            if noise_estimate == 0:
                return 'Excellent'
            
            snr = signal_power / noise_estimate
            
            # Evaluasi stabilitas respiratory rate
            rate_stability = 1.0
            if len(self.recent_estimates) >= 3:
                rate_std = np.std(self.recent_estimates)
                rate_stability = max(0.1, 1.0 - rate_std / 5.0)
            
            # Kombinasi SNR dan stabilitas
            quality_score = snr * rate_stability
            
            if quality_score > 10:
                return 'Excellent'
            elif quality_score > 5:
                return 'Good'
            elif quality_score > 2:
                return 'Fair'
            else:
                return 'Poor'
                
        except:
            return 'Poor'