#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul yang berisi filter digital untuk pemrosesan sinyal.
Dengan validasi untuk menangani NaN dan Inf values.
"""

import numpy as np
from scipy import signal

def validate_signal(data):
    """
    Validasi dan bersihkan sinyal dari NaN dan Inf values.
    
    Parameter
    ----------
    data : numpy.ndarray
        Data sinyal input
        
    Returns
    -------
    numpy.ndarray
        Data sinyal yang sudah dibersihkan
    bool
        True jika data valid, False jika tidak dapat diperbaiki
    """
    if data is None:
        return np.array([]), False
    
    # Convert ke numpy array jika belum
    data = np.asarray(data, dtype=np.float64)
    
    # Check jika array kosong
    if len(data) == 0:
        return data, False
    
    # Check dan replace NaN values
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        if np.all(nan_mask):
            # Semua nilai NaN, return array kosong
            return np.array([]), False
        
        # Replace NaN dengan interpolasi dari nilai valid
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 1:
            data[nan_mask] = np.interp(
                np.where(nan_mask)[0], 
                valid_indices, 
                data[valid_indices]
            )
        else:
            # Tidak cukup data valid, replace dengan mean dari nilai valid
            valid_mean = np.mean(data[~nan_mask])
            data[nan_mask] = valid_mean
    
    # Check dan replace Inf values
    inf_mask = np.isinf(data)
    if np.any(inf_mask):
        if np.all(inf_mask):
            # Semua nilai Inf, return array kosong
            return np.array([]), False
        
        # Replace Inf dengan nilai median dari data valid
        finite_data = data[np.isfinite(data)]
        if len(finite_data) > 0:
            replacement_value = np.median(finite_data)
            data[inf_mask] = replacement_value
        else:
            return np.array([]), False
    
    # Final check untuk memastikan semua nilai finite
    if not np.all(np.isfinite(data)):
        return np.array([]), False
    
    return data, True

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Menerapkan filter bandpass Butterworth pada data input dengan validasi.
    
    Parameter
    ----------
    data : numpy.ndarray
        Data sinyal input
    lowcut : float
        Frekuensi cutoff rendah dalam Hz
    highcut : float
        Frekuensi cutoff tinggi dalam Hz
    fs : float
        Frekuensi sampling dalam Hz
    order : int, opsional
        Orde filter, default 4
        
    Returns
    -------
    numpy.ndarray
        Data sinyal yang telah difilter
    """
    # Validasi input data
    clean_data, is_valid = validate_signal(data)
    if not is_valid or len(clean_data) < order * 2:
        return clean_data
    
    try:
        # Validasi parameter filter
        nyq = 0.5 * fs
        if lowcut >= nyq or highcut >= nyq or lowcut >= highcut:
            # Parameter filter tidak valid, return data original
            return clean_data
        
        low = lowcut / nyq
        high = highcut / nyq
        
        # Pastikan cutoff frequencies dalam range valid (0, 1)
        low = max(0.001, min(0.999, low))
        high = max(0.001, min(0.999, high))
        
        if low >= high:
            low = high * 0.8  # Adjust low cutoff
        
        # Buat filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter dengan zero-phase filtering
        y = signal.filtfilt(b, a, clean_data)
        
        # Validasi output
        filtered_data, is_filtered_valid = validate_signal(y)
        if is_filtered_valid:
            return filtered_data
        else:
            # Jika filtered data tidak valid, return original clean data
            return clean_data
            
    except Exception as e:
        # Jika filter gagal, return data yang sudah dibersihkan
        print(f"Warning: Bandpass filter gagal: {e}")
        return clean_data

def moving_average(data, window_size):
    """
    Menerapkan filter moving average pada data input dengan validasi.
    
    Parameter
    ----------
    data : numpy.ndarray
        Data sinyal input
    window_size : int
        Ukuran jendela moving average
        
    Returns
    -------
    numpy.ndarray
        Data sinyal yang telah difilter
    """
    # Validasi input data
    clean_data, is_valid = validate_signal(data)
    if not is_valid or len(clean_data) == 0:
        return clean_data
    
    # Validasi window size
    window_size = max(1, min(window_size, len(clean_data)))
    
    try:
        # Terapkan moving average dua kali untuk smoothing yang lebih baik
        if len(clean_data) >= window_size:
            smoothed = np.convolve(clean_data, np.ones(window_size)/window_size, mode='valid')
            
            # Terapkan sekali lagi untuk hasil lebih halus
            if len(smoothed) > max(3, window_size//2):
                second_window = max(3, window_size//2)
                smoothed = np.convolve(smoothed, np.ones(second_window)/second_window, mode='valid')
            
            # Validasi hasil
            final_result, is_final_valid = validate_signal(smoothed)
            if is_final_valid and len(final_result) > 0:
                return final_result
        
        # Jika moving average gagal, return original data
        return clean_data
        
    except Exception as e:
        print(f"Warning: Moving average gagal: {e}")
        return clean_data

def detrend(data):
    """
    Menghilangkan trend linear dari data input dengan validasi.
    
    Parameter
    ----------
    data : numpy.ndarray
        Data sinyal input
        
    Returns
    -------
    numpy.ndarray
        Data sinyal yang telah dihilangkan trendnya
    """
    # Validasi input data
    clean_data, is_valid = validate_signal(data)
    if not is_valid or len(clean_data) < 2:
        return clean_data
    
    try:
        # Apply detrend dengan scipy
        detrended = signal.detrend(clean_data)
        
        # Validasi hasil detrend
        final_result, is_final_valid = validate_signal(detrended)
        if is_final_valid:
            return final_result
        else:
            # Jika detrend gagal, lakukan detrend manual sederhana
            mean_value = np.mean(clean_data)
            return clean_data - mean_value
            
    except Exception as e:
        print(f"Warning: Detrend gagal: {e}")
        # Fallback: simple mean removal
        try:
            mean_value = np.mean(clean_data)
            return clean_data - mean_value
        except:
            return clean_data