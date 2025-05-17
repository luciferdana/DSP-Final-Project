"""
Modul yang berisi filter digital untuk pemrosesan sinyal.
"""

import numpy as np
from scipy import signal

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Menerapkan filter bandpass Butterworth pada data input.
    
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
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def moving_average(data, window_size):
    """
    Menerapkan filter moving average pada data input.
    
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
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def detrend(data):
    """
    Menghilangkan trend linear dari data input.
    
    Parameter
    ----------
    data : numpy.ndarray
        Data sinyal input
        
    Returns
    -------
    numpy.ndarray
        Data sinyal yang telah dihilangkan trendnya
    """
    return signal.detrend(data)