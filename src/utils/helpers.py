import time
import os
import numpy as np

def get_timestamp():
    """
    Mendapatkan timestamp saat ini.
    
    Returns
    -------
    float
        Waktu saat ini dalam detik sejak epoch
    """
    return time.time()

def format_time(seconds):
    """
    Format waktu dalam detik menjadi string yang mudah dibaca.
    
    Parameters
    ----------
    seconds : float
        Waktu dalam detik
        
    Returns
    -------
    str
        String waktu yang diformat (MM:SS.ss)
    """
    minutes = int(seconds) // 60
    seconds_rem = seconds - (minutes * 60)
    return f"{minutes:02d}:{seconds_rem:05.2f}"

def ensure_directory_exists(directory_path):
    """
    Memastikan direktori ada, membuat jika belum ada.
    
    Parameters
    ----------
    directory_path : str
        Path direktori yang akan diperiksa/dibuat
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
def save_data_to_csv(time_array, signal_array, filename):
    """
    Menyimpan data sinyal ke file CSV.
    
    Parameters
    ----------
    time_array : numpy.ndarray
        Array waktu
    signal_array : numpy.ndarray
        Array nilai sinyal
    filename : str
        Nama file untuk menyimpan data
    """
    data = np.column_stack((time_array, signal_array))
    np.savetxt(filename, data, delimiter=',', header='time,signal', comments='')