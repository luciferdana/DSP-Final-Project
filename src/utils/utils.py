# Parameter kamera
CAMERA_CONFIG = {
    'default_id': 0,
    'width': 640,
    'height': 480,
    'fps': 30,
}

# Parameter filter respirasi
RESPIRATION_CONFIG = {
    'buffer_size': 150,        # Ukuran buffer untuk menyimpan sinyal
    'sampling_rate': 30,       # Laju sampling (sama dengan FPS kamera)
    'lowcut': 0.1,             # Frekuensi cutoff rendah (Hz)
    'highcut': 0.5,            # Frekuensi cutoff tinggi (Hz)
    'filter_order': 4,         # Orde filter Butterworth
    'window_size': 10,         # Ukuran window untuk moving average
}

# Parameter filter rPPG
RPPG_CONFIG = {
    'buffer_size': 300,        # Ukuran buffer untuk menyimpan sinyal
    'sampling_rate': 30,       # Laju sampling (sama dengan FPS kamera)
    'lowcut': 0.7,             # Frekuensi cutoff rendah (Hz) ~ 42 BPM
    'highcut': 3.5,            # Frekuensi cutoff tinggi (Hz) ~ 210 BPM
    'filter_order': 4,         # Orde filter Butterworth
    'window_size': 5,          # Ukuran window untuk moving average
}

# Warna untuk visualisasi
VISUALIZATION_COLORS = {
    'respiration': '#2E86C1',  # Warna biru untuk sinyal respirasi
    'rppg': '#C0392B',         # Warna merah untuk sinyal rPPG
    'background': '#F5F5F5',   # Warna latar belakang plot
    'grid': '#DDDDDD',         # Warna grid plot
}

# Ukuran buffer plot untuk visualisasi
PLOT_BUFFER_SIZE = 300  # Menampilkan 10 detik terakhir pada 30 FPS

# ROI warna untuk visualisasi
ROI_COLORS = {
    'face': (0, 255, 0),       # Hijau untuk deteksi wajah
    'forehead': (255, 0, 0),   # Merah untuk ROI dahi (rPPG)
    'chest': (0, 0, 255),      # Biru untuk ROI dada (respirasi)
}

# Parameter untuk deteksi ROI
ROI_CONFIG = {
    'face_scale_factor': 1.1,  # Parameter untuk deteksi wajah
    'face_min_neighbors': 5,   # Parameter untuk deteksi wajah
    'forehead_ratio': 0.3,     # Proporsi tinggi dahi relatif terhadap wajah
    'chest_width_factor': 1.5, # Faktor lebar dada relatif terhadap wajah
    'chest_height_factor': 1.5 # Faktor tinggi dada relatif terhadap wajah
}