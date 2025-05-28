# SignalScope: Real-Time Respiration and rPPG Analyzer
**Tugas Besar Mata Kuliah Digital Signal Processing (IF3024)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 👨‍🏫 Dosen Pengampu
**Martin Clinton Tosima Manullang, S.T., M.T.**

## 👥 Anggota Kelompok

| **Nama** | **NIM** | **GitHub** |
|----------|---------|------------|
| Ferdana Al-Hakim | 122140012 | [@luciferdana](https://github.com/luciferdana) |
| Ihya Razky Hidayat | 121140167 | [@Ecstarssyy](https://github.com/Ecstarssyy) |
| Rayhan Fadel Irwanto | 121140238 | [@RayhanFadelIrwanto](https://github.com/RayhanFadelIrwanto) |

---

## 📖 Deskripsi Proyek

**SignalScope** adalah aplikasi desktop berbasis PyQt5 yang dikembangkan untuk menganalisis sinyal respirasi (pernapasan) dan *remote photoplethysmography* (rPPG) secara real-time menggunakan input webcam. Program ini mengimplementasikan teknik pemrosesan sinyal digital canggih dengan antarmuka pengguna yang intuitif untuk monitoring kesehatan non-invasif.

### 🎯 Tujuan Utama
- Ekstraksi sinyal respirasi dari pergerakan dada menggunakan computer vision
- Deteksi sinyal rPPG dari perubahan warna kulit wajah untuk estimasi detak jantung
- Implementasi filter digital untuk mendapatkan sinyal yang bersih dan akurat
- Visualisasi real-time dengan antarmuka yang user-friendly

---

## ✨ Fitur Unggulan

### 🖥️ **Antarmuka Pengguna Modern**
- **GUI PyQt5** dengan layout responsif dan intuitif
- **Real-time video display** dengan overlay ROI detection
- **Dual-panel design** untuk video dan signal analysis
- **Control buttons** untuk start/stop/save dengan status feedback

### 📊 **Visualisasi Real-Time**
- **Live plotting** menggunakan PyQtGraph untuk performa optimal
- **Dual signal display**: Respirasi (biru) dan rPPG (merah)
- **Dynamic scaling** dan smooth signal updates
- **BPM/respiratory rate indicators** dengan validasi range

### 🎛️ **Pemrosesan Sinyal Canggih**
- **Multiple estimation methods**: FFT, zero-crossing, peak-detection
- **Butterworth bandpass filtering** dengan parameter yang dioptimasi
- **Adaptive signal processing** dengan buffer management
- **Noise reduction** menggunakan moving average dan detrending

### 🤖 **Computer Vision Terdepan**
- **BlazeFace integration** untuk deteksi wajah yang akurat
- **MediaPipe Pose Landmarker** untuk penentuan ROI dada yang presisi
- **Automatic fallback mechanism** untuk robustness
- **Real-time ROI tracking** dengan visualisasi overlay

### 💾 **Data Management**
- **CSV export functionality** dengan timestamp otomatis
- **Organized data storage** dalam direktori `data/`
- **Signal preservation** untuk analisis lebih lanjut
- **Metadata inclusion** untuk reproducibility

---

## 🏗️ Struktur Proyek

```
DSP-Final-Project/
├── 📁 models/                          # Model MediaPipe
│   ├── 🔧 blaze_face_short_range.tflite
│   └── 🔧 pose_landmarker.task
├── 📁 src/
│   ├── 📁 gui/                         # Antarmuka Pengguna
│   │   ├── 🎨 main_window.py           # Main GUI application
│   │   └── 🔧 __init__.py
│   ├── 📁 signal/                      # Pemrosesan Sinyal
│   │   ├── 🔧 __init__.py
│   │   ├── 🌊 filters.py               # Filter digital (Butterworth, MA)
│   │   ├── 🫁 respiration.py           # Ekstraksi sinyal respirasi
│   │   └── ❤️ rppg.py                  # Ekstraksi sinyal rPPG
│   ├── 📁 video/                       # Computer Vision
│   │   ├── 🔧 __init__.py
│   │   ├── 📹 camera.py                # Interface webcam
│   │   └── 👁️ processor.py             # Deteksi wajah & ROI
│   └── 📁 utils/                       # Utilities
│       ├── 🔧 __init__.py
│       ├── 🛠️ helpers.py               # Helper functions
│       └── ⚙️ utils.py                 # Konfigurasi constants
├── 📁 data/                            # Output data (auto-generated)
├── 🐍 main.py                          # Entry point aplikasi
├── 📋 requirements.txt                 # Dependencies
├── 📝 test_models.py                   # Test script (opsional)
└── 📖 README.md                        # Dokumentasi
```

---

## 🛠️ Teknologi yang Digunakan

### **Core Technologies**
- **Python 3.8+**: Bahasa pemrograman utama
- **PyQt5**: Framework GUI untuk antarmuka desktop
- **OpenCV**: Computer vision dan video processing
- **NumPy**: Operasi numerik dan manipulasi array
- **SciPy**: Implementasi filter digital dan signal processing

### **Specialized Libraries**
- **PyQtGraph**: Real-time plotting dengan performa tinggi
- **MediaPipe**: Advanced face detection dan pose estimation
- **scikit-image**: Image processing tambahan

### **AI/ML Models**
- **BlazeFace**: Model deteksi wajah yang dioptimasi untuk real-time
- **Pose Landmarker**: Model estimasi pose untuk deteksi ROI dada

### **Development Tools**
- **Git & GitHub**: Version control dan kolaborasi
- **Virtual Environment**: Isolasi dependencies
- **Modular Architecture**: Clean code practices

---

## ⚙️ Instalasi & Setup

### **1. Clone Repository**
```bash
git clone https://github.com/luciferdana/DSP-Final-Project.git
cd DSP-Final-Project
```

### **2. Setup Virtual Environment (Direkomendasikan)**
```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Verifikasi Setup (Opsional)**
```bash
python test_models.py
```

---

## 📦 Requirements

```txt
PyQt5==5.15.9
opencv-python==4.8.1.78
numpy==1.24.3
scipy==1.11.1
pyqtgraph==0.13.3
mediapipe==0.10.7
scikit-image==0.21.0
```

---

## 🚀 Cara Penggunaan

### **1. Jalankan Aplikasi**
```bash
python main.py
```

### **2. Operasi Dasar**
1. **Start Camera**: Klik tombol "Mulai" untuk mengaktifkan webcam
2. **Position Yourself**: Posisikan wajah dalam frame dengan pencahayaan yang cukup
3. **Monitor Signals**: Amati grafik real-time untuk sinyal respirasi dan rPPG
4. **Save Data**: Klik "Simpan Data" untuk export sinyal ke CSV
5. **Stop Recording**: Klik "Berhenti" untuk menghentikan akuisisi

### **3. Tips Optimal Usage**
- **Pencahayaan**: Gunakan cahaya yang stabil dan cukup
- **Posisi**: Jaga wajah tetap dalam frame dan relatif stabil
- **Background**: Hindari background yang kompleks atau bergerak
- **Distance**: Posisikan pada jarak 50-100cm dari kamera

---

## 📊 Output & Data

### **Real-time Display**
- **Video Feed**: Live webcam dengan ROI overlay (hijau: wajah, biru: dahi, merah: dada)
- **Signal Plots**: Grafik sinyal respirasi dan rPPG dengan update real-time
- **Measurements**: BPM dan respiratory rate dengan validasi range

### **Exported Data**
- **Location**: `data/` directory (auto-created)
- **Format**: CSV files dengan timestamp
- **Naming**: `respirasi_YYYYMMDD_HHMMSS.csv` dan `rppg_YYYYMMDD_HHMMSS.csv`
- **Content**: Time series data (waktu, amplitudo sinyal)

---

## 🔧 Troubleshooting

### **Kamera Issues**
```bash
# Error: Tidak dapat mengakses kamera
- Pastikan webcam terhubung dan tidak digunakan aplikasi lain
- Restart aplikasi atau reboot sistem
- Check permissions kamera pada sistem operasi
```

### **Model Loading Issues**
```bash
# Error: Model tidak ditemukan
- Pastikan file model ada di direktori models/
- Download ulang model jika corrupted
- Check file permissions
```

### **Performance Issues**
```bash
# FPS rendah atau lag
- Tutup aplikasi lain yang menggunakan CPU/GPU intensif
- Turunkan resolusi webcam jika diperlukan
- Update driver graphics card
```

### **Signal Quality Issues**
```bash
# Sinyal noisy atau tidak stabil
- Improve lighting conditions
- Minimize movement selama recording
- Check ROI positioning (wajah harus terdeteksi dengan baik)
- Tunggu beberapa detik untuk stabilisasi filter
```

---

## 🧪 Testing & Validation

### **Run Test Suite**
```bash
python test_models.py
```

**Expected Output:**
```
🔍 DSP Final Project - Model & System Test
==================================================
✅ PASS Model Files
✅ PASS Video Processor  
✅ PASS Signal Processors
✅ PASS GUI Components

🎉 Semua test PASSED! Project siap dijalankan.
```

---

## 📈 Performance Metrics

- **Real-time Processing**: ~30 FPS video processing
- **Signal Update Rate**: 33ms refresh interval
- **Latency**: <100ms dari capture ke display
- **Accuracy**: Validasi dengan range fisiologis normal
- **Memory Usage**: <500MB typical operation

---

## 🗓️ Logbook Pengembangan

| **Minggu** | **Periode** | **Progress & Milestone** | **Tantangan & Solusi** |
|:----------:|:-----------:|:-------------------------|:-----------------------|
| **1** | 05-11 Mei | • Pembentukan tim & setup project<br>• Studi literatur rPPG & respirasi analysis<br>• **✅ Repository setup & invite dosen**<br>• Design arsitektur modular | • Sinkronisasi pemahaman antar anggota<br>• Setup development environment<br>• Planning project structure |
| **2** | 12-18 Mei | • **✅ Implementasi camera interface (OpenCV)**<br>• Basic face detection dengan Haar Cascade<br>• ROI extraction untuk rPPG (forehead area)<br>• Initial signal extraction algoritma | • Stabilitas ROI tracking<br>• Noise reduction pada raw signal<br>• **Solusi**: Buffer management & smoothing |
| **3** | 19-25 Mei | • **✅ Digital filter design (Butterworth bandpass)**<br>• Parameter tuning untuk rPPG (0.7-3.5 Hz)<br>• Respirasi signal extraction (chest movement)<br>• **✅ Real-time visualization dengan PyQtGraph** | • Filter parameter optimization<br>• Respiratory vs body movement separation<br>• **Solusi**: Multi-method estimation |
| **4** | 26 Mei-01 Jun | • **✅ MediaPipe integration (BlazeFace + Pose)**<br>• **✅ GUI development dengan PyQt5**<br>• Advanced ROI detection dengan pose landmarks<br>• **✅ Data export functionality** | • Real-time performance optimization<br>• Model integration challenges<br>• **Solusi**: Fallback mechanisms |
| **5** | 02-08 Jun | • **✅ Code refinement & documentation**<br>• **✅ Testing & validation suite**<br>• Performance optimization<br>• **📝 Laporan penulisan** | • Final testing across different hardware<br>• Documentation completion<br>• **Prep**: Demo presentation |

---

## 🎯 Future Enhancements

- [ ] **Multi-person detection** untuk analisis group
- [ ] **Historical data analysis** dengan trend visualization
- [ ] **Alert system** untuk abnormal vital signs
- [ ] **Export to other formats** (JSON, Excel)
- [ ] **Mobile app version** untuk portability
- [ ] **Cloud integration** untuk data backup

---

## 📄 Lisensi

MIT License - Lihat file [LICENSE](LICENSE) untuk detail lengkap.

---

## 🤝 Kontribusi

Project ini dikembangkan sebagai tugas akhir mata kuliah. Untuk diskusi atau pertanyaan, silakan hubungi anggota tim melalui GitHub.

---

## 📞 Kontak

**Repository**: [DSP-Final-Project](https://github.com/luciferdana/DSP-Final-Project)  
**Course**: IF3024 - Digital Signal Processing  
**Institution**: Institut Teknologi Sumatera 
**Year**: 2025

---

<div align="center">

**🎉 Developed with ❤️ by Team SignalScope**

*Real-time Health Monitoring through Computer Vision & Digital Signal Processing*

</div>