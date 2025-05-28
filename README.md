# Tugas Besar Mata Kuliah Digital Processing Signal (IF3024) SignalScope: Real-Time Respiration and rPPG Analyzer 🚦呼吸💓

## Dosen Pengampu: **Martin Clinton Tosima Manullang, S.T., M.T.**

## Anggota Kelompok

| **Nama** | **NIM** | **ID GITHUB** |
| ---------------------| --------- | ----------------------------------------------------------- |
| Ferdana Al- Hakim    | 122140012 | <a href="https://github.com/luciferdana">@luciferdana</a>   |
| Ihya Razky Hidayat   | 121140167 | <a href="https://github.com/Ecstarssyy">@Ecstarssyy</a>     |
| Rayhan Fadel Irwanto | 121140238 | <a href="https://github.com/RayhanFadelIrwanto">@RayhanFadelIrwanto</a>     |

---

## 📝 Deskripsi Proyek

SignalScope adalah sebuah program yang dikembangkan sebagai Proyek Akhir mata kuliah Pengolahan Sinyal Digital (IF3024). Program ini bertujuan untuk melakukan pengukuran sinyal respirasi (pernapasan) dan *remote photoplethysmography* (rPPG) secara *real-time* menggunakan input video dari webcam[cite: 4, 5]. Fokus utama proyek ini adalah pada desain filter dan teknik pemrosesan sinyal untuk mengekstraksi kedua sinyal tersebut secara akurat dan stabil[cite: 8].

---

## ✨ Fitur Utama

* **Deteksi Sinyal Respirasi Real-Time:** Menganalisis frame video untuk mendeteksi pergerakan halus yang mengindikasikan laju pernapasan.
* **Deteksi Sinyal rPPG Real-Time:** Mengekstraksi sinyal rPPG dari perubahan warna kulit wajah yang disebabkan oleh aliran darah.
* **Visualisasi Sinyal:** Menampilkan grafik sinyal respirasi dan rPPG secara *real-time* menggunakan `matplotlib` dan `cv2`[cite: 6].
* **Pemrosesan Sinyal:** Implementasi filter digital dan algoritma pemrosesan sinyal untuk mendapatkan sinyal yang bersih dan akurat[cite: 8].
* **Modularitas Kode:** Fungsi-fungsi kompleks dipisahkan ke dalam modul-modul terpisah untuk kemudahan pemeliharaan[cite: 9].
* **Deteksi Wajah/Pose Lanjut:** Menggunakan model `pose_landmarker.task` dan `blaze_face_short_range.tflite` untuk Region of Interest (ROI) yang lebih akurat.

---

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python [cite: 9]
* **Library Utama:**
    * OpenCV (`cv2`): Untuk akuisisi dan pemrosesan frame video.
    * NumPy: Untuk operasi numerik dan manipulasi array.
    * SciPy: Untuk implementasi filter digital dan fungsi pemrosesan sinyal lainnya.
    * Matplotlib: Untuk visualisasi grafik sinyal[cite: 6].
    * MediaPipe (atau library serupa): Untuk `pose_landmarker` dan `blaze_face`.
* **Model Machine Learning:**
    * `pose_landmarker.task`
    * `blaze_face_short_range.tflite`
* **Tools Lain:**
    * Git & GitHub: Untuk kontrol versi dan kolaborasi.
    * Overleaf: Untuk penulisan laporan (report.pdf)[cite: 11].

---

## ⚙️ Instruksi Instalasi

1.  **Clone Repository:**
    ```bash
    git clone <URL_REPOSITORY_ANDA>
    cd <NAMA_FOLDER_REPOSITORY>
    ```

2.  **Buat dan Aktifkan Virtual Environment (Opsional tapi Direkomendasikan):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Pastikan Anda memiliki file `requirements.txt` dalam repositori Anda[cite: 11].
    ```bash
    pip install -r requirements.txt
    ```
    *Contoh isi `requirements.txt` mungkin seperti ini:*
    ```
    numpy
    opencv-python
    scipy
    matplotlib
    mediapipe # atau library spesifik untuk model .tflite jika berbeda
    ```
4.  **Download Model (Jika Perlu):**
    Pastikan model `pose_landmarker.task` dan `blaze_face_short_range.tflite` tersedia di path yang benar sesuai konfigurasi program Anda. Anda mungkin perlu menyertakan instruksi untuk mengunduhnya atau menyertakannya langsung di repositori (jika ukurannya memungkinkan dan lisensinya mengizinkan).

---

## 🚀 Instruksi Penggunaan Program

1.  **Jalankan Program Utama:**
    Misalnya, jika file utama Anda bernama `main.py`:
    ```bash
    python main.py
    ```
2.  **Akses Webcam:**
    Program akan meminta izin untuk mengakses webcam Anda. Pastikan webcam terhubung dan tidak digunakan oleh aplikasi lain.
3.  **Visualisasi Sinyal:**
    Sebuah jendela akan muncul menampilkan feed video dari webcam, beserta grafik sinyal respirasi dan rPPG yang sedang diproses secara *real-time*[cite: 5].

---


---

## 🗓️ Logbook Mingguan

| **Minggu Ke-** | **Periode Tanggal (2025)** | **Aktivitas & Progress** | **Catatan/Tantangan** |
| :------------: | :------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|       1        | 05 Mei - 11 Mei            | - Pembentukan kelompok & diskusi awal pembagian tugas.<br>- Studi literatur intensif mengenai teknik ekstraksi sinyal respirasi dan rPPG dari video.<br>- Setup environment proyek (Python, OpenCV, Git).<br>- Inisialisasi repositori GitHub.<br>- **Invite dosen pengampu ke repositori GitHub (selesai sebelum 10 Mei).** [cite: 12]<br>- Desain awal alur program dan struktur folder. | - Memastikan pemahaman yang sama antar anggota tim mengenai metode yang akan diimplementasikan.<br>- Menyesuaikan dengan *hands-on* yang sudah diberikan.                                                                                                             |
|       2        | 12 Mei - 18 Mei            | - Implementasi akuisisi video dari webcam menggunakan OpenCV.<br>- Riset dan eksplorasi awal untuk deteksi wajah sebagai Region of Interest (ROI) untuk rPPG.<br>- Implementasi algoritma dasar ekstraksi sinyal rPPG (misalnya rata-rata nilai piksel pada ROI).<br>- Eksplorasi metode filtering awal untuk sinyal rPPG.               | - Stabilitas ROI wajah jika ada pergerakan signifikan.<br>- Noise yang cukup tinggi pada sinyal rPPG mentah.<br>- Memulai pemisahan fungsi ke file .py terpisah[cite: 9].                                                                                                     |
|       3        | 19 Mei - 25 Mei            | - Desain dan implementasi filter digital (misalnya Butterworth bandpass filter) untuk sinyal rPPG.<br>- Penentuan parameter filter (frekuensi cut-off) untuk rPPG.<br>- Implementasi metode ekstraksi sinyal respirasi (misalnya analisis pergerakan pada ROI tertentu atau Optical Flow sederhana).<br>- Visualisasi sinyal rPPG dan respirasi awal menggunakan matplotlib[cite: 6]. | - Tuning parameter filter agar efektif mengurangi noise tanpa menghilangkan komponen sinyal penting.<br>- Tantangan dalam membedakan pergerakan akibat napas dengan pergerakan tubuh lainnya.<br>- Sinkronisasi antara pemrosesan sinyal dan *real-time display*.                 |
|       4        | 26 Mei - (Saat Ini)        | - **Integrasi model `pose_landmarker.task` dan `blaze_face_short_range.tflite` untuk deteksi wajah/pose yang lebih robust dan penentuan ROI yang lebih akurat.**<br>- Penyesuaian algoritma ekstraksi sinyal rPPG dan respirasi berdasarkan output model baru.<br>- Uji coba efektivitas model dalam berbagai kondisi pencahayaan dan pergerakan subjek.<br>- Mulai penulisan dokumentasi kode (docstrings, komentar)[cite: 9]. | - Memastikan model dapat berjalan secara efisien untuk *real-time processing*[cite: 13].<br>- Mengelola dependensi model dan ukurannya.<br>- Kebutuhan untuk penyesuaian parameter filter setelah perubahan metode deteksi ROI.<br>- Memulai penulisan laporan dan `README.md`[cite: 10, 11]. |

---

Semoga berhasil dengan kelanjutan proyeknya!