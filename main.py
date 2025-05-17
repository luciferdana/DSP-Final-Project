import sys
from src.gui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication

def main():
    """Fungsi utama untuk menjalankan aplikasi"""
    app = QApplication(sys.argv)
    
    # Gaya aplikasi (opsional)
    app.setStyle("Fusion")
    
    # Buat dan tampilkan jendela utama
    window = MainWindow()
    window.show()
    
    # Jalankan event loop aplikasi
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()