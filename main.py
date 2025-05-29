import sys
import logging
from src.gui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication, QMessageBox

def setup_logging():
    """Menyiapkan sistem logging untuk debugging dan monitoring."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('signalscope.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Aplikasi SignalScope dimulai...")
    return logger

def handle_exception(exc_type, exc_value, exc_traceback):
    """Penanganan error global untuk mencegah crash tanpa pesan."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Catat error ke log
    logging.error("Error yang tidak tertangani", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Tampilkan dialog error yang user-friendly
    error_msg = f"Terjadi error tidak terduga:\n\n{str(exc_value)}\n\nCek file signalscope.log untuk detail lengkap."
    QMessageBox.critical(None, "SignalScope Error", error_msg)

def main():
    """Fungsi utama aplikasi."""
    # Setup sistem logging
    logger = setup_logging()
    
    # Setup penanganan error global
    sys.excepthook = handle_exception
    
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        # Buat dan tampilkan jendela utama
        logger.info("Menginisialisasi jendela utama...")
        window = MainWindow()
        window.show()
        
        logger.info("Aplikasi siap - memulai event loop")
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Gagal memulai aplikasi: {e}")
        QMessageBox.critical(None, "Error Startup", 
                           f"Gagal memulai aplikasi:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()