#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script untuk test model MediaPipe dan deteksi ROI.
Jalankan untuk memverifikasi setup model benar.
"""

import cv2
import os
import sys

def check_model_files():
    """Check apakah model files ada di direktori models/."""
    print("=== Checking Model Files ===")
    
    models_dir = "models"
    required_files = [
        "blaze_face_short_range.tflite",
        "pose_landmarker.task"
    ]
    
    if not os.path.exists(models_dir):
        print(f"❌ Directory {models_dir}/ tidak ditemukan!")
        return False
    
    all_found = True
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file} - {file_size} bytes")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_found = False
    
    return all_found

def test_processor():
    """Test VideoProcessor dengan webcam."""
    print("\n=== Testing VideoProcessor ===")
    
    try:
        # Import processor
        from src.video.processor import get_processor, detect_face, get_forehead_roi, get_chest_roi
        
        print("✅ Import processor berhasil")
        
        # Initialize processor
        processor = get_processor()
        print("✅ VideoProcessor initialized")
        
        # Test dengan webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Tidak bisa akses webcam")
            return False
        
        print("✅ Webcam terhubung")
        
        # Capture satu frame untuk test
        ret, frame = cap.read()
        if not ret:
            print("❌ Tidak bisa capture frame")
            cap.release()
            return False
        
        print("✅ Frame captured")
        
        # Test face detection
        face_rect = detect_face(frame)
        if face_rect is not None:
            print(f"✅ Face detected: {face_rect}")
            
            # Test forehead ROI
            forehead_result = get_forehead_roi(face_rect, frame)
            if forehead_result is not None:
                print("✅ Forehead ROI extracted")
            else:
                print("❌ Forehead ROI failed")
            
            # Test chest ROI
            chest_result = get_chest_roi(face_rect, frame)
            if chest_result is not None:
                print("✅ Chest ROI extracted (pose detection)")
            else:
                print("⚠️  Chest ROI failed (trying fallback)")
        else:
            print("⚠️  No face detected (pastikan wajah terlihat di kamera)")
        
        # Cleanup
        cap.release()
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_gui():
    """Test import GUI components."""
    print("\n=== Testing GUI Components ===")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from src.gui.main_window import MainWindow
        print("✅ GUI imports berhasil")
        return True
    except ImportError as e:
        print(f"❌ GUI import error: {e}")
        print("Pastikan PyQt5 terinstall: pip install PyQt5")
        return False

def test_signal_processors():
    """Test signal processors."""
    print("\n=== Testing Signal Processors ===")
    
    try:
        from src.signal.respiration import RespirationSignalProcessor
        from src.signal.rppg import RPPGSignalProcessor
        from src.signal.filters import bandpass_filter
        
        # Test initialization
        resp_proc = RespirationSignalProcessor()
        rppg_proc = RPPGSignalProcessor()
        
        print("✅ Signal processors initialized")
        
        # Test dengan dummy data
        import numpy as np
        dummy_data = np.random.randn(100)
        filtered = bandpass_filter(dummy_data, 0.1, 0.5, 30)
        
        print("✅ Filter test berhasil")
        return True
        
    except ImportError as e:
        print(f"❌ Signal import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Signal error: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 DSP Final Project - Model & System Test")
    print("=" * 50)
    
    # Test sequence
    tests = [
        ("Model Files", check_model_files),
        ("Video Processor", test_processor),
        ("Signal Processors", test_signal_processors),
        ("GUI Components", test_gui)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Semua test PASSED! Project siap dijalankan.")
        print("Jalankan: python main.py")
    else:
        print("\n⚠️  Ada test yang FAILED. Periksa error di atas.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()