"""
Modul untuk memproses video dan mendeteksi area yang diinginkan.
"""

import cv2
import numpy as np

def detect_face(frame):
    """
    Mendeteksi wajah pada frame yang diberikan.
    
    Parameter
    ----------
    frame : numpy.ndarray
        Frame video input
        
    Returns
    -------
    tuple atau None
        (x, y, w, h) koordinat wajah, atau None jika tidak ada wajah terdeteksi
    """
    # Konversi ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Muat detektor wajah yang sudah dilatih
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         'haarcascade_frontalface_default.xml')
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Kembalikan wajah pertama yang ditemukan, atau None
    if len(faces) > 0:
        return faces[0]  # (x, y, w, h)
    return None

def get_forehead_roi(face_rect, frame):
    """
    Dapatkan area dahi untuk sinyal rPPG.
    
    Parameter
    ----------
    face_rect : tuple
        (x, y, w, h) koordinat wajah
    frame : numpy.ndarray
        Frame video input
        
    Returns
    -------
    tuple
        (roi, (x, y, w, h)) - Data ROI dan koordinatnya
    """
    if face_rect is None:
        return None
    
    x, y, w, h = face_rect
    
    # Tentukan area dahi (sekitar 1/3 bagian atas wajah)
    forehead_h = int(h * 0.3)
    forehead_y = y + int(h * 0.1)  # Offset dari bagian atas wajah
    
    # Ekstrak ROI
    roi = frame[forehead_y:forehead_y+forehead_h, x:x+w]
    roi_coords = (x, forehead_y, w, forehead_h)
    
    return roi, roi_coords

def get_chest_roi(face_rect, frame):
    """
    Dapatkan area dada untuk sinyal respirasi.
    
    Parameter
    ----------
    face_rect : tuple
        (x, y, w, h) koordinat wajah
    frame : numpy.ndarray
        Frame video input
        
    Returns
    -------
    tuple
        (roi, (x, y, w, h)) - Data ROI dan koordinatnya
    """
    if face_rect is None:
        return None
    
    x, y, w, h = face_rect
    frame_h, frame_w = frame.shape[:2]
    
    # Perkirakan posisi dada berdasarkan posisi wajah
    chest_h = int(h * 1.5)
    chest_w = int(w * 1.5)
    chest_x = max(0, x + w//2 - chest_w//2)
    chest_y = min(frame_h - chest_h, y + h + h//4)  # Di bawah wajah
    
    # Pastikan ROI berada dalam batas frame
    chest_x = max(0, min(chest_x, frame_w - chest_w))
    chest_y = max(0, min(chest_y, frame_h - chest_h))
    
    # Ekstrak ROI
    roi = frame[chest_y:chest_y+chest_h, chest_x:chest_x+chest_w]
    roi_coords = (chest_x, chest_y, chest_w, chest_h)
    
    return roi, roi_coords