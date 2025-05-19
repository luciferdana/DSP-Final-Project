#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul untuk memproses video dan mendeteksi area yang diinginkan.
"""

import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi model MediaPipe (jika menggunakan MediaPipe)
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Inisialisasi detektor wajah dan face mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    USE_MEDIAPIPE = True
except ImportError:
    # Fallback ke OpenCV jika MediaPipe tidak tersedia
    USE_MEDIAPIPE = False

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
    if USE_MEDIAPIPE:
        # Deteksi dengan MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        # Mengambil landmark wajah
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Mencari batas kotak dari landmark
        h, w, _ = frame.shape
        x_min = w
        y_min = h
        x_max = 0
        y_max = 0
        
        # Mencari koordinat minimum dan maksimum dari landmark wajah
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Menambahkan margin
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        # Mengembalikan (x, y, width, height)
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    else:
        # Fallback ke OpenCV Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                           'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
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
    
    if USE_MEDIAPIPE:
        # Deteksi dengan MediaPipe landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # Fallback ke metode berbasis wajah jika landmarks tidak terdeteksi
            x, y, w, h = face_rect
            forehead_h = int(h * 0.3)
            forehead_y = y + int(h * 0.1)
            roi = frame[forehead_y:forehead_y+forehead_h, x:x+w]
            roi_coords = (x, forehead_y, w, forehead_h)
            return roi, roi_coords
        
        # Index landmark untuk dahi
        forehead_landmarks = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]
        
        h, w, _ = frame.shape
        x_min = w
        y_min = h
        x_max = 0
        y_max = 0
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Mencari batas kotak dari landmark dahi
        for idx in forehead_landmarks:
            landmark = landmarks[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Menambahkan margin
        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        # Mengekstrak ROI
        roi = frame[y_min:y_max, x_min:x_max]
        roi_coords = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        return roi, roi_coords
    else:
        # Metode berbasis wajah
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
    
    # PERBAIKAN: Posisikan ROI dada lebih ke bawah dan lebih besar
    chest_w = int(w * 1.8)                # Lebih lebar dari wajah
    chest_h = int(h * 1.2)                # Tinggi area dada
    chest_x = max(0, x + w//2 - chest_w//2)  # Tengah dada sejajar dengan tengah wajah
    
    # PERBAIKAN: Posisikan ROI lebih jauh di bawah wajah
    chest_y = min(frame_h - chest_h, y + h + h)  # Perkiraan posisi dada (di bawah wajah)
    
    # Pastikan ROI berada dalam batas frame
    chest_x = max(0, min(chest_x, frame_w - chest_w))
    chest_y = max(0, min(chest_y, frame_h - chest_h))
    
    # Ekstrak ROI
    roi = frame[chest_y:chest_y+chest_h, chest_x:chest_x+chest_w]
    roi_coords = (chest_x, chest_y, chest_w, chest_h)
    
    return roi, roi_coords

def draw_face_landmarks(frame):
    """
    Menggambar landmark wajah pada frame (untuk visualisasi dan debugging).
    
    Parameter
    ----------
    frame : numpy.ndarray
        Frame video input
        
    Returns
    -------
    numpy.ndarray
        Frame dengan landmark wajah yang digambar
    """
    if not USE_MEDIAPIPE:
        return frame
        
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Convert back to BGR for display
    output_frame = frame.copy()
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    
    return output_frame