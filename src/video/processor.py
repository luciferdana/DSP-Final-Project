#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul untuk memproses video dan mendeteksi area yang diinginkan.
Menggunakan BlazeFace dan Pose Landmarker dari folder models/.
"""

import cv2
import numpy as np
import mediapipe as mp
import os

class VideoProcessor:
    """Kelas untuk memproses video dan mendeteksi ROI dengan MediaPipe models."""
    
    def __init__(self):
        """
        Inisialisasi processor dengan model MediaPipe dari folder models/.
        """
        # Path ke model files di direktori models/
        self.blaze_face_model = "models/blaze_face_short_range.tflite"
        self.pose_model = "models/pose_landmarker.task"
        
        # Verifikasi file model ada
        if not os.path.exists(self.blaze_face_model):
            print(f"Warning: BlazeFace model tidak ditemukan di {self.blaze_face_model}")
        if not os.path.exists(self.pose_model):
            print(f"Warning: Pose model tidak ditemukan di {self.pose_model}")
        
        # Inisialisasi MediaPipe solutions
        mp_face_detection = mp.solutions.face_detection
        mp_pose = mp.solutions.pose
        
        try:
            # Inisialisasi BlazeFace detector
            self.face_detector = mp_face_detection.FaceDetection(
                model_selection=0,  # 0 untuk short-range model
                min_detection_confidence=0.5
            )
            
            # Inisialisasi Pose detector
            self.pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_pose = mp_pose
            self.mp_face_detection = mp_face_detection
            
            print("MediaPipe models berhasil dimuat dari direktori models/")
            
        except Exception as e:
            print(f"Error loading MediaPipe models: {e}")
            # Fallback ke OpenCV jika MediaPipe gagal
            self.use_opencv_fallback = True
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("Menggunakan OpenCV Haar Cascade sebagai fallback")
        else:
            self.use_opencv_fallback = False
    
    def detect_face(self, frame):
        """
        Mendeteksi wajah menggunakan BlazeFace atau OpenCV fallback.
        
        Parameter
        ----------
        frame : numpy.ndarray
            Frame video input
            
        Returns
        -------
        tuple atau None
            (x, y, w, h) koordinat wajah, atau None jika tidak ada wajah terdeteksi
        """
        if self.use_opencv_fallback:
            # Fallback ke OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                return faces[0]  # (x, y, w, h)
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame dengan BlazeFace
        results = self.face_detector.process(rgb_frame)
        
        if not results.detections:
            return None
        
        # Ambil deteksi wajah pertama
        detection = results.detections[0]
        
        # Get bounding box
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        
        # Convert relative coordinates to absolute coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Ensure coordinates are within frame bounds
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        return (x, y, width, height)
    
    def get_forehead_roi(self, face_rect, frame):
        """
        Dapatkan area dahi untuk sinyal rPPG berdasarkan deteksi wajah.
        ROI diperbesar untuk stabilitas sinyal yang lebih baik.
        
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
        
        # Area dahi yang diperbesar untuk stabilitas rPPG yang lebih baik
        forehead_height = int(h * 0.35)  # 35% tinggi wajah (lebih besar)
        forehead_width = int(w * 0.9)    # 90% lebar wajah (lebih lebar)
        
        # Posisi dahi (centered horizontally, di bagian atas wajah)
        forehead_x = x + int((w - forehead_width) / 2)
        forehead_y = y + int(h * 0.08)   # Sedikit lebih tinggi untuk menghindari alis
        
        # Pastikan koordinat dalam batas frame
        frame_h, frame_w = frame.shape[:2]
        forehead_x = max(0, min(forehead_x, frame_w - forehead_width))
        forehead_y = max(0, min(forehead_y, frame_h - forehead_height))
        
        # Extract ROI
        roi = frame[forehead_y:forehead_y+forehead_height, 
                   forehead_x:forehead_x+forehead_width]
        roi_coords = (forehead_x, forehead_y, forehead_width, forehead_height)
        
        return roi, roi_coords
    
    def get_chest_roi(self, frame):
        """
        Dapatkan area dada untuk sinyal respirasi menggunakan pose detection.
        Prioritaskan area kulit yang terlihat untuk akurasi yang lebih baik.
        
        Parameter
        ----------
        frame : numpy.ndarray
            Frame video input
            
        Returns
        -------
        tuple
            (roi, (x, y, w, h)) - Data ROI dan koordinatnya, atau None jika gagal
        """
        if self.use_opencv_fallback:
            return None  # Tidak bisa deteksi pose dengan OpenCV fallback
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Process the frame
            results = self.pose_detector.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # Landmark indices untuk area dada dan leher
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Gunakan area leher untuk respirasi yang lebih akurat (area kulit)
            try:
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
                
                # Koordinat dalam pixel
                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)
                left_shoulder_x = int(left_shoulder.x * w)
                left_shoulder_y = int(left_shoulder.y * h)
                right_shoulder_x = int(right_shoulder.x * w)
                right_shoulder_y = int(right_shoulder.y * h)
                
                # Area leher/chest atas (area kulit yang lebih responsif)
                chest_center_x = (left_shoulder_x + right_shoulder_x) // 2
                chest_center_y = (nose_y + left_shoulder_y + right_shoulder_y) // 3
                
                # Ukuran ROI yang lebih kecil tapi fokus pada area kulit
                chest_width = abs(right_shoulder_x - left_shoulder_x)
                chest_height = int(chest_width * 0.6)  # Rasio yang proporsional
                
                # Posisi ROI
                chest_x = max(0, chest_center_x - chest_width // 2)
                chest_y = max(0, chest_center_y - chest_height // 4)
                
                # Pastikan tidak keluar batas frame
                chest_x = min(chest_x, w - chest_width)
                chest_y = min(chest_y, h - chest_height)
                
                # Extract ROI
                roi = frame[chest_y:chest_y+chest_height, chest_x:chest_x+chest_width]
                roi_coords = (chest_x, chest_y, chest_width, chest_height)
                
                return roi, roi_coords
                
            except:
                # Fallback ke metode shoulder-hip jika nose detection gagal
                left_shoulder_x = int(left_shoulder.x * w)
                left_shoulder_y = int(left_shoulder.y * h)
                right_shoulder_x = int(right_shoulder.x * w)
                right_shoulder_y = int(right_shoulder.y * h)
                
                # Area dada berdasarkan bahu
                chest_x_min = min(left_shoulder_x, right_shoulder_x)
                chest_x_max = max(left_shoulder_x, right_shoulder_x)
                chest_y_min = min(left_shoulder_y, right_shoulder_y)
                
                # ROI parameters
                margin = 20
                chest_x = max(0, chest_x_min - margin)
                chest_y = max(0, chest_y_min + margin)
                chest_width = min(w - chest_x, chest_x_max - chest_x_min + 2*margin)
                chest_height = min(h - chest_y, int(chest_width * 0.8))
                
                # Pastikan ROI valid
                if chest_width <= 0 or chest_height <= 0:
                    return None
                
                # Extract ROI
                roi = frame[chest_y:chest_y+chest_height, chest_x:chest_x+chest_width]
                roi_coords = (chest_x, chest_y, chest_width, chest_height)
                
                return roi, roi_coords
            
        except Exception as e:
            print(f"Error dalam pose detection: {e}")
            return None
    
    def get_chest_roi_fallback(self, face_rect, frame):
        """
        Fallback method untuk ROI dada berdasarkan deteksi wajah.
        Digunakan jika pose detection gagal.
        
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
        
        # Estimasi posisi dada berdasarkan wajah dengan fokus pada area leher
        chest_w = int(w * 1.2)                # Sedikit lebih lebar dari wajah
        chest_h = int(h * 0.8)                # Tinggi area leher/chest atas
        chest_x = max(0, x + w//2 - chest_w//2)  # Tengah sejajar dengan wajah
        chest_y = min(frame_h - chest_h, y + int(h * 1.1))  # Tepat di bawah wajah (area leher)
        
        # Pastikan ROI berada dalam batas frame
        chest_x = max(0, min(chest_x, frame_w - chest_w))
        chest_y = max(0, min(chest_y, frame_h - chest_h))
        
        # Extract ROI
        roi = frame[chest_y:chest_y+chest_h, chest_x:chest_x+chest_w]
        roi_coords = (chest_x, chest_y, chest_w, chest_h)
        
        return roi, roi_coords
    
    def draw_landmarks(self, frame, draw_pose=True, draw_face=True):
        """
        Menggambar landmarks pada frame untuk visualisasi.
        
        Parameter
        ----------
        frame : numpy.ndarray
            Frame video input
        draw_pose : bool
            Apakah menggambar pose landmarks
        draw_face : bool
            Apakah menggambar face detection
            
        Returns
        -------
        numpy.ndarray
            Frame dengan landmarks yang digambar
        """
        output_frame = frame.copy()
        
        if self.use_opencv_fallback:
            return output_frame  # Tidak ada landmarks untuk digambar dengan OpenCV
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            if draw_pose:
                # Draw pose landmarks
                pose_results = self.pose_detector.process(rgb_frame)
                if pose_results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        output_frame,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
            
            if draw_face:
                # Draw face detection
                face_results = self.face_detector.process(rgb_frame)
                if face_results.detections:
                    for detection in face_results.detections:
                        self.mp_drawing.draw_detection(output_frame, detection)
        except Exception as e:
            print(f"Error menggambar landmarks: {e}")
        
        return output_frame
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'face_detector') and not self.use_opencv_fallback:
                self.face_detector.close()
            if hasattr(self, 'pose_detector') and not self.use_opencv_fallback:
                self.pose_detector.close()
        except:
            pass


# Fungsi wrapper untuk kompatibilitas dengan kode yang sudah ada
_processor = None

def get_processor():
    """Get singleton processor instance."""
    global _processor
    if _processor is None:
        _processor = VideoProcessor()
    return _processor

def detect_face(frame):
    """Wrapper function untuk deteksi wajah."""
    return get_processor().detect_face(frame)

def get_forehead_roi(face_rect, frame):
    """Wrapper function untuk mendapatkan ROI dahi."""
    return get_processor().get_forehead_roi(face_rect, frame)

def get_chest_roi(face_rect, frame):
    """
    Wrapper function untuk mendapatkan ROI dada.
    Mencoba pose detection dulu, fallback ke face-based jika gagal.
    """
    processor = get_processor()
    
    # Coba gunakan pose detection
    chest_roi = processor.get_chest_roi(frame)
    
    # Jika gagal, gunakan fallback berdasarkan wajah
    if chest_roi is None:
        chest_roi = processor.get_chest_roi_fallback(face_rect, frame)
    
    return chest_roi

def draw_face_landmarks(frame):
    """Wrapper function untuk menggambar landmarks."""
    return get_processor().draw_landmarks(frame, draw_pose=False, draw_face=True)