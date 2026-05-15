import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# --- ΡΥΘΜΙΣΕΙΣ ΣΥΣΤΗΜΑΤΟΣ ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'data', 'best_thermal.pt') # Ιδανικά βάλε ένα IR YOLO model
VIDEO_PATH = os.path.join(BASE_DIR, 'data', 'test_ir.mp4') 

# Ο ΔΙΑΚΟΠΤΗΣ: Άλλαξε το σε 'RGB' για κανονική κάμερα, ή 'IR' για θερμική
CAMERA_MODE = 'IR' 

class HybridFireTracker:
    def __init__(self, weights_path=WEIGHTS_PATH):
        try:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Λείπει το αρχείο μοντέλου: {weights_path}")
                
            self.model = YOLO(weights_path)
            print(f"[SUCCESS] Το AI μοντέλο φορτώθηκε από: {weights_path}")
            print(f"[INFO] Camera Mode: {CAMERA_MODE}")
        except Exception as e:
            print(f"[ERROR] Αποτυχία φόρτωσης YOLO: {e}")

        self.area_history = [] 
        self.history_length = 15 

    # ==========================================
    # ΥΠΟ-ΣΥΣΤΗΜΑ 1: THERMAL / IR ΜΑΘΗΜΑΤΙΚΑ
    # ==========================================
    def thermal_segmentation(self, crop):
        """
        Η φωτιά σε μια IR κάμερα (συνήθως 'White Hot') είναι απλά τα πιο φωτεινά pixels.
        Αγνοούμε τα χρώματα και ψάχνουμε για ακραία ένταση φωτός/θερμότητας.
        """
        # Μετατροπή σε Grayscale αν η κάμερα στέλνει pseudo-color (π.χ. Ironbow)
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
            
        # Ελαφρύ blur για αφαίρεση του θορύβου του αισθητήρα (IR κάμερες έχουν πολύ θόρυβο)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Κρατάμε μόνο τα "καυτά" pixels (πάνω από 230 στο 0-255 scale)
        # Προσοχή: Αυτό το 230 μπορεί να χρειαστεί πείραγμα ανάλογα με την κάμερά σου.
        _, thermal_mask = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)
        
        return thermal_mask

    # ==========================================
    # ΥΠΟ-ΣΥΣΤΗΜΑ 2: RGB ΜΑΘΗΜΑΤΙΚΑ (BRAVONOID)
    # ==========================================
    def apply_rgb_rules(self, image):
        B, G, R = cv2.split(image)
        R_f, G_f, B_f = R.astype(np.float32), G.astype(np.float32), B.astype(np.float32)
        R_denom, G_denom = R_f + 1.0, G_f + 1.0

        rule1 = (R > G) & (G > B)
        rule2 = (R > 190) & (G > 90) & (B < 140)
        
        rule3_1 = (0.1 <= (G_f / R_denom)) & ((G_f / R_denom) <= 1.0)
        rule3_2 = (0.1 <= (B_f / R_denom)) & ((B_f / R_denom) <= 0.85)
        rule3_3 = (0.1 <= (B_f / G_denom)) & ((B_f / G_denom) <= 0.85)
        rule3 = rule3_1 & rule3_2 & rule3_3
        
        rgb_mask = rule1 & rule2 & rule3
        
        hsv_crop = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat_mask = hsv_crop[:, :, 1] > 50 
        return np.where(rgb_mask & sat_mask, 255, 0).astype(np.uint8)

    def full_bravonoid_segmentation(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        low_light_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        rgb_img = self.apply_rgb_rules(crop)
        rgb_low = self.apply_rgb_rules(low_light_img)

        # (Απλοποιημένο για ταχύτητα: Κρατάμε μόνο τους RGB + Saturation κανόνες)
        return cv2.bitwise_or(rgb_img, rgb_low)

    # ==========================================
    # ΚΕΝΤΡΙΚΗ ΛΟΓΙΚΗ ΕΠΕΞΕΡΓΑΣΙΑΣ
    # ==========================================
    def process_frame(self, frame):
        results = self.model.predict(source=frame, imgsz=320, conf=0.3, verbose=False)
        fire_polygon = None
        fire_center = None
        optical_growth_rate = 0.0
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id].lower()
                
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                h, w = frame.shape[:2]
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                if 'fire' in class_name: 
                    fire_crop = frame[y_min:y_max, x_min:x_max]
                    
                    # --- Η ΜΕΓΑΛΗ ΑΛΛΑΓΗ ΤΟΥ IR ---
                    if CAMERA_MODE == 'IR':
                        local_mask = self.thermal_segmentation(fire_crop)
                    else:
                        local_mask = self.full_bravonoid_segmentation(fire_crop)
                    # -------------------------------
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel)
                    local_mask = cv2.dilate(local_mask, kernel, iterations=2)
                    
                    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_local_contour = max(contours, key=cv2.contourArea)
                        current_area = cv2.contourArea(largest_local_contour)
                        
                        if current_area > 50:
                            fire_polygon = largest_local_contour + np.array([x_min, y_min])
                            
                            current_time = time.time()
                            self.area_history.append((current_time, current_area))
                            
                            if len(self.area_history) > self.history_length:
                                self.area_history.pop(0)
                                
                            if len(self.area_history) == self.history_length:
                                past_time, past_area = self.area_history[0]
                                dt = current_time - past_time
                                da = current_area - past_area
                                if dt > 0:
                                    optical_growth_rate = da / dt
                            
                            M = cv2.moments(largest_local_contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"]) + x_min
                                cy = int(M["m01"] / M["m00"]) + y_min
                                fire_center = (cx, cy)
                                
                                cv2.circle(frame, fire_center, 5, (0, 0, 255), -1)
                                color = (0, 0, 255) if optical_growth_rate > 0 else (255, 255, 0)
                                sign = "+" if optical_growth_rate > 0 else ""
                                text = f"Growth: {sign}{int(optical_growth_rate)} px/s"
                                cv2.putText(frame, text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                            cv2.drawContours(frame, [fire_polygon], -1, (0, 255, 0), 2)

                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (150, 150, 150), 2)
                    cv2.putText(frame, class_name.upper(), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

        return frame, fire_polygon, fire_center, optical_growth_rate

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"[FATAL ERROR] Το βίντεο δεν βρέθηκε στο: {VIDEO_PATH}")
        exit()

    tracker = HybridFireTracker(WEIGHTS_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    mock_drone_telemetry = {"gps_lat": 38.2466, "gps_lon": 21.7346, "altitude_m": 120.0, "gimbal_pitch": -45.0}
    prev_time = 0
    window_name = f'Tactical Edge Tracker ({CAMERA_MODE} Mode)'

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret: break

        tracked_frame, polygon, fire_center, optical_growth_rate = tracker.process_frame(current_frame)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        cv2.putText(tracked_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if fire_center:
            print(f"[TARGET LOCKED] Pixel: {fire_center} | Growth: {int(optical_growth_rate)} px/s | Alt: {mock_drone_telemetry['altitude_m']}m")

        cv2.imshow(window_name, tracked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()