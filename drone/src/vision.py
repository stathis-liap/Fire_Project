import cv2
import numpy as np
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'data', 'fire_yolov8n.pt')
VIDEO_PATH = os.path.join(BASE_DIR, 'data', 'test2.mp4') 

class HybridFireTracker:
    def __init__(self, weights_path=WEIGHTS_PATH):
        try:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing weights file at: {weights_path}")
                
            self.model = YOLO(weights_path)
            # Εκτύπωση των πραγματικών Classes του μοντέλου για να δεις τι αναγνωρίζει!
            print(f"[SUCCESS] Model loaded. Classes found: {self.model.names}")
        except Exception as e:
            print(f"[ERROR] Could not load YOLO weights: {e}")

    def low_light_enhance(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        hsv[:, :, 2] = cv2.medianBlur(hsv[:, :, 2], 5)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_ycbcr_rules(self, image):
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cb, cr = cv2.split(ycbcr)
        mask = ((y >= 170) | (y < 145)) & (cb <= 120) & (cb >= 50) & (cr > 120) & (cr < 220)
        return np.where(mask, 255, 0).astype(np.uint8)

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
        return np.where(rgb_mask, 255, 0).astype(np.uint8)

    def full_bravonoid_segmentation(self, crop):
        low_light_img = self.low_light_enhance(crop)

        rgb_img = self.apply_rgb_rules(crop)
        rgb_low = self.apply_rgb_rules(low_light_img)

        ycbcr_img = self.apply_ycbcr_rules(crop)
        ycbcr_low = self.apply_ycbcr_rules(low_light_img)

        combined_img = cv2.bitwise_or(rgb_img, ycbcr_img) 
        combined_low = cv2.bitwise_or(rgb_low, ycbcr_low)

        ultimate_combined = cv2.bitwise_and(combined_img, combined_low)
        rgb_combined = cv2.bitwise_or(rgb_img, rgb_low)

        percentage = np.sum(ultimate_combined == rgb_combined) / (crop.shape[0] * crop.shape[1])
        
        if percentage >= 0.75:
            output = rgb_combined
        else:
            output = ultimate_combined

        return output

    def process_frame(self, frame):
        results = self.model.predict(source=frame, conf=0.3, verbose=False)
        fire_polygon = None
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                # Παίρνουμε το όνομα του class από το μοντέλο δυναμικά, σε πεζά γράμματα
                class_name = self.model.names[class_id].lower()
                
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                
                h, w = frame.shape[:2]
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                # --- BRANCH A: FIRE ---
                # Τρέχει μόνο αν η λέξη "fire" υπάρχει στο όνομα της κατηγορίας
                if 'fire' in class_name: 
                    fire_crop = frame[y_min:y_max, x_min:x_max]
                    local_mask = self.full_bravonoid_segmentation(fire_crop)
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel)
                    local_mask = cv2.dilate(local_mask, kernel, iterations=2)
                    
                    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_local_contour = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_local_contour) > 50:
                            fire_polygon = largest_local_contour + np.array([x_min, y_min])
                            
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                            cv2.drawContours(frame, [fire_polygon], -1, (0, 255, 0), 2)
                            cv2.putText(frame, "FIRE", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # --- BRANCH B: SMOKE ---
                # Αν η λέξη είναι "smoke" (ή οτιδήποτε άλλο), απλά βάζει γκρι κουτί
                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (150, 150, 150), 2)
                    cv2.putText(frame, class_name.upper(), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

        return frame, fire_polygon


# --- TEST EXECUTION BLOCK WITH SEEK BAR ---
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"[FATAL ERROR] Video file not found at {VIDEO_PATH}")
        exit()

    tracker = HybridFireTracker(WEIGHTS_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    # 1. Παίρνουμε τον συνολικό αριθμό frames για τη μπάρα
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_name = 'Full Bravonoid Edge Tracker'
    
    cv2.namedWindow(window_name)

    # 2. Callback συνάρτηση για όταν κουνάς το ποντίκι σου στη μπάρα
    def on_trackbar(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    # 3. Δημιουργία του Slider (Trackbar)
    cv2.createTrackbar('Timeline', window_name, 0, total_frames, on_trackbar)

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret: 
            break

        tracked_frame, polygon = tracker.process_frame(current_frame)
        cv2.imshow(window_name, tracked_frame)

        # 4. Ανανέωση της θέσης της μπάρας καθώς παίζει το βίντεο
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('Timeline', window_name, current_pos)

        # Το βίντεο θα παίζει αλλά μπορείς να πατήσεις 'q' για έξοδο
        if cv2.waitKey(30) & 0xFF == ord('q'): 
            break
        
    cap.release()
    cv2.destroyAllWindows()