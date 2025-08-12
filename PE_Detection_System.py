from ultralytics import YOLO # type: ignore
from config import Config
from utils import calculate_iou
import cv2 # type: ignore
import numpy as np # type: ignore
from utils import get_timestamp
from utils import create_directories
import os
import sys

class ObjectDetector:
    def __init__(self, model_path=None, confidence_threshold=None):
        self.model_path = model_path or Config.MODEL_PATH
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD

        self.model = YOLO(self.model_path)
        print(f"Loaded model: {self.model_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")

    def detect_objects(self, frame):
        results = self.model.track(frame, conf=self.confidence_threshold, tracker="bytetrack.yaml", persist=True, verbose=False)
        return results
    
    def extract_detections(self, results):
        persons = []
        helmets = []
        boots = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None

                detection = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class_id': class_id,
                    'track_id': track_id
                }

                if class_id == Config.PERSON_CLASS:
                    persons.append(detection)
                elif class_id == Config.HELMET_CLASS:
                    helmets.append(detection)
                elif class_id == Config.BOOT_CLASS:
                    boots.append(detection)
        
        return persons, helmets, boots

class ComplainceAnalyzer:
    def __init__(self, iou_threshold = None):
        self.iou_threshold = iou_threshold or Config.IOU_THRESHOLD
        self.worker_id_counter = 1

    def is_ppe_on_person(self, person_box, ppe_bbox):
        if calculate_iou(person_box, ppe_bbox) > self.iou_threshold:
            return True
        
        px1, py1, px2, py2 = person_box
        ppx1, ppy1, ppx2, ppy2 = ppe_bbox

        tolerance = 0.1 * max(px2 - px1, py2 - py1)
        expanded_px1 = px1 - tolerance
        expanded_py1 = py1 - tolerance
        expanded_px2 = px2 + tolerance  
        expanded_py2 = py2 + tolerance

        ppe_center_x = (ppx1 + ppx2) / 2
        ppe_center_y = (ppy1 + ppy2) / 2

        return (expanded_px1 <= ppe_center_x <= expanded_px2 and expanded_py1 <= ppe_center_y <= expanded_py2)
    
    def analyze_ppe_complaince(self, persons, helmets, boots):
        compliance_results = []
        for i, person in enumerate(persons):
            person_bbox = person['bbox']

            has_helmet = False
            associated_helmet = None

            has_boots = False
            associated_boots = None

            for helmet in helmets:
                helmet_bbox = helmet["bbox"]
                if self.is_ppe_on_person(person_bbox, helmet_bbox):
                    has_helmet = True  
                    associated_helmet = helmet
                    break

            for boot in boots:
                boot_bbox = boot["bbox"]
                if self.is_ppe_on_person(person_bbox, boot_bbox):
                    has_boots = True
                    associated_boots = boot
                    break

            compliance_results.append({
                'person': person,
                'has_helmet': has_helmet,
                'has_boots': has_boots,
                'helmet_detection': associated_helmet,
                'boots_detection': associated_boots,
                'is_complaint': has_helmet and has_boots
            })
        
        return compliance_results

class SnapshotManager:
    def __init__(self):
        self.snapshot_dir = Config.SNAPSHOTS_DIR
        os.makedirs(self.snapshot_dir, exist_ok = True)
        self.padding = Config.PADDING
        self.last_snapshot_frame = None
        self.change_threshold = Config.PIXEL_PERCENTAGE_THRESHOLD

    def has_significant_change(self, frame1, frame2):
        if frame1 is None or frame2 is None:
            return True
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        non_zero_count = np.count_nonzero(diff)

        change_percent = (non_zero_count / diff.size) * 100
        return change_percent > self.change_threshold

    def save_snapshot(self, frame, complaince_data):
        snapshots_paths = []
        timestamp = get_timestamp()

        date_dir = os.path.join(self.snapshot_dir, timestamp["date_str"])
        os.makedirs(date_dir, exist_ok = True)

        to_unblur_all = []
        save_snapshot_flag = False

        for data in complaince_data:
            if not data["is_complaint"]:
                save_snapshot_flag = True
                if not data["has_helmet"]:
                    if data["helmet_detection"]:
                        to_unblur_all.append(data["helmet_detection"]["bbox"])
                    else:
                        person_bbox = data["person"]["bbox"]
                        head_x1 = person_bbox[0] + int(0.2 * (person_bbox[2] - person_bbox[0]))
                        head_y1 = person_bbox[1]                            
                        head_x2 = person_bbox[2] - int(0.2 * (person_bbox[2] - person_bbox[0]))
                        head_y2 = person_bbox[1] + int(0.3 * (person_bbox[3] - person_bbox[1]))
                        to_unblur_all.append((head_x1, head_y1, head_x2, head_y2))

                if not data["has_boots"]:
                    if data["boots_detection"]:
                        to_unblur_all.append(data["boots_detection"]["bbox"])   
                    else:
                        person_bbox = data["person"]["bbox"]
                        foot_x1 = person_bbox[0] + int(0.1 * (person_bbox[2] - person_bbox[0])) 
                        foot_y1 = person_bbox[3] - int(0.2 * (person_bbox[3] - person_bbox[1]))
                        foot_x2 = person_bbox[2] - int(0.1 * (person_bbox[2] - person_bbox[0]))
                        foot_y2 = person_bbox[3] 
                        to_unblur_all.append((foot_x1, foot_y1, foot_x2, foot_y2))

        if save_snapshot_flag and to_unblur_all:
            blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)
            for (x1, y1, x2, y2) in to_unblur_all:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                blurred_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

            if self.has_significant_change(blurred_frame, self.last_snapshot_frame):
                filename = f"ppe_snapshot_{timestamp['filename_timestamp']}.jpg"
                full_path =  os.path.join(date_dir, filename)
                cv2.imwrite(full_path, blurred_frame)

                snapshots_paths.append(filename)
                self.last_snapshot_frame = blurred_frame.copy()
                print(f"Snapshot saved for non-compliant worker: {filename}")
            else:
                snapshots_paths.append("No significant change - skipped")

        else:
            snapshots_paths.append("N/A")
        return snapshots_paths
    
class PPEVisualizer:
    def __init__(self):
        self.colors = Config.COLORS
        self.box_thickness = Config.BOX_THICKNESS
        self.font_scale = Config.FONT_SCALE
        self.font_thickness = Config.FONT_THICKNESS

    def draw_bounding_boxes(self, frame, compliance_data):
        annotated_frame = frame.copy()

        for data in compliance_data:
            person_bbox = data['person']['bbox']  

            if not data['has_helmet']:
                head_x1 = person_bbox[0] + int(0.2 * (person_bbox[2] - person_bbox[0]))
                head_y1 = person_bbox[1]
                head_x2 = person_bbox[2] - int(0.2 * (person_bbox[2] - person_bbox[0]))
                head_y2 = person_bbox[1] + int(0.3 * (person_bbox[3] - person_bbox[1]))

                cv2.rectangle(annotated_frame, (head_x1, head_y1), 
                            (head_x2, head_y2), 
                            self.colors["RED_BOX"], 
                            self.box_thickness)
                
                cv2.putText(annotated_frame, 'NO Helmet',
                            (head_x1, head_y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale, self.colors['RED_BOX'], self.font_thickness)
                
            if not data['has_boots']:
                foot_x1 = person_bbox[0] + int(0.1 * (person_bbox[2] - person_bbox[0])) 
                foot_y1 = person_bbox[3] - int(0.2 * (person_bbox[3] - person_bbox[1]))
                foot_x2 = person_bbox[2] - int(0.1 * (person_bbox[2] - person_bbox[0]))
                foot_y2 = person_bbox[3] 

                cv2.rectangle(annotated_frame, (foot_x1, foot_y1), 
                            (foot_x2, foot_y2), 
                            self.colors['RED_BOX'],
                            self.box_thickness)
                
                cv2.putText(annotated_frame, 'NO Boots',
                        (foot_x1, foot_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, self.colors['RED_BOX'], self.font_thickness)

        return annotated_frame

class PPEDetectionSystem:
    def __init__(self):

        create_directories([Config.SNAPSHOTS_DIR])

        self.detector = ObjectDetector(Config.MODEL_PATH, Config.CONFIDENCE_THRESHOLD)
        self.complaince_analyzer = ComplainceAnalyzer()
        self.visualizer = PPEVisualizer()
        self.snapshot_manager = SnapshotManager()

        print("PPE Detection System initialized successfully:)")

    def process_frame(self, frame):

        results = self.detector.detect_objects(frame)

        persons, helmets, boots = self.detector.extract_detections(results)

        if not persons:
            return frame, []
        
        compliance_data = self.complaince_analyzer.analyze_ppe_complaince(persons, helmets, boots)

        annotated_frame = self.visualizer.draw_bounding_boxes(frame, compliance_data)

        snapshot_paths = self.snapshot_manager.save_snapshot(annotated_frame, compliance_data)

        compliance_count = sum(1 for data in compliance_data if data['is_complaint'])
        total_count = len(compliance_data)

        tracked_count = sum(1 for person in persons if person['track_id'] is not None)

        print(f"Frame Processed: {compliance_count}/{total_count} workers complaint | Tracked Count: {tracked_count}/{len(persons)}")

        return annotated_frame, compliance_data
    

    def run_video_detection(self):
        cap = cv2.VideoCapture(Config.VIDEO_PATH)

        if not cap.isOpened():
            print("Error opening video source")
            return
        
        out = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*Config.OUTPUT_VIDEO_CODEC)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            video_save_path = os.path.join(Config.SNAPSHOTS_DIR, "output_video.avi")
            out = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

            if not out.isOpened():
                print("Warning: Could not open VideoWriter - output video will not be saved")
                out = None
        except Exception as e:
            print(f"VideeWriter init failed: {e}")
            out = None
        print("Starting PPE Detection System. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, compliance_data = self.process_frame(frame)

            compliance_count = sum(1 for data in compliance_data if data["is_complaint"])
            total_count = len(compliance_data)

            cv2.imshow("PPE Detection System", annotated_frame)

            if Config.SNAPSHOTS_DIR:
                out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        print('PPE Detection System stopped')

    def run_camera_detection(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Starting PPE Detection with Camera. Press 'q' to quit, 's' to save current frame")

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame from camera:( ")
                break

            annotated_frame, compliance_data = self.process_frame(frame)

            frame_count += 1
            timestamp = get_timestamp()

            cv2.putText(annotated_frame, f"Frame: {frame_count} | Time: {timestamp['time_str']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            compliance_count = sum(1 for data in compliance_data if data["is_complaint"])
            total_count = len(compliance_data)

            cv2.imshow("PPE Detection - Live Camera", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = get_timestamp()
                save_path = f"{Config.SNAPSHOTS_DIR}/manual_capture_{timestamp['filename_timestamp']}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"Frame saved manually: {save_path}")

        cap.release()
        cv2.destroyAllWindows()
        print("Camera detection stopped")

def main():
    print("------------Enter 0 for live web camera and 1 for video file-------------")
    source = int(input())

    try:
        ppe_detection = PPEDetectionSystem()

        if source == 1:
            print(f"Starting video processing from source: {Config.VIDEO_PATH}")
            ppe_detection.run_video_detection()
        else:
            print("Starting live camera detection")
            ppe_detection.run_camera_detection()

    except Exception as e:
        import traceback
        print("Exception occurred:", e)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()


        

