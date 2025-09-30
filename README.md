Weapon Detection using YOLO11

This project demonstrates real-time weapon detection using the Ultralytics YOLO11 model. The system is trained on a custom dataset in Kaggle and deployed with OpenCV for live webcam inference. It not only detects weapons but also calculates the distance of detected objects from the image center, enabling tracking and analysis.

Features
Train YOLO11 on a custom weapon dataset.
Detect weapons in real-time via webcam.
Bounding boxes with class, confidence score, and distance.
Highlights both image center (yellow dot) and object center (red dot).
Works with Kaggle for training and local PC for inference

Dataset
Custom dataset uploaded to Kaggle (/kaggle/input/weapon-detection).
data.yaml defines training, validation, and test splits.

Training on Kaggle
!ls /kaggle/input/weapon-detection/
!cat /kaggle/input/weapon-detection/data.yaml

# Install Ultralytics
!pip install ultralytics -q

# Train YOLO11 model
!yolo detect train data=/kaggle/input/weapon-detection/data.yaml model=yolo11n.pt epochs=20 imgsz=640

Testing on Test Images
!yolo detect predict model=/kaggle/working/runs/detect/train/weights/best.pt

Real-Time Detection with Webcam
import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("C:/Users/HP/Downloads/Weapon_yolo11/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO inference
    results = model(frame[..., ::-1])
    annotated_frame = results[0].plot()
    
    # Get image center
    h, w = frame.shape[:2]
    cx_img, cy_img = w // 2, h // 2
    cv2.circle(annotated_frame, (cx_img, cy_img), 5, (0, 255, 255), -1)  
    
    # Iterate over detections
    for box in results[0].boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Bounding box center
        cx_box = (xmin + xmax) // 2
        cy_box = (ymin + ymax) // 2
        cv2.circle(annotated_frame, (cx_box, cy_box), 5, (0, 0, 255), -1)
        
        # Distance calculation
        dx, dy = cx_box - cx_img, cy_box - cy_img
        distance = (dx**2 + dy**2)**0.5
        
        text = f"Cls:{cls} Conf:{conf:.2f} Dist:{int(distance)}px dx:{dx} dy:{dy}"
        cv2.putText(annotated_frame, text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(f"Class:{cls}, Conf:{conf:.2f}, Dist:{distance:.2f}px, dx:{dx}, dy:{dy}")
    
    cv2.imshow("YOLO Webcam", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Results
Weapons detected in test images and real-time webcam stream.
Output includes bounding box, class, confidence score, and distance.

Tech Stack
YOLO11 (Ultralytics)
OpenCV
Python
Kaggle (for training)

Future Improvements
Deploy on Jetson Nano / Raspberry Pi for edge AI.
Integrate alert system (e.g., sound, email, or IoT).
Use tracking algorithms (DeepSORT, ByteTrack).

With this setup, the project can be extended for surveillance, security monitoring, and defense applications.
