import cv2
import cvzone
from ultralytics import YOLO, YOLOWorld

# img = cv2.imread('D:/CV/Multiple_Object_Detection/2.png')
# img = cv2.resize(img, (900, 700))
cap = cv2.VideoCapture(r'D:\CV\Multiple_Object_Detection\14750358-hd_1080_1920_30fps.mp4')

# Load all models
models = [
    YOLO('yolo26s.pt'),
    YOLOWorld('yolov8s-world.pt'),
    YOLO('last.pt'),
    YOLO(r'D:\CV\Multiple_Object_Detection\Models\Bridge\best.pt'),
    YOLO(r'D:\CV\Multiple_Object_Detection\Models\Bridge\2best.pt'),
    YOLO(r'D:\CV\Multiple_Object_Detection\Models\Bridge Demage\best.pt'),
    YOLO(r'D:\CV\Multiple_Object_Detection\Models\Desert\best.pt'),
    YOLO(r'D:\CV\Multiple_Object_Detection\Models\forest\best.pt'),
    YOLO(r'D:\CV\Multiple_Object_Detection\Models\sea\best.pt'),
    YOLO(r'D:\CV\Multiple_Object_Detection\Models\store\best.pt'),
    YOLO(r'D:\CV\Multiple_Object_Detection\Models\University\1best.pt'),
    YOLO(r'D:\CV\Multiple_Object_Detection\Models\University\best.pt')
]

while True:
    success, img = cap.read()
    img = cv2.resize(img, (700, 700))
    if not  success:
        break

    # Run all models
    for model in models:

        results = model(img, conf=0.5)

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]

                cvzone.cornerRect(img, (x1, y1, x2, y2))
                cvzone.putTextRect(img,f'{name} {conf:.2f}',(x1, y1 - 10),colorR=(255, 0, 0),scale=0.8, thickness=1)


    cv2.imshow('Object Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()