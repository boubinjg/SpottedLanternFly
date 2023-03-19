from ultralytics import YOLO
import cv2 as cv
import os
import time
import queue
import threading
import supervision as sv

q = queue.Queue()

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport:udp"

model = YOLO("slf.pt")

def Receive():
    print("start Receive")
    cap = cv.VideoCapture("rtsp://192.168.53.1/live", cv.CAP_FFMPEG)
    
    ret, frame = cap.read()
    count = 0
    while True:
        ret, frame = cap.read()

        if ret:
            count += 1
            if count % 3 == 0:
                q.put(frame)

def Display():
    while True:
        if q.empty() != True:
            frame = q.get()
            
            infer_result = model.predict(frame, device=0)[0]
            box_annotator = sv.BoxAnnotator(
                thickness=2,
                text_thickness=2,
                text_scale=1
            )
            bbox = sv.Detections.from_yolov8(infer_result)
            frame = box_annotator.annotate(scene=frame, detections=bbox)

            cv.imshow("frame1", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
    
