# Detection of vehicles and pothole using webcam in real time
# No GUI components are added
import cv2
from ultralytics import YOLO
import supervision as sv
frame_width = 800
frame_height = 500

def main():

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        # print("Result",result)
        detections = sv.Detections.from_yolov8(result)
        
  
        labels = [
            f"{'Vehicle' if class_id in range(1,6) and confidence > 0.5 else 'Pothole' if class_id == 0 and confidence > 0.6 else ''}"
            for _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        print(labels)
        cv2.imshow("yolov8", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()

