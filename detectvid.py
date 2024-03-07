# Detection using the input video footages
# This is for testing purpose
import cv2
from ultralytics import YOLO
import supervision as sv

def main():
    video_path = 'test.mp4'
    cap = cv2.VideoCapture(video_path)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
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
        
        # Resize the frame to reduce the size of the screen
        frame = cv2.resize(frame, (800, 600))
        cv2.imshow("yolov8", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
