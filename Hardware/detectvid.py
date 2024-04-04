# Detection using the input video footages
# This is for testing purpose
from ultralytics import YOLO
import supervision as sv
import cv2
import gpiod
import time
vehicle_delay = 10
pothole_delay = 10

def main():
    vehicle_frame_count=vehicle_delay
    pothole_frame_count=pothole_delay
    chip = gpiod.Chip('gpiochip4')
    led_line_bright = chip.get_line(18)
    led_line_dim = chip.get_line(21)
    led_line_pothole = chip.get_line(20)
    led_line_bright.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
    led_line_dim.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
    led_line_pothole.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
    led_line_bright.set_value(1)
    video_path = 'Test_Videos/Night1.mp4'
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
        vehicle_status = "detected" if any(class_id in range(1, 6) and confidence > 0.5 for _, confidence, class_id, _ in detections) else "not_detected"
        pothole_status = "detected" if any(class_id == 0 and confidence > 0.6 for _, confidence, class_id, _ in detections) else "not_detected"
        # Control the vehicle headlight
        if vehicle_status=="detected":
            vehicle_frame_count = vehicle_delay;
            led_line_bright.set_value(0)
            led_line_dim.set_value(1)
        elif vehicle_status=="not_detected":
            if vehicle_frame_count==0:
                led_line_dim.set_value(0)
                led_line_bright.set_value(1)
            else:
                vehicle_frame_count=vehicle_frame_count-1
           
        # Control pothole alert signal 
        if pothole_status=="detected":
            pothole_frame_count=pothole_delay
            led_line_pothole.set_value(1)
        elif pothole_status=="not_detected":
            if pothole_frame_count==0:
                led_line_pothole.set_value(0)
            else:
                pothole_frame_count=pothole_frame_count-1
            

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        
        # Resize the frame to reduce the size of the screen
        frame = cv2.resize(frame, (800, 600))
        cv2.imshow("yolov8", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            led_line_pothole.set_value(0)
            led_line_bright.set_value(0)
            led_line_dim.set_value(0)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
