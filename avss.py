# This is the detection code for the AVSS Project
# Uses input video footage and detects the presents of vehicles and potholes
# The GUI used to perform the functions
import tkinter as tk
import cv2
from ultralytics import YOLO
import supervision as sv
vehicle_detected=False
pothole_detected=False
# pothole_test.mp4
# Definition of detect function
def detect():
    # Store the video path
    video_path = 'Test_Videos/pothole_test.mp4'
    cap = cv2.VideoCapture(video_path)
    # Load the model best.pt
    model = YOLO("best.pt")
    # Create bounding boxes around the objects
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    # Definition of run_process function
    def run_process():
        # Read the frame from video
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        
        labels = [
            f"{'Vehicle' if class_id in range(1,6) and confidence > 0.4 else 'Pothole' if class_id == 0 and confidence > 0.6 else ''}"
            for _, confidence, class_id, _
            in detections
        ]        
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        frame = cv2.resize(frame, (800, 600))
        cv2.imshow("yolov8", frame)
        print(labels)
        
        # Update Headlight box
        if "Vehicle" in labels:
            light_canvas.config(bg="yellow")
        else:
            light_canvas.config(bg="white")
        # Update Pothole alert box
        if "Pothole" in labels:
            pothole_canvas.config(bg="red")
        else:
            pothole_canvas.config(bg="white")
        
        # Quit the program by pressing q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        root.after(100, run_process)
    # Call run process() function
    run_process()


root = tk.Tk()
root.title("Light and Pothole Alert")

heading_label = tk.Label(root, text="AVSS", font=("Arial", 24))
heading_label.pack()

light_canvas = tk.Canvas(root, width=100, height=100, bg="white")
light_canvas.pack()
light_canvas.create_text(50, 50, text="Light")

pothole_canvas = tk.Canvas(root, width=100, height=100, bg="green")
pothole_canvas.pack()
pothole_canvas.create_text(50, 50, text="Pothole Alert")

# Button to start the process to run model
# Call the detect function
start_button = tk.Button(root, text="Start", command=detect, font=("Arial", 14), bg="green", fg="white")
start_button.pack()

# Add the labels to identify the functions
status_label = tk.Label(root, text="White : Bright light", font=("Arial", 10))
status_label.pack()
status_label = tk.Label(root, text="Yellow : Dim light", font=("Arial", 10))
status_label.pack()
status_label = tk.Label(root, text="Green : Reduce speed", font=("Arial", 10))
status_label.pack()
status_label = tk.Label(root, text="Red : Reduce speed", font=("Arial", 10))
status_label.pack()

root.geometry("400x400")
root.mainloop()






# Notes

# ret, frame = cap.read(): This line reads a frame from a video capture device or file.
# The cap.read() method returns two values: ret, which is a boolean indicating whether the frame was successfully read,
# and frame, which is the actual frame data.

# result = model(frame, agnostic_nms=True)[0]: This line passes the frame to the YOLOv8 model for object detection. 
# The model is likely an instance of the YOLOv8 model, and the agnostic_nms=True argument specifies that non-maximum suppression (NMS)
# should be performed in an agnostic manner, meaning that it will be applied to all classes simultaneously.
# The [0] at the end of the line is used to access the first element of the result, which is a list of detected objects.
