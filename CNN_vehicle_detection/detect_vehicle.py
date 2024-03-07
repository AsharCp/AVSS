import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the vehicle model
Vehicle_model = tf.keras.models.load_model("/content/drive/MyDrive/VSS PROJET/Models/model_vehiscan.h5")

# Load the image
image_path = "/content/drive/MyDrive/VSS PROJET/Test Images/bus2.jpeg"
# cap = cv2.VideoCapture('/content/drive/MyDrive/VSS PROJET/Video/Car moving on a Highway.mp4')
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Resize the image
H, W = 224, 224
resized_image = cv2.resize(image, (W, H))
image_array = np.asarray(resized_image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# Predict the vehicle type
predicted_vehicle = Vehicle_model.predict(data)
index = np.argmax(predicted_vehicle)

# Map the index to vehicle type
if index == 0:
    vehicle_type = "auto"
elif index == 1:
    vehicle_type = "bike"
elif index == 2:
    vehicle_type = "bus"
elif index == 3:
    vehicle_type = "car"

# Draw the text on the image
font = cv2.FONT_HERSHEY_SIMPLEX
text = f"{vehicle_type}"
color = (0, 0, 255)  # Red color
thickness = 2
cv2.putText(image, text, (10, 30), font, 1, color, thickness, cv2.LINE_AA)

# Convert the image from BGR to RGB (Matplotlib expects images in RGB format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(image_rgb)
plt.axis('off')  # Hide axis labels
plt.show()
