from tensorflow.keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time  # Import time module for the delay
import serial  # For communication with Arduino

# Set up serial communication with Arduino (adjust the port if necessary)
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcam's image
    ret, image = camera.read()

    # Resize the raw image into (224-height, 224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name)
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Map class names to signals for Arduino
    if class_name == "0 Grasshoppers":
        arduino.write(b'1')  # Send '1' to Arduino
        print("Sent '1' to Arduino for grasshopper")
    elif class_name == "1 Moths":
        arduino.write(b'2')  # Send '2' to Arduino
        print("Sent '2' to Arduino for moth")
    elif class_name in ["2 Ants", "3 Beetles"]:
        arduino.write(b'3')  # Send '3' to Arduino
        print("Sent '3' to Arduino for ants or beetle")

    # Wait for 5 seconds before taking the next capture
    time.sleep(15)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
arduino.close()
