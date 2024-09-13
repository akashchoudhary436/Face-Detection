import numpy as np
import cv2
import dlib
import imutils
import math
from imutils import face_utils
from scipy.ndimage import uniform_filter1d

# Paths for dlib's shape predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Threshold values for different face shapes
thresholds = {
    "heart": (53.4850795291, 26.0),
    "round": (51.9181404322, 28.0),
    "square": (33.6356433044, 42.0),
    "long": (27.349185998, 56.00)
}

# Initialize dlib's face detector and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# List to store previous face shape classifications for smoothing
shape_history = []

def smooth(value_list, window_size=5):
    if len(value_list) < window_size:
        return value_list
    return uniform_filter1d(value_list, size=window_size)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    rects = detector(gray, 1)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Draw facial landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        
        # Extract specific landmarks
        try:
            x1, y1 = shape[0]      # Left eyebrow outer corner
            x3, y3 = shape[2]      # Left eyebrow inner corner
            x5, y5 = shape[4]      # Nose bridge
            x7, y7 = shape[6]      # Nose tip
            x9, y9 = shape[8]      # Right nostril
            x17, y17 = shape[16]   # Jawline
            x28, y28 = shape[27]   # Right cheekbone
        except IndexError:
            continue
        
        # Compute slopes
        slope1 = (y3 - y1) / (x3 - x1) if (x3 - x1) != 0 else float('inf')
        slope2 = (y5 - y3) / (x5 - x3) if (x5 - x3) != 0 else float('inf')
        slope3 = (y7 - y5) / (x7 - x5) if (x7 - x5) != 0 else float('inf')
        slope4 = (y9 - y7) / (x9 - x7) if (x9 - x7) != 0 else float('inf')

        # Compute distances and threshold
        distx = math.sqrt((x1 - x17)**2 + (y1 - y17)**2)
        disty = math.sqrt((x9 - x28)**2 + (y9 - y28)**2)
        thresh = distx - disty

        # Determine face shape
        face_shape = "Unknown"
        if thresh <= thresholds["long"][1]:
            if slope1 >= 7.395:
                if slope3 >= 1.15:
                    face_shape = "long face"
                else:
                    face_shape = "square face"
            else:
                if slope3 >= 1.15:
                    face_shape = "square face"
                else:
                    face_shape = "long face"
        else:
            if slope1 >= 11.75:
                if slope3 <= 1.1:
                    face_shape = "heart face"
                else:
                    face_shape = "round face"
            else:
                if slope3 >= 1.1:
                    face_shape = "round face"
                else:
                    face_shape = "heart face"
        
        # Add face shape to history for smoothing
        shape_history.append(face_shape)
        shape_history = shape_history[-10:]  # Keep the last 10 shapes
        
        # Smooth the face shape detection
        most_common_shape = max(set(shape_history), key=shape_history.count)
        
        # Display the face shape on the image
        cv2.putText(frame, most_common_shape, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Real-Time Face Shape Detection", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
