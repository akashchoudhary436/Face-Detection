import cv2
import dlib
import numpy as np
import os
import time
import h5py  # For HDF5 support

# Load OpenCV's pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Dlib's pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define directory for saving data
data_dir = 'face_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define HDF5 file path
data_h5_path = os.path.join(data_dir, 'face_data.h5')

# Initialize or open HDF5 file for storing data
def initialize_hdf5_file(file_path):
    with h5py.File(file_path, 'w') as f:
        # Create chunked datasets for landmarks, features, and dot markings
        f.create_dataset('X_train', shape=(0, 136), maxshape=(None, 136), dtype=np.float32, chunks=(1, 136))
        f.create_dataset('dot_markings', shape=(0, 68, 2), maxshape=(None, 68, 2), dtype=np.float32, chunks=(1, 68, 2))
        # Update the feature size according to the output of `extract_features`
        feature_size = 2278  # Update this according to the actual number of features you have
        f.create_dataset('features', shape=(0, feature_size), maxshape=(None, feature_size), dtype=np.float32, chunks=(1, feature_size))

initialize_hdf5_file(data_h5_path)

# Function to extract facial landmarks and return the points
def get_facial_landmarks(image, rect):
    landmarks = predictor(image, rect)
    points = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
    return points

# Function to calculate the distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to extract features from landmarks
def extract_features(landmarks):
    features = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            features.append(euclidean_distance(landmarks[i], landmarks[j]))
    return features

# Function to save face details including landmarks and features
def save_face_details(landmarks, features):
    with h5py.File(data_h5_path, 'a') as f:
        # Convert landmarks and features to numpy arrays
        landmarks_np = np.array(landmarks, dtype=np.float32)
        features_np = np.array(features, dtype=np.float32)
        
        # Append landmarks and features to the datasets
        f['dot_markings'].resize((f['dot_markings'].shape[0] + 1, f['dot_markings'].shape[1], f['dot_markings'].shape[2]))
        f['dot_markings'][-1, :, :] = landmarks_np
        
        f['features'].resize((f['features'].shape[0] + 1, f['features'].shape[1]))
        f['features'][-1, :] = features_np

# Start video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Capture data every second
capture_interval = 1  # seconds
last_capture_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to grayscale (needed for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV Haar Cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # For each detected face, detect landmarks using Dlib
    for (x, y, w, h) in faces:
        # Convert the face to Dlib rectangle object
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get the facial landmarks
        landmarks = get_facial_landmarks(gray, dlib_rect)
        
        # Draw the landmarks on the face with smaller and denser marking
        for i, point in enumerate(landmarks):
            color = (0, 255, 0) if i < 27 else (0, 0, 255)
            cv2.circle(frame, point, 2, color, -1)  # Smaller circles
        
        # Draw additional intermediate points for denser marking
        for i in range(len(landmarks) - 1):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            num_intermediate_points = 5
            for t in np.linspace(0, 1, num_intermediate_points):
                x = int(p1[0] * (1 - t) + p2[0] * t)
                y = int(p1[1] * (1 - t) + p2[1] * t)
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)  # White dots for intermediate points

        # Extract features and save face details every second
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            features = extract_features(landmarks)
            save_face_details(landmarks, features)
            last_capture_time = current_time

    # Display the resulting frame
    cv2.imshow('Face Detection and Landmarking', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
