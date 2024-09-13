import cv2
import dlib
import numpy as np
import os
import csv
from scipy.spatial.distance import euclidean

# Load OpenCV's pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Load Dlib's pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define directories for saving data
data_dir = 'face_data'
csv_file_path = os.path.join(data_dir, 'face_data.csv')
csv_file_path2 = os.path.join('face_data2', 'face_data2.csv')  # Update path for face_data2.csv

# Function to extract facial landmarks and return the points
def get_facial_landmarks(image, rect):
    landmarks = predictor(image, rect)
    points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
    return points

# Function to calculate the distance between two feature vectors
def calculate_distance(features1, features2):
    features1 = np.array(features1)
    features2 = np.array(features2)
    if features1.shape != features2.shape:
        # Print lengths for debugging
        print(f"Feature length mismatch: {features1.shape[0]} vs {features2.shape[0]}")
        # Adjust length by padding with zeros
        max_len = max(features1.shape[0], features2.shape[0])
        features1 = np.pad(features1, (0, max_len - len(features1)))
        features2 = np.pad(features2, (0, max_len - len(features2)))
    return np.linalg.norm(features1 - features2)

# Function to extract features from landmarks
def extract_features(landmarks):
    features = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            features.append(euclidean(landmarks[i], landmarks[j]))
    # Ensure features length is 2278
    features = features[:2278]  # Truncate to 2278
    features.extend([0] * (2278 - len(features)))  # Fill with zero if less than 2278
    return features

# Function to save face details including landmarks and features to CSV
def save_face_details_to_csv(file_path, landmarks, features):
    with open(file_path, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        # Flatten landmarks and features to a single list
        landmarks_flat = [coord for point in landmarks for coord in point]
        row = landmarks_flat + features  # Combine landmarks and features
        csv_writer.writerow(row)

# Function to find the closest match from the database
def find_closest_match(features, csv_file_path2):
    min_distance = float('inf')
    closest_image_path = None
    
    # Ensure the CSV file exists and is not empty
    if not os.path.exists(csv_file_path2) or os.stat(csv_file_path2).st_size == 0:
        print("face_data2.csv is missing or empty.")
        return None

    with open(csv_file_path2, mode='r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)  # Skip header
        for row in csv_reader:
            if len(row) < 2:
                continue  # Skip rows that don't have enough data
            image_path = row[1]  # Get the image path
            stored_features = list(map(float, row[2:2+2278]))  # Get the stored features
            distance = calculate_distance(features, stored_features)
            if distance < min_distance:
                min_distance = distance
                closest_image_path = image_path
    
    return closest_image_path

# Start video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Flag to capture data only once
data_collected = False

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

        # Extract features from landmarks
        features = extract_features(landmarks)

        # Draw landmarks and additional points (if desired)
        for i, point in enumerate(landmarks):
            color = (0, 255, 0) if i < 27 else (0, 0, 255)
            cv2.circle(frame, point, 2, color, -1)

        # Draw additional intermediate points for denser marking
        for i in range(len(landmarks) - 1):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            num_intermediate_points = 5
            for t in np.linspace(0, 1, num_intermediate_points):
                x_inter = int(p1[0] * (1 - t) + p2[0] * t)
                y_inter = int(p1[1] * (1 - t) + p2[1] * t)
                cv2.circle(frame, (x_inter, y_inter), 1, (255, 255, 255), -1)

        # Extract features and save face details only once
        if not data_collected:
            save_face_details_to_csv(csv_file_path, landmarks, features)
            data_collected = True  # Prevent further data collection
            print("Data collected and saved to CSV")

            # Find closest match from face_data2.csv
            closest_image_path = find_closest_match(features, csv_file_path2)
            if closest_image_path:
                print(f"Closest match image path: {closest_image_path}")
            else:
                print("No match found")

    # Display the resulting frame
    cv2.imshow('Face Detection and Landmarking', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
