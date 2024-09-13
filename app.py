import cv2
import dlib
import numpy as np
import os
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template

# Load OpenCV's pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Dlib's pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define directory for saving data
data_dir = 'face_data'
images_dir = os.path.join(data_dir, 'images')  # Folder for saving uploaded images
csv_file_path = os.path.join(data_dir, 'face_data.csv')

# Ensure the images directory exists
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# If CSV doesn't exist, create it with appropriate headers
if not os.path.exists(csv_file_path):
    df = pd.DataFrame(columns=['Image Name', 'Image Path', 'Landmarks'])
    df.to_csv(csv_file_path, index=False)

# Function to extract facial landmarks and return the points
def get_facial_landmarks(image, rect):
    landmarks = predictor(image, rect)
    points = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
    return points

# Function to save face details (landmarks and image path) to a CSV file
def save_face_details(image_name, image_path, landmarks):
    # Flatten landmarks (68 points) into a 136-value array
    landmarks_flat = np.array(landmarks).flatten().tolist()
    
    # Create a dictionary with the image name, image path, and landmarks
    data = {
        'Image Name': image_name,
        'Image Path': image_path,
        'Landmarks': landmarks_flat
    }

    # Convert the dictionary to a DataFrame and append to CSV
    df = pd.DataFrame([data])
    df.to_csv(csv_file_path, mode='a', index=False, header=False)

# Function to draw landmarks on the image
def draw_landmarks(image, landmarks):
    # Draw each landmark with a green dot
    for i, point in enumerate(landmarks):
        color = (0, 255, 0) if i < 27 else (0, 0, 255)  # Use green for first 27, red for the rest
        cv2.circle(image, point, 2, color, -1)  # Draw green or red dot for each landmark

    # Draw additional intermediate points for denser marking
    for i in range(len(landmarks) - 1):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        num_intermediate_points = 5
        for t in np.linspace(0, 1, num_intermediate_points):
            x = int(p1[0] * (1 - t) + p2[0] * t)
            y = int(p1[1] * (1 - t) + p2[1] * t)
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)  # Draw white dots for intermediate points
    
    return image

# Function to process the uploaded image
def process_uploaded_image(image_path):
    # Load the image from the path
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV Haar Cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print(f"No face detected in the image {image_path}.")
        return

    # For each detected face, detect landmarks using Dlib
    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get the facial landmarks
        landmarks = get_facial_landmarks(gray, dlib_rect)
        
        # Get the image name from the path
        image_name = os.path.basename(image_path)
        
        # Draw landmarks on the image
        image_with_landmarks = draw_landmarks(image, landmarks)
        
        # Save the modified image with landmarks
        modified_image_path = os.path.join(images_dir, f"landmarks_{image_name}")
        cv2.imwrite(modified_image_path, image_with_landmarks)
        
        # Save the face details (landmarks and image path) to the CSV file
        save_face_details(image_name, image_path, landmarks)
        
        print(f"Processed and saved details for {image_name} with landmarks.")

# Function to handle the upload and processing of the images
def upload_and_process_images(files):
    for file in files:
        # Save the uploaded image to the images directory
        image_filename = secure_filename(file.filename)
        image_path = os.path.join(images_dir, image_filename)
        file.save(image_path)

        # Process the uploaded image to detect landmarks
        process_uploaded_image(image_path)

# Flask application
app = Flask(__name__)

# Route to display the upload form and process uploaded files
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if files are part of the request
        if 'files[]' not in request.files:
            return "No file part"
        
        files = request.files.getlist('files[]')
        
        # Check if files are selected
        if not files:
            return "No selected files"
        
        # If files are valid, process them
        if files:
            upload_and_process_images(files)  # Call the function to save and process images
            return "Files successfully uploaded and processed"
    
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
