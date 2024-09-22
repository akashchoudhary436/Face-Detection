from flask import Flask, request, render_template, Response
import cv2
import numpy as np
import pickle
import os
import warnings
from werkzeug.utils import secure_filename
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from flask import Flask, request, render_template, Response, url_for, send_from_directory
# Suppress specific deprecation warnings from protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Load the pre-trained model (for inference)
with open('Best_RandomForest.pkl', 'rb') as f:
    face_shape_model = pickle.load(f)

def distance_3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_face_features(coords):
    # Define indices for landmarks
    landmark_indices = {
        'forehead': 10,
        'chin': 152,
        'left_cheek': 234,
        'right_cheek': 454,
        'left_eye': 263,
        'right_eye': 33,
        'nose_tip': 1
    }

    # Extract features based on landmark indices
    features = []
    landmarks_dict = {name: coords[idx] for name, idx in landmark_indices.items()}

    # Calculate distances between important landmarks
    features.append(distance_3d(landmarks_dict['forehead'], landmarks_dict['chin']))  # Face height
    features.append(distance_3d(landmarks_dict['left_cheek'], landmarks_dict['right_cheek']))  # Face width
    features.append(distance_3d(landmarks_dict['left_eye'], landmarks_dict['right_eye']))  # Eye distance

    # Additional distances
    features.append(distance_3d(landmarks_dict['nose_tip'], landmarks_dict['left_eye']))  # Nose to left eye
    features.append(distance_3d(landmarks_dict['nose_tip'], landmarks_dict['right_eye']))  # Nose to right eye
    features.append(distance_3d(landmarks_dict['chin'], landmarks_dict['left_cheek']))  # Chin to left cheek
    features.append(distance_3d(landmarks_dict['chin'], landmarks_dict['right_cheek']))  # Chin to right cheek
    features.append(distance_3d(landmarks_dict['forehead'], landmarks_dict['left_eye']))  # Forehead to left eye
    features.append(distance_3d(landmarks_dict['forehead'], landmarks_dict['right_eye']))  # Forehead to right eye

    # Additional features

    # # Facial aspect ratios
    # face_width = distance_3d(landmarks_dict['left_cheek'], landmarks_dict['right_cheek'])
    # face_height = distance_3d(landmarks_dict['forehead'], landmarks_dict['chin'])
    # eye_distance = distance_3d(landmarks_dict['left_eye'], landmarks_dict['right_eye'])

    # features.append(face_width / face_height)  # Aspect ratio of face width to height
    # features.append(face_height / eye_distance)  # Aspect ratio of face height to eye distance

    # # More distance features
    # features.append(distance_3d(landmarks_dict['left_eye'], landmarks_dict['chin']))  # Eye to chin
    # features.append(distance_3d(landmarks_dict['right_eye'], landmarks_dict['chin']))  # Eye to chin
    # features.append(distance_3d(landmarks_dict['left_cheek'], landmarks_dict['forehead']))  # Cheek to forehead
    # features.append(distance_3d(landmarks_dict['right_cheek'], landmarks_dict['forehead']))  # Cheek to forehead

    return np.array(features)

def get_face_shape_label(label):
    shapes = ["Heart", "Oval", "Round", "Square"]
    return shapes[label]

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define function to compute Euclidean distance in 3D
def distance_3d(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Create landmark proto
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        # Draw face landmarks
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image



def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = face_landmarker.detect(image)

            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks]
                    landmarks = np.array(landmarks)
                    face_features = calculate_face_features(landmarks)
                    face_shape_label = face_shape_model.predict([face_features])[0]
                    face_shape = get_face_shape_label(face_shape_label)
                    annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
                    cv2.putText(annotated_image, f"Face Shape: {face_shape}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                annotated_image = rgb_frame

            ret, buffer = cv2.imencode('.jpg', annotated_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    face_shape = None
    error = None
    file_url = None  # Path to the annotated image file

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file part"
        else:
            file = request.files['file']
            if file.filename == '':
                error = "No selected file"
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Read and process the image using OpenCV and MediaPipe
                img = cv2.imread(file_path)
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                detection_result = face_landmarker.detect(image)

                if detection_result.face_landmarks:
                    for face_landmarks in detection_result.face_landmarks:
                        landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks]
                        landmarks = np.array(landmarks)
                        face_features = calculate_face_features(landmarks)
                        face_shape_label = face_shape_model.predict([face_features])[0]
                        face_shape = get_face_shape_label(face_shape_label)

                        # Draw the landmarks and face shape text on the image
                        annotated_image = draw_landmarks_on_image(rgb_image, detection_result)
                        cv2.putText(annotated_image, f"Face Shape: {face_shape}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Save the annotated image
                        annotated_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
                        cv2.imwrite(annotated_file_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                        # Create the URL to send the annotated image to the front end
                        file_url = url_for('uploaded_file', filename='annotated_' + filename)
                        break
                else:
                    error = "No face detected in the image"

    return render_template('upload.html', face_shape=face_shape, error=error, file_url=file_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/real_time')
def real_time():
    return render_template('real_time.html')

if __name__ == '__main__':
    app.run(debug=True)
