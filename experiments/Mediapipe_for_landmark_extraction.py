import cv2
import numpy as np
import mediapipe as mp

from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


cheek_1_indices = [
    322, 426, 425, 280, 352, 366,
    410, 436, 427, 411, 376,
    287, 432, 434, 416,
    273, 422, 430,
    335, 424, 431,
    391, 423, 266, 330, 347, 346, 345, 447,
    394, 365, 364, 397, 367, 388, 435, 433, 401
]

cheek_2_indices = [
    92, 206, 205, 50, 123, 137,
    186, 216, 207, 187, 147,
    57, 212, 214, 192,
    43, 202, 210,
    106, 204, 211,
    165, 203, 36, 101, 118, 117, 116, 227,
    169, 136, 135, 172, 138, 58, 215, 213, 177
]

lip_indices = list(set([idx for pair in mp.solutions.face_mesh.FACEMESH_LIPS for idx in pair]))

# --------------------------------------------------------------------------------------------------------

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_utils.DrawingSpec(
                color=(255, 0, 0),  # Red
                thickness=2,
                circle_radius=1
            ))
        
        image_height, image_width, _ = annotated_image.shape
        for idx in cheek_1_indices:
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)  # green dots
        
        image_height, image_width, _ = annotated_image.shape
        for idx in cheek_2_indices:
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                cv2.circle(annotated_image, (x, y), 3, (0, 0, 255), -1)  # red dots
        
        image_height, image_width, _ = annotated_image.shape
        for idx in lip_indices:
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                cv2.circle(annotated_image, (x, y), 3, (255, 0, 0), -1)  # green dots

    return annotated_image

# --------------------------------------------------------------------------------------------------------

# Cheek inflation detection parameters
THRESHOLD_FACTOR = 1.05  # 5% increase from baseline
CALIBRATION_FRAMES = 20  # Frames to establish baseline

# Initialize variables for adaptive threshold
baseline_left = None
baseline_right = None
calibration_values = []

def get_cheek_metric(face_landmarks, indices, inter_eye_dist):
    points = [face_landmarks[i] for i in indices if i < len(face_landmarks)]
    if len(points) < 3:
        return 0
    
    # Calculate centroid
    centroid_x = sum(lm.x for lm in points) / len(points)
    centroid_y = sum(lm.y for lm in points) / len(points)
    
    # Calculate average distance from centroid
    distances = [np.sqrt((lm.x - centroid_x)**2 + (lm.y - centroid_y)**2) 
                for lm in points]
    avg_distance = sum(distances) / len(distances)
    
    # Normalize by inter-eye distance
    return avg_distance / inter_eye_dist

def detect_cheek_inflation(face_landmarks, frame_count):
    global baseline_left, baseline_right, calibration_values
    
    # Get relevant landmarks
    left_eye_outer = face_landmarks[33]
    right_eye_outer = face_landmarks[263]
    nose_tip = face_landmarks[4]
    
    # Calculate inter-eye distance
    inter_eye_dist = np.sqrt((right_eye_outer.x - left_eye_outer.x)**2 + 
                            (right_eye_outer.y - left_eye_outer.y)**2)
    
    
    left_metric = get_cheek_metric(face_landmarks, cheek_1_indices, inter_eye_dist)
    right_metric = get_cheek_metric(face_landmarks, cheek_2_indices, inter_eye_dist)
    
    # Calibration phase
    if frame_count < CALIBRATION_FRAMES:
        calibration_values.append((left_metric, right_metric))
        if frame_count == CALIBRATION_FRAMES - 1:
            # Calculate baseline as average of calibration values
            avg_left = sum(v[0] for v in calibration_values) / CALIBRATION_FRAMES
            avg_right = sum(v[1] for v in calibration_values) / CALIBRATION_FRAMES
            baseline_left = avg_left
            baseline_right = avg_right
        return False, False
    
    # Detection phase
    left_inflated = left_metric > baseline_left * THRESHOLD_FACTOR
    right_inflated = right_metric > baseline_right * THRESHOLD_FACTOR
    
    return left_inflated, right_inflated

# Lip landmarks (MediaPipe indices)
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
NOSE_TIP = 4

# Pouting detection parameters
POUT_WIDTH_THRESHOLD = 0.75  # 25% reduction in lip width
POUT_VERTICAL_THRESHOLD = 1.05  # 5% increase in vertical protrusion
CALIBRATION_FRAMES = 30  # More frames for stable baseline

# Initialize calibration storage
baseline_width = None
baseline_vertical = None
calibration_widths = []
calibration_verticals = []

def get_lip_metrics(face_landmarks):
    """Returns normalized lip width and vertical protrusion"""
    # Get required landmarks
    nose_tip = face_landmarks[NOSE_TIP]
    mouth_left = face_landmarks[MOUTH_LEFT]
    mouth_right = face_landmarks[MOUTH_RIGHT]
    upper_lip = face_landmarks[UPPER_LIP_CENTER]
    lower_lip = face_landmarks[LOWER_LIP_CENTER]

    # Calculate inter-eye distance for normalization
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
    inter_eye_dist = np.sqrt((right_eye.x - left_eye.x)**2 + 
                           (right_eye.y - left_eye.y)**2)

    # Horizontal lip width (distance between mouth corners)
    lip_width = np.sqrt((mouth_right.x - mouth_left.x)**2 + 
                       (mouth_right.y - mouth_left.y)**2)
    norm_width = lip_width / inter_eye_dist

    # Vertical protrusion (distance from nose to lip center)
    lip_center = ((upper_lip.x + lower_lip.x)/2, 
                 (upper_lip.y + lower_lip.y)/2)
    vertical_dist = np.sqrt((lip_center[0] - nose_tip.x)**2 + 
                           (lip_center[1] - nose_tip.y)**2)
    norm_vertical = vertical_dist / inter_eye_dist

    return norm_width, norm_vertical

def detect_pouting(face_landmarks, frame_count):
    global baseline_width, baseline_vertical, calibration_widths, calibration_verticals

    norm_width, norm_vertical = get_lip_metrics(face_landmarks)

    # Calibration phase
    if frame_count < CALIBRATION_FRAMES:
        calibration_widths.append(norm_width)
        calibration_verticals.append(norm_vertical)
        if frame_count == CALIBRATION_FRAMES - 1:
            baseline_width = np.mean(calibration_widths)
            baseline_vertical = np.mean(calibration_verticals)
        return False

    # Detection logic
    width_ratio = norm_width / baseline_width
    vertical_ratio = norm_vertical / baseline_vertical

    # Pouting requires both compressed width and increased protrusion
    return (width_ratio < POUT_WIDTH_THRESHOLD and vertical_ratio > POUT_VERTICAL_THRESHOLD)

# --------------------------------------------------------------------------------------------------------
base_options = python.BaseOptions(model_asset_path='experiments/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Load the video
video_path = "experiments/ex4_certo_full.mp4"  # Change this to your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h = int(cv2_frame.shape[0]*0.5)
    w = int(cv2_frame.shape[1]*0.5)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_frame)
    detection_result = detector.detect(image)

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)       

    # Detect cheek inflation
    inflation_text = ""
    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        left_inflated, right_inflated = detect_cheek_inflation(face_landmarks, frame_count)
        
        pouting = detect_pouting(face_landmarks, frame_count)

        if left_inflated:
            cv2.putText(
                annotated_image, 
                "Left cheek inflated", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        if right_inflated:
            cv2.putText(
                annotated_image, 
                "Right cheek inflated", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        if pouting:
            cv2.putText(annotated_image, "Pouting", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
               
    
    frame_count += 1

    annotated_image = cv2.resize(annotated_image, (w, h))
    cv2.imshow('', annotated_image)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
