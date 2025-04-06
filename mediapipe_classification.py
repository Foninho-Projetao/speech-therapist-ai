import cv2
import numpy as np
import mediapipe as mp
from collections import deque

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
    394, 365, 364, 397, 367, 388, 435, 433, 401,
    269, 270, 409, 321,
]

cheek_2_indices = [
    92, 206, 205, 50, 123, 137,
    186, 216, 207, 187, 147,
    57, 212, 214, 192,
    43, 202, 210,
    106, 204, 211,
    165, 203, 36, 101, 118, 117, 116, 227,
    169, 136, 135, 172, 138, 58, 215, 213, 177,
    39, 40, 185, 91,
]

lip_indices = list(set([idx for pair in mp.solutions.face_mesh.FACEMESH_LIPS for idx in pair]))

# --------------------------------------------------------------------------------------------------------

# Cheek inflation detection parameters
THRESHOLD_FACTOR = 1.05  # 10% increase from baseline
CALIBRATION_FRAMES = 10  # Frames to establish baseline

# Initialize variables for adaptive threshold
baseline_left = None
baseline_right = None
calibration_values = []

# Lip landmarks (MediaPipe indices)
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
NOSE_TIP = 4

# Pouting detection parameters
POUT_WIDTH_THRESHOLD = 0.75  # 25% reduction in lip width
POUT_VERTICAL_THRESHOLD = 1.0  # 0% increase in vertical protrusion
CALIBRATION_FRAMES = 30  # More frames for stable baseline

# Initialize calibration storage
baseline_width = None
baseline_vertical = None
calibration_widths = []
calibration_verticals = []

# Lip Vibration Detection Parameters
VIBRATION_CALIBRATION_FRAMES = 30      # Frames to establish baseline
VIBRATION_THRESHOLD_FACTOR = 1.20       # Movement threshold multiplier
VIBRATION_CONSECUTIVE_FRAMES = 2       # Min frames to confirm detection

# Initialize vibration detection variables
vibration_baseline = None
calibration_movements = []
prev_lip_positions = None
vibration_buffer = deque(maxlen=5)     # Smooth detection

# --------------------------------------------------------------------------------------------------------

def get_cheek_metric(face_landmarks, cheek_indices, inter_eye_dist):
    points = [face_landmarks[i] for i in cheek_indices if i < len(face_landmarks)]
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

def detect_lip_vibration(face_landmarks, frame_count):
    global vibration_baseline, calibration_movements, prev_lip_positions
    
    current_lip_positions = []
    for idx in lip_indices:
        if idx < len(face_landmarks):
            lm = face_landmarks[idx]
            current_lip_positions.append((lm.x, lm.y))
    
    if prev_lip_positions is None or len(current_lip_positions) != len(prev_lip_positions):
        prev_lip_positions = current_lip_positions
        return False
    
    # Calculate movement metrics
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
    inter_eye_dist = np.sqrt((right_eye.x - left_eye.x)**2 + 
                           (right_eye.y - left_eye.y)**2)
    
    total_movement = 0
    for (curr, prev) in zip(current_lip_positions, prev_lip_positions):
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        displacement = np.sqrt(dx**2 + dy**2) / inter_eye_dist
        total_movement += displacement
    
    avg_movement = total_movement / len(current_lip_positions)
    
    # Calibration phase
    if frame_count < VIBRATION_CALIBRATION_FRAMES:
        calibration_movements.append(avg_movement)
        if frame_count == VIBRATION_CALIBRATION_FRAMES-1:
            vibration_baseline = np.mean(calibration_movements)
        return False
    
    # Detection phase
    threshold = vibration_baseline * VIBRATION_THRESHOLD_FACTOR
    vibration_detected = avg_movement > threshold
    
    # Update previous positions
    prev_lip_positions = current_lip_positions
    
    return vibration_detected

# --------------------------------------------------------------------------------------------------------

def count_true_groups(lst, max_false_gap=5):
    group_count = 0
    in_group = False
    false_count = 0

    for val in lst:
        if val:
            if not in_group:
                # Start of a new group
                group_count += 1
                in_group = True
            false_count = 0  # Reset false counter when we see a True
        else:
            if in_group:
                false_count += 1
                if false_count > max_false_gap:
                    in_group = False  # End current group if gap too large
                    false_count = 0

    return group_count

def true_percentage(bool_list):
    if True not in bool_list:
        return 0.0
    
    if not bool_list:  # Handle empty list case
        return 0.0

    if not True in bool_list:
        return 0.0
    start_index = bool_list.index(True)
    end_index = len(bool_list) - 1 - bool_list[::-1].index(True)

    # Slice the list between first and last True
    trimmed = bool_list[start_index:end_index + 1]

    true_count = sum(trimmed)  # True is treated as 1, False as 0
    percentage = (true_count / len(trimmed)) * 100
    return percentage

# --------------------------------------------------------------------------------------------------------

def get_mediapipe_cheek_classification(video_file):
    base_options = python.BaseOptions(
        model_asset_path='experiments/face_landmarker_v2_with_blendshapes.task'
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, output_face_blendshapes=True, 
        output_facial_transformation_matrixes=True, num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Load the video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    left_expansions = []
    right_expansions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_frame)
        detection_result = detector.detect(image)    

        # Detect cheek inflation
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            left_inflated, right_inflated = detect_cheek_inflation(face_landmarks, frame_count)

            right_expansions.append(right_inflated)
            left_expansions.append(left_inflated)

        frame_count += 1

    # Clean up
    cap.release()

    right_cheek_reps = count_true_groups(right_expansions)
    left_cheek_reps = count_true_groups(left_expansions)
    # print(right_cheek_reps, left_cheek_reps)

    if (left_cheek_reps < 5 or right_cheek_reps < 5):
        return "Incorreto"
    
    if (left_cheek_reps < 8 or right_cheek_reps < 8):
        return "Parcialmente Correto"
    
    return "Correto"


def get_mediapipe_pouting_classification(video_file):
    base_options = python.BaseOptions(
        model_asset_path='experiments/face_landmarker_v2_with_blendshapes.task'
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, output_face_blendshapes=True, 
        output_facial_transformation_matrixes=True, num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Load the video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    pouting_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_frame)
        detection_result = detector.detect(image)    

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            pouting = detect_pouting(face_landmarks, frame_count)

            pouting_frames.append(pouting)
                
        frame_count += 1

    # Clean up
    cap.release()

    pouting_reps = count_true_groups(pouting_frames)
    # print(pouting_reps)

    if pouting_reps < 5:
        return "Incorreto"
    
    if pouting_reps < 8:
        return "Parcialmente Correto"
    
    return "Correto"

def get_mediapipe_vibration_classification(video_file):
    base_options = python.BaseOptions(
        model_asset_path='experiments/face_landmarker_v2_with_blendshapes.task'
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, output_face_blendshapes=True, 
        output_facial_transformation_matrixes=True, num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Load the video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    vibration_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_frame)
        detection_result = detector.detect(image)    

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]

            vibration = detect_lip_vibration(face_landmarks, frame_count)
            vibration_buffer.append(vibration)
            
            # Require sustained detection
            if sum(vibration_buffer) >= VIBRATION_CONSECUTIVE_FRAMES:
                vibration_frames.append(True)
            else:
                vibration_frames.append(False)
                
        frame_count += 1

    # Clean up
    cap.release()

    vibration_percentage = true_percentage(vibration_frames)
    # print(vibration_frames)
    # print(vibration_percentage)

    if vibration_percentage < 50:
        return "Incorreto"
    if vibration_percentage < 70:
        return "Parcialmente Correto"
    return "Correto"

if __name__ == "__main__":
    print(get_mediapipe_cheek_classification("experiments/ex_a.mp4"))
    # print(get_mediapipe_vibration_classification("experiments/fono/ex4_fono.mp4"))
    # print(get_mediapipe_pouting_classification('experiments/ex4_certo_full.mp4'))