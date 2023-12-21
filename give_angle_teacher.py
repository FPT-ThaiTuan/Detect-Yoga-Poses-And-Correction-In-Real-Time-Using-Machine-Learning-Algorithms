import cv2
import csv
import os
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
    x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
    x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    return angle


def angles_teacher_yoga_csv(input_folder, output_csv):

    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['left_wrist_angle', 'right_wrist_angle', 'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle', 'left_ankle_angle', 'right_ankle_angle', 'left_hip_angle', 'right_hip_angle', 'name_yoga'])

        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith("jpeg"): 
                image_path = os.path.join(input_folder, filename)

                image = cv2.imread(image_path)
                h, w, _ = image.shape

               
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    results = pose.process(image)

                    if results.pose_landmarks is not None:
                        landmarks = results.pose_landmarks.landmark
                        angles = []

                        # Get the angle between the left elbow, wrist and left index points.
                        left_wrist_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value])
                        angles.append(left_wrist_angle)
                        # Get the angle between the right elbow, wrist and left index points.
                        right_wrist_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value])
                        angles.append(right_wrist_angle)


                        # Get the angle between the left shoulder, elbow and wrist points.
                        left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
                        angles.append(left_elbow_angle)
                        # Get the angle between the right shoulder, elbow and wrist points.
                        right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
                        angles.append(right_elbow_angle)
                        # Get the angle between the left elbow, shoulder and hip points.
                        left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
                        angles.append(left_shoulder_angle)

                        # Get the angle between the right hip, shoulder and elbow points.
                        right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                        angles.append(right_shoulder_angle)

                        # Get the angle between the left hip, knee and ankle points.
                        left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
                        angles.append(left_knee_angle)

                        # Get the angle between the right hip, knee and ankle points
                        right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
                        angles.append(right_knee_angle)

                        # Get the angle between the left hip, ankle and LEFT_FOOT_INDEX points.
                        left_ankle_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])
                        angles.append(left_ankle_angle)

                        # Get the angle between the right hip, ankle and RIGHT_FOOT_INDEX points
                        right_ankle_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])
                        angles.append(right_ankle_angle)

                        # Get the angle between the left knee, hip and right hip points.
                        left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                        angles.append(left_hip_angle)

                        # Get the angle between the left hip, right hip and right kneee points
                        right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
                        angles.append(right_hip_angle)

                        # Ghi dòng dữ liệu vào file CSV
                        csv_writer.writerow([ angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], angles[6], angles[7], angles[8], angles[9], angles[10], angles[11],os.path.basename(os.path.splitext(image_path)[0])])
#Change your path
path_input = ''
path_output = 'angle_teacher_yoga.csv'

angles_teacher_yoga_csv(path_input,path_output)