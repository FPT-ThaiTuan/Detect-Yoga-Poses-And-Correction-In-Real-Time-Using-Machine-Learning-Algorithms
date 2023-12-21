import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os


# Create a pose instance
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate angle between three points
def calculate_angle(landmark1, landmark2, landmark3,select=''):
    if select == '1':
        x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
        x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
        x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

        angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    else:
        radians = np.arctan2(landmark3[1] - landmark2[1], landmark3[0] - landmark2[0]) - np.arctan2(landmark1[1] - landmark2[1], landmark1[0] - landmark2[0])
        angle = np.abs(np.degrees(radians))
    
    angle_calc = angle + 360 if angle < 0 else angle
    #angle_calc = 360-angle_calc if angle_calc > 215 else angle_calc
    return angle_calc

def correct_feedback(model,video='0',input_csv='0'):
    # Load video
    cap = cv2.VideoCapture(video)  # Replace with your video path

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    accurate_angle_lists = []
    #Your accurate angle list
    # with open(input_csv, 'r') as inputCSV:
    #     for row in csv.reader(inputCSV):
    #         if row[12] == selectedPose: 
    #             accurate_angle_lists = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11])]

    angle_name_list = ["L-wrist","R-wrist","L-elbow", "R-elbow","L-shoulder", "R-shoulder", "L-knee", "R-knee","L-ankle","R-ankle","L-hip", "R-hip"]
    angle_coordinates = [[13, 15, 19], [14, 16, 18], [11, 13, 15], [12, 14, 16], [13, 11, 23], [14, 12, 24], [23, 25, 27], [24, 26, 28],[23,27,31],[24,28,32],[24,23,25],[23,24,26]]
    correction_value = 30

    fps_time = 0
   
    while cap.isOpened():
        ret_val, image = cap.read()
        #Resize the image to 50% of the original size
        # scale_factor = 1.5
        # image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        if not ret_val:
            break

        # Convert the image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_rgb = cv2.resize(image_rgb, (0, 0), None, .50, .50)
        # Get the pose landmarks
        results = pose.process(image_rgb)
        #save angle main
        angles = []
        
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            # Get the angle between the left elbow, wrist and left index points.
            left_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value],'1')
            angles.append(left_wrist_angle)
            # Get the angle between the right elbow, wrist and left index points.
            right_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value],'1')
            angles.append(right_wrist_angle)


            # Get the angle between the left shoulder, elbow and wrist points.
            left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],'1')
            angles.append(left_elbow_angle)
            # Get the angle between the right shoulder, elbow and wrist points.
            right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],'1')
            angles.append(right_elbow_angle)
            # Get the angle between the left elbow, shoulder and hip points.
            left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],'1')
            angles.append(left_shoulder_angle)

            # Get the angle between the right hip, shoulder and elbow points.
            right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],'1')
            angles.append(right_shoulder_angle)

            # Get the angle between the left hip, knee and ankle points.
            left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],'1')
            angles.append(left_knee_angle)

            # Get the angle between the right hip, knee and ankle points
            right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],'1')
            angles.append(right_knee_angle)

            # Get the angle between the left hip, ankle and LEFT_FOOT_INDEX points.
            left_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value],'1')
            angles.append(left_ankle_angle)

            # Get the angle between the right hip, ankle and RIGHT_FOOT_INDEX points
            right_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value],'1')
            angles.append(right_ankle_angle)

            # Get the angle between the left knee, hip and right hip points.
            left_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],'1')
            angles.append(left_hip_angle)

            # Get the angle between the left hip, right hip and right kneee points
            right_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],'1')
            angles.append(right_hip_angle)
            
            y = model.predict([angles])
            
            Name_Yoga_Classification = str(y[0])

            # Dự đoán xác suất cho mỗi lớp
            probabilities = model.predict_proba([angles])
            print([probabilities])
            # Lấy danh sách các lớp
            class_labels = model.classes_
            check_accry_class = False
            # Kiểm tra xác suất của từng lớp
            for i,class_label in enumerate(class_labels):
                probability = probabilities[0, i]
                if probability > 0.5 :
                    check_accry_class = True
                else:
                    continue

            with open(input_csv, 'r') as inputCSV:
                for row in csv.reader(inputCSV):
                    if row[12] == Name_Yoga_Classification: 
                        accurate_angle_lists = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11])]
                
            folder_path = 'F:/Anaconda_Project/Utilizing_Deep_Learning_for_Human_Pose_Estimation_in_Yoga/teacher_yoga/angle_teacher_yoga.csv'

            # Tiền tố cần kiểm tra
            prefix_to_match = Name_Yoga_Classification


            if check_accry_class == True :
                # # Duyệt qua tất cả các tệp trong thư mục
                # for file_name in os.listdir(folder_path):
                #     # Tạo đường dẫn đầy đủ của tệp
                #     file_path = os.path.join(folder_path, file_name)

                #     # Kiểm tra nếu tên tệp bắt đầu bằng tiền tố mong muốn
                #     if file_name.startswith(prefix_to_match):
                #         # Đọc và hiển thị ảnh (hoặc thực hiện các thao tác khác tùy ý)
                #         # image_ins = cv2.imread(file_path)
                #         # ins_resize = cv2.resize(image_ins, (0, 0), None, .50, .50)

                # Display the classification result in the bottom-left corner
                (w, h), _ = cv2.getTextSize(Name_Yoga_Classification, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(image, (10, image.shape[0] - 30), (10 + w, image.shape[0] - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, Name_Yoga_Classification, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            else :
                # Display the classification result in the bottom-left corner
                (w, h), _ = cv2.getTextSize('None', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(image, (10, image.shape[0] - 30), (10 + w, image.shape[0] - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, 'None', (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            
            
            correct_angle_count = 0
            for itr in range(12):
                point_a = (int(landmarks[angle_coordinates[itr][0]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][0]].y * image.shape[0]))
                point_b = (int(landmarks[angle_coordinates[itr][1]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][1]].y * image.shape[0]))
                point_c = (int(landmarks[angle_coordinates[itr][2]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][2]].y * image.shape[0]))

                angle_obtained = calculate_angle(point_a, point_b, point_c,'0')
                    
                if angle_obtained < accurate_angle_lists[itr] - correction_value:
                    status = "more"
                elif accurate_angle_lists[itr] + correction_value < angle_obtained:
                    status = "less"
                else:
                    status = "OK"
                    correct_angle_count += 1

                # Display status
                status_position = (point_b[0] - int(image.shape[1] * 0.03), point_b[1] + int(image.shape[0] * 0.03))
                cv2.putText(image, f"{status}", status_position, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


                # Display angle values on the image
                cv2.putText(image, f"{angle_name_list[itr]}", (point_b[0] - 50, point_b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                
                # pos_onScreen = [200,1000]
                # # Chọn vị trí xuất hiện dựa trên giá trị của itr
                # cv2.putText(image, angle_name_list[itr]+":- %s" % (status), (pos_onScreen[itr%2], (itr+1)*60),  cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


                


            # # Draw the entire pose on the person
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            
            posture = "CORRECT" if correct_angle_count > 9 else "WRONG"
            posture_color = (0, 255, 0) if posture == "CORRECT" else (0, 0, 255)  # Màu xanh cho đúng, màu đỏ cho sai

            # Thiết lập văn bản và màu cho tư thế
            posture_position = (10, 30)  # Điều chỉnh giá trị này để đặt văn bản
            cv2.putText(image, f"Yoga movements: {posture}", posture_position, cv2.FONT_HERSHEY_PLAIN, 1.5, posture_color, 2)

            # Thiết lập màu văn bản và định dạng cho FPS
            fps_text = f"FPS: {1.0 / (time.time() - fps_time):.3f}"  # Hiển thị FPS với 3 chữ số sau dấu thập phân
            fps_position = (10, 60)  # Điều chỉnh giá trị này để đặt văn bản
            cv2.putText(image, fps_text, fps_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # # Xác định chiều dài nhỏ nhất của hai ảnh
            # min_height = min(ins_resize.shape[0], resize_rgb.shape[0])

            # # Thay đổi kích thước ảnh để chiều dài bằng nhau
            # image_resized = cv2.resize(image, (int(min_height / image.shape[0] * image.shape[1]), min_height))
            # instruction_image_resized = cv2.resize(image_ins, (int(min_height / image_ins.shape[0] * image_ins.shape[1]), min_height))

            # # Ghép nối ảnh theo chiều ngang (50/50)
            # result = np.concatenate((image_resized, instruction_image_resized), axis=1)


            cv2.imshow('Mediapipe Pose Estimation', image)
            #cv2.imshow('Mediapipe Pose Estimation', result)
            fps_time = time.time()



            

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
