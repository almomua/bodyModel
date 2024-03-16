import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5,enable_segmentation=True)

image_path = "C:/Users/twolface/Downloads/endoooooo.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if image.shape[1] > 1500:
    target_width = 230
else:
    target_width = 330
aspect_ratio = image.shape[1] / image.shape[0]
target_height = int(target_width / aspect_ratio)
resized_image = cv2.resize(image_rgb, (target_width, target_height))






results = pose.process(resized_image)
landmarks = results.pose_landmarks

# Rest of the code to calculate the ratio R and perform calculations
if landmarks:
    # Update landmark indices based on the correct indices for your use case
    nose_landmark = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    right_ankle_landmark = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    x0, y0 = nose_landmark.x * resized_image.shape[1], nose_landmark.y * resized_image.shape[0]
    x28, y28 = right_ankle_landmark.x * resized_image.shape[1], right_ankle_landmark.y * resized_image.shape[0]

    # Calculate the Euclidean distance between key point (0) and key point (28)
    distance = np.sqrt((x28 - x0)**2 + (y28 - y0)**2)

    # Provide the actual height of the person
    actual_height_cm = 180  # Example height in centimeters

    # Calculate the ratio R
    ratio = actual_height_cm / distance
    print(distance)
    print("Calculated Ratio (R):", ratio)
else:
    print("No landmarks detected.")

pose.close()




if landmarks:
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
    center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
    hip_x = (  left_hip.x + right_hip.x)/2
    hip_y = (  left_hip.y + right_hip.y)/2
    waist_x = (  center_x + hip_x )/2
    waist_y = (  center_y + hip_y )/2
    image_width = resized_image.shape[1]
    image_height = resized_image.shape[0]

    upper_body_center = (int(center_x * image_width), int(center_y * image_height))
    hip_center = (int(hip_x * image_width), int(hip_y * image_height))
    waist_center = (int(waist_x * image_width), int(waist_y * image_height))

    print("Center of Upper Body Points:", upper_body_center)
    print("Center of waist Body Points:", waist_center)
    print("Center of hip Body Points:", hip_center)


    cv2.circle(resized_image, upper_body_center, 3, (0, 0, 255), -1)  # Draw a green circle at the upper body center
    cv2.circle(resized_image, hip_center, 3, (0, 0, 255), -1)  # Draw a green circle at the upper body center
    cv2.circle(resized_image, waist_center, 3, (0, 0, 255), -1)  # Draw a green circle at the upper body center

cv2.imshow("resized_image",resized_image)
cv2.waitKey()
cv2.destroyAllWindows()