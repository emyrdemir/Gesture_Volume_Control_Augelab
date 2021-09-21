import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    cv2.imshow("image", image)
    print(image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
