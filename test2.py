import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)
flip = False
pTime = 0
cTime = 0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                list = []
                xList = []
                yList = []
                bbox = []
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_landmarks.landmark):
                    height, width, channel = image.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    list.append([id, cx, cy])
                    xList.append(cx)
                    yList.append(cy)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                cv2.rectangle(
                    image, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), colorVol, 2)
                if len(list) != 0:

                    x1, y1 = list[4][1], list[4][2]
                    x2, y2 = list[8][1], list[8][2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

                    if 250 < area < 1000:
                        length = math.hypot(x2 - x1, y2 - y1)
                        volBar = np.interp(length, [50, 200], [400, 150])
                        volPer = np.interp(length, [50, 200], [0, 100])
                        if list[20][2] > list[18][2]:
                            volume.SetMasterVolumeLevelScalar(
                                volPer / 100, None)
                            colorVol = (0, 255, 0)
                            if list[16][2] > list[14][2] and list[12][2] > list[10][2]:
                                flip = not flip
                                print(int(flip))
                                time.sleep(0.2)
                            volume.SetMute(int(flip), None)
                        else:
                            colorVol = (0, 0, 255)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(image, (50, int(volBar)),
                      (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, f'{int(volPer)} %', (40, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
        cv2.putText(image, f'Vol Set: {int(cVol)}', (400, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)
        cv2.putText(image, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("image", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
