from studio.custom_block import *
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class Hand2(Block):
    op_code = 'Hand2'  # DO NOT CHANGE !!!, must be same as class name

    def init(self):
        self.width = 200
        self.height = 400
        self.input_names = ['Input Image']
        self.input_types = [socket_types.image_any]

        self.output_names = ['Generic', 'lm_list']
        self.output_types = [socket_types.generic, socket_types.generic]
        self.param['Detec_Conf'] = Slider(
            slider_min=0, slider_max=100, init_value=50)
        self.param['Track_Conf'] = Slider(
            slider_min=0, slider_max=100, init_value=50)
        self.param['Num_Hands'] = CheckBox(text='Two Hand')

    def run(self):
        image = self.input['Input Image'].data
        height, width, channel = image.shape
        with mp_hands.Hands(min_detection_confidence=(self.param['Detec_Conf'].value / 100), min_tracking_confidence=(self.param['Track_Conf'].value / 100)) as hands:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    listlm = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        listlm.append([id, cx, cy])
        self.output['Generic'].data = image
        self.output['lm_list'].data = self.param['Detec_Conf']


add_block(Hand2.op_code, Hand2)
