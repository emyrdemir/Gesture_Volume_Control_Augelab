from studio.custom_block import *
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class Hand1(Block):
    op_code = 'Hand1'  # DO NOT CHANGE !!!, must be same as class name

    def init(self):
        self.width = 200
        self.height = 400
        self.input_names = ['Input Image']
        self.input_types = [socket_types.image_any]

        self.output_names = ['Generic']
        self.output_types = [socket_types.generic]

    def run(self):
        image = self.input['Input Image'].data

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        self.output['Generic'].data = image


add_block(Hand1.op_code, Hand1)
