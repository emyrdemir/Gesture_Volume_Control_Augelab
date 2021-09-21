# handaudio
from studio.custom_block import *
import cv2
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


class audio(Block):
    op_code = 'audio'  # DO NOT CHANGE !!!, must be same as class name

    def init(self):
        self.width = 200
        self.height = 200

        self.input_names = ['image', 'listlm']
        self.input_types = [socket_types.image_any, socket_types.generic]

        self.output_names = ['image', 'volume']
        self.output_types = [socket_types.image_any, socket_types.generic]

    def run(self):
        cVol = 0
        length = None
        listlm = []
        image = self.input['image'].data
        try:
            listlm = self.input['listlm'].data[0]
            length = self.input['listlm'].data[1]

            if len(listlm) > 20 and length != None:
                xmin, xmax = min(listlm[1]), max(listlm[1])
                ymin, ymax = min(listlm[2]), max(listlm[2])
                area = (xmax - xmin) * (ymax - ymin) // 100
                if 900 < area < 1400:
                    volBar = np.interp(length, [50, 200], [400, 150])
                    volPer = np.interp(length, [50, 200], [0, 100])
                    if listlm[20][2] > listlm[18][2]:
                        volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                        colorVol = (0, 255, 0)
                    else:
                        colorVol = (0, 0, 255)
                cv2.rectangle(image, (50, int(volBar)),
                              (85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(image, f'{int(volPer)} %', (40, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
                cv2.putText(
                    image, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)
                cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)
        except:
            None
        self.output['image'].data = image
        self.output['volume'].data = 0


add_block(audio.op_code, audio)
