# =======================================================#
# This script is used to track the hand in the video.    #
# Project Name: HandTracking                             #
# Created By : Raj Dalsaniya                             #
# Created On : 12/04/2019                                #
# Library Used for this projects: Numpy,mediapipe,opencv #
# =======================================================#


import numpy as np
import cv2
import mediapipe as mp
import time


class handDetector():
    """
    HandDetector class is used to detect the hand in the video. It will create keypoints and descriptors for the
    hand. after keypoint it will create lines that will join them. For creating keypoints and lines it uses mediapipe
    library. For more details about mediapipe library please visit: https://mediapipe.com .

    Function findHand() will return the hand keypoints and lines.
    Function findPosition() will return the detected keypoints id and position.

    mediapipe is tensorflow based library that can be used for hand tracking and other computer vision tasks.trained
    model on large image dataset that help to detect hands etc in the video real time. FPS of the model is around 30.
    """

    def __init__(self, image_mode=False, num_hands=2, modelComplex=1, detection_confidence=0.5,
                 tracking_confidence=0.5):
        self.image_mode = image_mode
        self.num_hands = num_hands
        self.detection_confidence = detection_confidence
        self.tracking_con = tracking_confidence
        self.modelComplex = modelComplex

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.image_mode, self.num_hands, self.modelComplex, self.detection_confidence,
                                        self.tracking_con)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS
                    )

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        return lmList


# ================== DUMMY CODE START ===========================#
#  This code is you can used for reference                                                               #
# ===============================================================#
def main():
    pre_time = 0
    curr_time = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = detector.findHands(frame)

            curr_time = time.time()
            fps = 1 / (curr_time - pre_time)
            pre_time = curr_time

            cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ================== DUMMY CODE END =================================#
#                                                                    #
# ===================================================================#

if __name__ == '__main__':
    main()
