# -*- coding: utf-8 -*-

import numpy as np
import cv2


cap = cv2.VideoCapture(0)

while True:
    # ret，True False
    ret, frame = cap.read()

    # bgr to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV顏色表，網路提供
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

