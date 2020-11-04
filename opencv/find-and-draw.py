# -*- coding: utf-8 -*-


import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # bgr to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV顏色表，網路提供
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
