# -*- coding: utf-8 -*-


import cv2


clicked = False


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


camera_capture = cv2.VideoCapture(0)
cv2.namedWindow('my window')
cv2.setMouseCallback('my window', on_mouse)

success, frame = camera_capture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('my window', frame)
    success, frame = camera_capture.read()

cv2.destroyWindow('my window')
camera_capture.release()
