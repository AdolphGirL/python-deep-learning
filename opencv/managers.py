# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time


class WindowManager(object):
    def __init__(self, windowName='default', keypressCallback=None):
        self.keypressCallback = keypressCallback

        # 私有屬性，且不提供外部修改
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    # @isWindowCreated.setter
    # def isWindowCreated(self, isWindowCreated):
    #     self._isWindowCreated = isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keyCode = cv2.waitKey(1)
        if self.keypressCallback is not None and keyCode != -1:
            # 提取最後一個字節，確保返回值為ASCII
            # waitKey() may return a value that encodes more than just the ASCII keycode.
            # (A bug is known to occur on Linux when OpenCV uses GTK as its backend GUI
            # library.)
            keyCode &= 0xFF
            self.keypressCallback(keyCode)


class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrive()
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """Capture the next frame, if any"""
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """Draw to the window. Write to files. Release the frame"""
        if self.frame is None:
            self._enteredFrame = False
            return

        # 計算fps
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed

        self._framesElapsed += 1

        # 顯示影象
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                # 向左/右方向翻轉陣列，翻轉影象
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        # 圖片檔案生成
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # 錄影生成
        self._writeVideoFrame()

        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """Wirte the next exited frame to an image file."""
        self._imageFilename = filename

    def startWritingVideo(self, filename, encodeing=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        """Start wirting exited frames to a video file."""
        self._videoFilename = filename
        self._videoEncoding = encodeing

    def stopWritingVideo(self):
        """Stop writing eited frames to a video file."""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                if self._framesElapsed < 20:
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)
        self._videoWriter.write(self._frame)


if __name__ == '__main__':
    pass

