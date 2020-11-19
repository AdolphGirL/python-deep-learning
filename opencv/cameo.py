# -*- coding: utf-8 -*-
import cv2
import datetime
from model import WindowManager, CaptureManager


class Cameo(object):
    def __init__(self):
        # 建立一個視窗，並將鍵盤的回撥函式傳入
        self._windowManager = WindowManager('Video_Date', self.onKeypress)

        # 告訴程式資料來自攝像頭， 還有鏡面效果
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            # 這裡的enterFrame就是告訴程式從攝像頭中取資料
            self._captureManager.enterFrame()

            # 下面的這個frame是原始幀資料，這裡沒有做任何修改，後面的教程會對這個幀資料進行修改
            frame = self._captureManager.frame

            # exitFrame看起來是像是退出的意思，其實主要功能都是在這裡方法裡實現的，截圖、錄影都是在這裡
            self._captureManager.exitFrame()

            # 回撥函式
            self._windowManager.processEvents()

    # 定義鍵盤的回撥函式，用於self._windowManager.processEvents()的呼叫
    def onKeypress(self, keyCode):
        """
        快捷鍵設定：
        當按下“空格”鍵的時候，會進行抓屏。
        當按下‘tab’鍵的時候，就開始或者停止錄影。
        當然相應的目錄也會生成圖片或者視訊檔案
        :param keyCode:
        :return:
        """
        s = datetime.datetime.now().strftime('%Y-%m-%d')

        # space
        if keyCode == 32:
            # 截圖儲存的檔名字
            self._captureManager.writeImage(s + '.png')
        # tab
        elif keyCode == 9:
            if not self._captureManager.isWritingVideo:
                # 告訴程式，錄影儲存的檔名字
                self._captureManager.startWritingVideo(s + '_DateVideo.avi')
            else:
                self._captureManager.stopWritingVideo()
        # escape
        elif keyCode == 27:
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Cameo().run()
