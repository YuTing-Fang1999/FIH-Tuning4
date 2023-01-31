from tkinter.messagebox import NO
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np


class ROI_coordinate(object):
    r1 = -1
    r2 = -1
    c1 = -1
    c2 = -1

class ImageViewer(QtWidgets.QGraphicsView):

    def __init__(self, w, idx):
        super().__init__()
        self.w = w
        self.idx = idx

        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene()
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        # self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        # self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        # 設置view可以進行鼠標的拖拽選擇
        self.setDragMode(self.RubberBandDrag)

        # self.roi_coordinate = ROI_coordinate()  # 圖片座標
        self.origin_pos = None  # 螢幕座標
        self.end_pos = None
        self.scenePos1 = None
        self.scenePos2 = None

        self.img = []

        # img = cv2.imdecode(np.fromfile(file="D:\FIH\FIH-Tuning2\Galaxy A52s.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
        # self.set_img(img)


    def hasPhoto(self):
        return not self._empty

    def resizeEvent(self, event):
        self.fitInView()

    def fitInView(self):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        # self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())
        # self.fitInView()

    def set_img(self, img):
        self.img = img
        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1] * img.shape[2], QImage.Format_RGB888).rgbSwapped()

        self.setPhoto(QPixmap(qimg))
        self.fitInView()

    def wheelEvent(self, event):
        if self.idx==0: self.w.target_viewer.wheelEvent(event)
        if self.hasPhoto():
            # https://blog.csdn.net/GoForwardToStep/article/details/77035287
            # 獲取當前鼠標相對於view的位置
            cursorPoint = event.pos()
            # 獲取當前鼠標相對於scene的位置
            scenePos = self.mapToScene(QPoint(cursorPoint.x(), cursorPoint.y()))

            # 獲取view的寬高
            viewWidth = self.viewport().width()
            viewHeight = self.viewport().height()

            # 獲取當前鼠標位置相當於view大小的橫縱比例
            hScale = cursorPoint.x() / viewWidth
            vScale = cursorPoint.y() / viewHeight

            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1

            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

            # 將scene坐標轉換為放大縮小後的坐標
            viewPoint = self.transform().map(scenePos)
            # 通過滾動條控制view放大縮小後的展示scene的位置
            self.horizontalScrollBar().setValue(int(viewPoint.x() - viewWidth * hScale))
            self.verticalScrollBar().setValue(int(viewPoint.y() - viewHeight * vScale))

    def mousePressEvent(self, event):
        super(ImageViewer, self).mousePressEvent(event)
        if self.idx==0: self.w.target_viewer.mousePressEvent(event)
        if self.dragMode() == self.RubberBandDrag:
            # if event.buttons() == Qt.LeftButton:
            self.origin_pos = event.pos()

    def mouseMoveEvent(self, event):
        super(ImageViewer, self).mouseMoveEvent(event)
        if self.idx==0: self.w.target_viewer.mouseMoveEvent(event)
        # print(self.idx, "mouse move", event.x(), event.y())

    def mouseReleaseEvent(self, event):
        super(ImageViewer, self).mouseReleaseEvent(event)
        if self.idx==0: self.w.target_viewer.mouseReleaseEvent(event)
        if self.dragMode() == self.RubberBandDrag:
            # if event.buttons() == Qt.LeftButton:
            self.end_pos = event.pos()
            # print(self.origin_pos.x(), self.origin_pos.y())

            self.set_ROI_draw()

    def set_ROI_draw(self):

        img = self.img.copy()
        r1 = 0
        c1 = 0
        r2 = img.shape[0]
        c2 = img.shape[1]

        if self.origin_pos != None:
            # print(self.origin_pos)
            # print(self.end_pos)
            self.scenePos1 = self.mapToScene(self.origin_pos).toPoint()
            c1 = max(0, self.scenePos1.x())
            r1 = max(0, self.scenePos1.y())

            self.scenePos2 = self.mapToScene(self.end_pos).toPoint()
            c2 = min(img.shape[1], self.scenePos2.x())
            r2 = min(img.shape[0], self.scenePos2.y())
            # print(r1,r2,c1,c2)
            if r2-r1<2 or c2-c1<2:
                r1 = 0
                c1 = 0
                r2 = img.shape[0]
                c2 = img.shape[1]

        cv2.rectangle(img, (c1, r1), (c2, r2), (0, 0, 255), 5)

        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]
                        * img.shape[2], QImage.Format_RGB888).rgbSwapped()
        self.setPhoto(QPixmap(qimg))

    def get_roi_coordinate(self):
        
        img = self.img
        roi_coor = ROI_coordinate()

        if self.scenePos1 == None:

            roi_coor.r1 = 0
            roi_coor.c1 = 0
            roi_coor.r2 = img.shape[0]
            roi_coor.c2 = img.shape[1]

        else:

            self.fitInView()
            self.origin_pos = self.mapFromScene(self.scenePos1)
            self.end_pos = self.mapFromScene(self.scenePos2)

            # fitInView 要重新生座標才不會有誤差
            self.scenePos1 = self.mapToScene(self.origin_pos).toPoint()
            c1 = max(0, self.scenePos1.x())
            r1 = max(0, self.scenePos1.y())

            self.scenePos2 = self.mapToScene(self.end_pos).toPoint()
            c2 = min(img.shape[1], self.scenePos2.x())
            r2 = min(img.shape[0], self.scenePos2.y())

            if r2-r1<2 or c2-c1<2:
                r1 = 0
                c1 = 0
                r2 = img.shape[0]
                c2 = img.shape[1]

            roi_coor.r1 = r1
            roi_coor.c1 = c1
            roi_coor.r2 = r2
            roi_coor.c2 = c2

        return roi_coor

class ROI_Select_Window(QtWidgets.QWidget):
    to_main_window_signal = pyqtSignal(list, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.filefolder = "./"
        self.tab_idx = -1
        self.filename = ""

        # Widgets
        self.my_viewer = ImageViewer(self, 0)
        self.target_viewer = ImageViewer(self, 1)
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText('按下Ctrl可以使用滑鼠縮放拖曳\n左圖為拍攝的照片，右圖為目標照片\n對左邊的圖操作可同時操作兩張圖')
        self.btn_OK = QtWidgets.QPushButton(self)
        self.btn_OK.setText("OK")
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        HBlayout = QtWidgets.QHBoxLayout()

        HBlayout.addWidget(self.my_viewer)
        HBlayout.addWidget(self.target_viewer)

        VBlayout.addWidget(self.label)
        VBlayout.addLayout(HBlayout)
        VBlayout.addWidget(self.btn_OK)

        # # 接受信號後要連接到什麼函數(將值傳到什麼函數)
        self.btn_OK.clicked.connect(lambda: self.btn_ok_function())

        self.setStyleSheet(
            "QWidget{background-color: rgb(66, 66, 66);}"
            "QLabel{font-size:12pt; font-family:微軟正黑體; color:white;}"
            "QPushButton{font-size:12pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}")

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key_Control:
            self.my_viewer.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.target_viewer.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key_Control:
            self.my_viewer.setDragMode(self.target_viewer.RubberBandDrag)
            self.target_viewer.setDragMode(self.target_viewer.RubberBandDrag)

    def open_img(self):
        filepath, filetype = QFileDialog.getOpenFileName(
            self,
            "選擇target照片",
            self.filefolder,  # start path
            'Image Files(*.png *.jpg *.jpeg *.bmp)'
        )

        if filepath == '': return

        self.filefolder = '/'.join(filepath.split('/')[:-1])
        self.filename = filepath.split('/')[-1]
        
        # load img
        img = cv2.imdecode(np.fromfile(file=filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.target_viewer.set_img(img)

    def select_ROI(self):
        
        if len(self.target_viewer.img)==0: self.open_img()
        self.my_viewer.set_ROI_draw()
        self.target_viewer.set_ROI_draw()
        self.showMaximized()
    
    def roi_coor2x_y_w_h(self, roi_coor):
        return [roi_coor.c1, roi_coor.r1, (roi_coor.c2-roi_coor.c1), (roi_coor.r2-roi_coor.r1)]

    def btn_ok_function(self):
        my_roi_coor = self.my_viewer.get_roi_coordinate()
        target_roi_coor = self.target_viewer.get_roi_coordinate()

        x, y, w, h = self.roi_coor2x_y_w_h(target_roi_coor)
        target_roi_img = self.target_viewer.img[y: y+h, x:x+w]

        x, y, w, h = self.roi_coor2x_y_w_h(my_roi_coor)
        my_roi_img = self.my_viewer.img[y: y+h, x:x+w]

        self.close()

        self.to_main_window_signal.emit([x, y, w, h], my_roi_img, target_roi_img)
        

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ROI_Select_Window()
    sys.exit(app.exec_())
