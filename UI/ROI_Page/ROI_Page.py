from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QSpacerItem, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QFrame, QGridLayout,
    QPushButton, QLabel, QApplication, QCheckBox, QTableWidget, QHeaderView, QLineEdit,
    QTableWidgetItem, QFileDialog, QScrollArea
)    
import cv2
import numpy as np
from copy import deepcopy

import sys
sys.path.append(".")

from .ImageViewer import ImageViewer
from .ROI_Select_Window import ROI_Select_Window
from .MeasureWindow import MeasureWindow
import os
import random

class DeleteBtn(QPushButton):
    def __init__(self, table, page):
        super().__init__()
        self.table = table
        self.page = page
        self.setText("刪除")
        self.clicked.connect(self.deleteClicked)

    def deleteClicked(self):
        button = self.sender()
        if button:
            row = self.table.indexAt(button.pos()).row()
            self.table.removeRow(row)
            # print("del row", row)
            self.page.rois.pop(row)
            self.page.draw_ROI(self.page.rois)

class ROI_Page(QWidget):
    alert_info_signal = pyqtSignal(str, str)
    capture_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.filefolder="./"
        self.rois = []

        self.setup_UI()
        self.setup_controller()

    def setup_UI(self):

        # Widgets
        self.label_img = ImageViewer()
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.ROI_select_window = ROI_Select_Window()
        self.measure_window = MeasureWindow()

        self.table = QTableWidget()
        self.headers = ["type", "IQM", "weight", "刪除紐"]
        self.table.setColumnCount(len(self.headers))   ##设置列数
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.table.verticalHeader().setVisible(False)
        # self.table.setVerticalHeaderLabels(QLabel("id"))
        QTableWidget.resizeRowsToContents(self.table)


        self.btn_capture = QPushButton()
        self.btn_capture.setText("拍攝照片")
        self.btn_capture.setToolTip("會拍攝一張照片，請耐心等候")

        self.btn_load_target_pic = QPushButton("Load Target PIC")
        self.btn_load_target_pic.setToolTip("選擇目標照片")

        self.btn_add_ROI_item = QPushButton()
        self.btn_add_ROI_item.setText("增加ROI區域")
        self.btn_add_ROI_item.setToolTip("按下後會新增一個目標區域")

        self.GLayout = QGridLayout()

        # Arrange layout
        HLayout = QHBoxLayout()

        VBlayout = QVBoxLayout()
        VBlayout.addWidget(self.label_img)
        VBlayout.addWidget(self.btn_capture)
        VBlayout.addWidget(self.btn_load_target_pic)
        VBlayout.addWidget(self.btn_add_ROI_item)
        HLayout.addLayout(VBlayout)

        VBlayout = QVBoxLayout()
        VBlayout.addWidget(self.table)
        HLayout.addLayout(VBlayout)

        HLayout.setStretch(0,3)
        HLayout.setStretch(1,2)

        #Scroll Area Properties
        scroll_wrapper = QHBoxLayout(self)
        layout_wrapper = QWidget()
        layout_wrapper.setLayout(HLayout)
        scroll = QScrollArea() 
        scroll.setWidgetResizable(True)
        scroll.setWidget(layout_wrapper)
        scroll_wrapper.addWidget(scroll)

        self.setStyleSheet(
            "QWidget{background-color: rgb(66, 66, 66);}"
            "QLabel{font-size:12pt; font-family:微軟正黑體; color:white;}"
            "QPushButton{font-size:12pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}"
            "QLineEdit{font-size:12pt; font-family:微軟正黑體; background-color: rgb(255, 255, 255); border: 2px solid gray; border-radius: 5px;}")

    def setup_controller(self):
        self.ROI_select_window.to_main_window_signal.connect(self.select_ROI)
        self.measure_window.to_main_window_signal.connect(self.set_target_score)

        self.btn_add_ROI_item.clicked.connect(self.add_ROI_item_click)
        self.btn_load_target_pic.clicked.connect(self.load_target_img)

        ##### capture #####
        self.btn_capture.clicked.connect(lambda: self.capture_signal.emit("capture"))
        
    def select_ROI(self, my_x_y_w_h, my_roi_img, target_roi_img):
        self.measure_window.measure_target(my_x_y_w_h, my_roi_img, target_roi_img)

    def set_target_score(self, my_x_y_w_h, target_type, target_score):
        
        for i in range(len(target_type)):
            self.add_to_table(target_type[i], target_score[i], 1)
            self.rois.append(my_x_y_w_h)
            self.draw_ROI(self.rois)

    def add_to_table(self, target_type, target_score, target_weight):

        row = self.table.rowCount()
        self.table.setRowCount(row + 1)

        label = QLabel(target_type)
        label.setAlignment(Qt.AlignCenter)
        self.table.setCellWidget(row,0,label)

        self.table.setCellWidget(row,1,QLineEdit(str(target_score)))

        self.table.setCellWidget(row,2,QLineEdit(str(target_weight)))

        self.table.setCellWidget(row,3,DeleteBtn(self.table, self))
    
    def add_ROI_item_click(self):
        if len(self.ROI_select_window.my_viewer.img)==0:
            self.alert_info_signal.emit("未拍攝照片", "請先固定好拍攝位置，按拍攝鈕拍攝拍攝照片後，再選取區域")
            return
        
        if len(self.ROI_select_window.target_viewer.img)==0:
            self.alert_info_signal.emit("請先Load目標照片", "請先Load目標照片，再選取區域")
            return

        self.ROI_select_window.select_ROI()

    def draw_ROI(self, rois):
        img_select = self.img.copy()
        # print('draw_ROI', rois)
        for i, roi in enumerate(rois):
            if len(roi)==0: continue

            x, y, w, h = roi
            # 隨機產生顏色
            color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]
            cv2.rectangle(img_select, (x, y), (x+w, y+h), color, 10)
            cv2.putText(img_select, text=str(i+1), org=(x+w//2, y+h//2), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=min(w//50,h//50)+1, color=color, thickness=min(w//50,h//50)+1)

        self.label_img.setPhoto(img_select)

    def load_target_img(self):
        filepath, filetype = QFileDialog.getOpenFileName(
            self,
            "選擇target照片",
            self.filefolder,  # start path
            'Image Files(*.png *.jpg *.jpeg *.bmp)'
        )

        if filepath == '': return

        self.filefolder = '/'.join(filepath.split('/')[:-1])
        self.filename = filepath.split('/')[-1]

        self.btn_load_target_pic.setText("Load Target PIC ({})".format(self.filename))
        
        # load img
        img = cv2.imdecode(np.fromfile(file=filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.ROI_select_window.target_viewer.set_img(img)
        
    def set_photo(self, img_name):
        img = cv2.imread(img_name)
        self.img = img
        self.label_img.setPhoto(img)
        self.ROI_select_window.my_viewer.set_img(img)
        self.draw_ROI(self.rois)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = ROI_Page(None)
    window.showMaximized()
    sys.exit(app.exec_()) 