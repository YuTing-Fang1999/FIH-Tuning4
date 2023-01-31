from PyQt5.QtWidgets import (
    QApplication, QMainWindow,
    QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QLineEdit, QCheckBox
)
from PyQt5.QtWidgets import QComboBox

class MethodSelector(QComboBox):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("font-size:12pt; font-family:微軟正黑體; background-color: rgb(255, 255, 255);")

        item_names = ["globel search", "local search"]
        self.clear()
        self.addItems(item_names) # -> set_trigger_idx 0

        

class HyperSettingBlock(QWidget):
    def __init__(self):
        super().__init__()
        self.lineEdits_hyper_setting = []
        self.setup_UI()
        
    def setup_UI(self):
        VLayout = QVBoxLayout(self)

        gridLayout = QGridLayout()
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.setHorizontalSpacing(7)

        title_wraper = QHBoxLayout()
        label_title = QLabel("Hyper Parameters")
        label_title.setStyleSheet("background-color:rgb(72, 72, 72);")
        title_wraper.addWidget(label_title)

        self.hyper_param_name = ["population size", "generations", "capture num"] # "F", "Cr", 
        tip = ["要初始化幾組參數(不可小於5)\n實際使用建議為10", "總共跑幾輪", "每次計算分數時要拍幾張照片"] # "變異的程度(建議不要超過1)", "替換掉參數的比率(建議不要超過0.5)", 
        self.hyper_param_title = self.hyper_param_name
        for i in range(len(self.hyper_param_name)):
            label = QLabel(self.hyper_param_name[i])

            lineEdit = QLineEdit()
            label.setToolTip(tip[i])
            self.lineEdits_hyper_setting.append(lineEdit)

            gridLayout.addWidget(label, i, 0)
            gridLayout.addWidget(lineEdit, i, 1)

        VLayout.addLayout(title_wraper)
        VLayout.addLayout(gridLayout)

        self.method_selector = MethodSelector()
        self.method_intro = QLabel()
        self.method_selector.currentIndexChanged[int].connect(self.set_idx)
        self.set_idx(0)

        VLayout.addWidget(self.method_selector)
        VLayout.addWidget(self.method_intro)

    def set_idx(self, idx):
        if idx==0:
            self.method_intro.setText("globel search\n使用差分進化演算法\n隨機重新產生")
        if idx==1:
            self.method_intro.setText("globel search\n使用Nelder-Mead Simplex\n用前一個gain的參數當做初始化參數\n(不能用於第一個gain)")



