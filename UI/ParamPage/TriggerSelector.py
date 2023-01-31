from PyQt5.QtWidgets import QComboBox
import xml.etree.ElementTree as ET
import json
from scipy.optimize import curve_fit
from numpy import arange

class TriggerSelector(QComboBox):
    def __init__(self):
        super().__init__()
        self.setup_UI()
        # self.setup_controller()

    def setup_UI(self):
        self.setStyleSheet("font-size:12pt; font-family:微軟正黑體; background-color: rgb(255, 255, 255);")

    def update_UI(self, aec_trigger_datas):
        item_names = ["lux_idx from {} to {},  gain from {} to {}".format(d[0], d[1], d[2], d[3])for d in aec_trigger_datas]
        self.clear()
        self.addItems(item_names) # -> set_trigger_idx 0
        

    

