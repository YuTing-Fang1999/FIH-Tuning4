from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
    QSpacerItem, QSizePolicy, QScrollArea, QLabel
)
from PyQt5.QtCore import Qt

from .TriggerSelector import TriggerSelector
from .ParamModifyBlock import ParamModifyBlock
from .ParamRangeBlock import ParamRangeBlock
from .HyperSettingBlock import HyperSettingBlock
from .PushAndSaveBlock import PushAndSaveBlock
from .ISP_Tree import ISP_Tree

import os
import xml.etree.ElementTree as ET

class ParamPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_UI()
        # self.setup_controller()

    def setup_UI(self):
        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        HLayout = QHBoxLayout()

        self.ISP_tree = ISP_Tree()
        HLayout.addWidget(self.ISP_tree)

        ###### Left Part ######
        VLayout = QVBoxLayout()
        VLayout.setContentsMargins(0, 0, 0, 0)

        self.trigger_selector = TriggerSelector()
        VLayout.addWidget(self.trigger_selector)

        self.param_modify_block = ParamModifyBlock()
        VLayout.addWidget(self.param_modify_block)

        self.push_and_save_block = PushAndSaveBlock()
        VLayout.addWidget(self.push_and_save_block)

        VLayout.addItem(spacerItem)
        
        HLayout.addLayout(VLayout)
        ###### Left Part ######

        ###### Middle Part ######
        VLayout = QVBoxLayout()
        self.param_range_block = ParamRangeBlock()
        VLayout.addWidget(self.param_range_block)
        VLayout.addItem(spacerItem)
        HLayout.addLayout(VLayout)
        ###### Middle Part ######

        ###### Right Part ######
        VLayout = QVBoxLayout()
        self.hyper_setting_block = HyperSettingBlock()
        VLayout.addWidget(self.hyper_setting_block)

        VLayout.addItem(spacerItem)
        HLayout.addLayout(VLayout)
        ###### Right Part ######

        #Scroll Area Properties
        scroll_wrapper = QHBoxLayout(self)
        layout_wrapper = QWidget()
        layout_wrapper.setLayout(HLayout)
        scroll = QScrollArea() 
        scroll.setWidgetResizable(True)
        scroll.setWidget(layout_wrapper)
        scroll_wrapper.addWidget(scroll)

        # Set Style
        self.setStyleSheet(
            "QLabel{font-size:12pt; font-family:微軟正黑體; color:white;}"
            "QPushButton{font-size:12pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}"
            "QLineEdit{font-size:10pt; font-family:微軟正黑體; background-color: rgb(255, 255, 255); border: 2px solid gray; border-radius: 5px;}"
        )
    
    def reset_UI(self):
        self.ISP_tree.reset_UI()
        self.trigger_selector.clear()
        self.param_modify_block.reset_UI()
        self.param_range_block.reset_UI()


    # def update_UI(self):
    #     self.logger = .logger
    #     self.data = .data
    #     self.config = .config

    #     root, key = self.data["page_root"], self.data["page_key"]
    #     self.logger.signal.emit('Change param page to {}/{}'.format(root, key))
    #     self.ISP_tree.update_UI()
    #     self.param_modify_block.update_UI(root, key)
    #     self.param_range_block.update_UI(root, key)
    #     self.hyper_setting_block.update_UI()
    #     self.push_and_save_block.update_UI()

    #     if "project_path" in self.data:
    #         if os.path.exists(self.data["project_path"]):
    #             self.set_project(self.data["project_path"])
    #         else:
    #             self.logger.show_info(self.data["project_path"]+" not found")

    # def setup_controller(self):
    #     pass
    #     # self.trigger_selector.currentIndexChanged[int].connect(self.set_trigger_idx)
    #     # self.ISP_tree.tree.itemClicked.connect(self.change_param_page)

    # def change_param_page(self, item, col):
    #     if item.parent() is None: 
    #         if item.isExpanded():item.setExpanded(False)
    #         else: item.setExpanded(True)
    #         return

    #     root = item.parent().text(0)
    #     key = item.text(0)
    #     # print('change param page to', root, key)
    #     self.param_modify_block.update_UI(root, key)
    #     self.param_range_block.update_UI(root, key)
    #     self.data["page_root"] = root
    #     self.data["page_key"] = key
    #     self.trigger_selector.set_trigger_idx(self.data["trigger_idx"])

    # def set_project(self, folder_path):
    #     self.logger.show_info('\nset_project')
    #     self.data['project_path'] = folder_path
    #     self.data['project_name'] = folder_path.split('/')[-1]
    #     self.data['tuning_dir'] = '/'.join(folder_path.split('/')[:-1])
    #     self.data['xml_path'] = folder_path + '/Scenario.Default/XML/'
    #     self.set_project_XML(self.data['xml_path'])

    # def set_project_XML(self, xml_path):
    #     self.logger.show_info("set_project_XML")
    #     if "page_root" not in self.data: 
    #         self.logger.show_info('Return because no page root')
    #         return
    #     if "page_key" not in self.data: 
    #         self.logger.show_info('Return because no page key')
    #         return
        
    #     config = self.config[self.data["page_root"]][self.data["page_key"]]

    #     xml_path+=config["file_path"]

    #     # 從檔案載入並解析 XML 資料
    #     if not os.path.exists(xml_path):
    #         self.logger.show_info('Return because no such file: '+xml_path)
    #         self.logger.show_info("請確認是否為"+self.data["platform"])
    #         .project_setting_page.label_project_path.setText("請確認"+self.data["project_path"]+"是否為"+self.data["platform"])
    #         return

    #     tree = ET.parse(xml_path)
    #     root = tree.getroot()

    #     # 子節點與屬性
    #     mod_wnr24_aec_datas  =  root.findall(config["xml_node"])
    #     # hdr_aec_data 下面有多組 gain 的設定 (mod_wnr24_aec_data)
    #     # 每組mod_wnr24_aec_data分別有 aec_trigger 與 wnr24_rgn_data
    #     # 其中 aec_trigger 代表在甚麼樣的ISO光源下觸發
    #     # wnr24_rgn_data 代表所觸發的參數

    #     aec_trigger_datas = []
    #     for ele in mod_wnr24_aec_datas:
    #         data = []
    #         aec_trigger = ele.find("aec_trigger")
    #         data.append(aec_trigger.find("lux_idx_start").text)
    #         data.append(aec_trigger.find("lux_idx_end").text)
    #         data.append(aec_trigger.find("gain_start").text)
    #         data.append(aec_trigger.find("gain_end").text)
    #         aec_trigger_datas.append(data)

    #     self.param_modify_block.update_UI(self.data["page_root"],self.data["page_key"])
    #     self.param_range_block.update_UI(self.data["page_root"],self.data["page_key"])
    #     self.trigger_selector.update_UI(aec_trigger_datas)
        
    #     self.logger.show_info("Load {} Successfully".format(self.data["project_name"]))



    
    
       



