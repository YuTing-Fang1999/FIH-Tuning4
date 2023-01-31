from PyQt5.QtWidgets import QHBoxLayout, QWidget, QMessageBox
from PyQt5.QtCore import pyqtSignal, QThread, QObject, Qt

from .HyperOptimizer import HyperOptimizer
from .MplCanvasTiming import MplCanvasTiming
from .ImageMeasurement import *
from myPackage.ML.ML import ML

import numpy as np
from time import sleep
from subprocess import call
import xml.etree.ElementTree as ET
from scipy import stats  # zcore
import os
import sys
import cv2
import math
import threading
import shutil
import json
from random import randrange
import csv
from decimal import Decimal
import copy

class Tuning(QObject):  # 要繼承QWidget才能用pyqtSignal!!
    finish_signal = pyqtSignal()
    # UI
    set_score_signal = pyqtSignal(str)
    set_generation_signal = pyqtSignal(str)
    set_individual_signal = pyqtSignal(str)
    set_statusbar_signal = pyqtSignal(str)
    # param window
    show_param_window_signal = pyqtSignal()
    setup_param_window_signal = pyqtSignal(int, int, np.ndarray) # popsize, param_change_num, IQM_names
    update_param_window_signal = pyqtSignal(int, np.ndarray, float, np.ndarray)
    update_param_window_scores_signal = pyqtSignal(list)
    # logger
    log_info_signal = pyqtSignal(str)
    run_cmd_signal = pyqtSignal(str)
    alert_info_signal = pyqtSignal(str, str)

    def __init__(self, logger, run_page_lower_part, setting, config, capture, set_param_value, build_and_push, get_file_path):
        super().__init__()
        self.logger = logger
        self.run_page_lower_part = run_page_lower_part
        self.tab_info = self.run_page_lower_part.tab_info
        self.set_param_value = set_param_value
        self.build_and_push = build_and_push
        self.get_file_path = get_file_path
        
        self.setting = setting
        self.config = config
        self.capture = capture
        self.is_run = False
        self.TEST_MODE = False
        self.PRETRAIN = False
        self.TRAIN = False

        self.calFunc = get_cal_func()

        # plot
        self.bset_score_plot = MplCanvasTiming(
            self.run_page_lower_part.tab_score.label_plot, color=['r', 'g'], name=['score'], axis_name=["Generation", "Score"]
        )
        self.hyper_param_plot = MplCanvasTiming(
            self.run_page_lower_part.tab_hyper.label_plot, color=['g', 'r'], name=['F', 'Cr'], axis_name=["Generation", "HyperParam Value"]
        )
        self.loss_plot = MplCanvasTiming(
            self.run_page_lower_part.tab_loss.label_plot, color=['b','g', 'r'], name=['loss'], axis_name=["Epoch/10", "Loss"]
        )
        self.update_plot = MplCanvasTiming(
            self.run_page_lower_part.tab_update.label_plot, color=['b', 'k'], name=['using ML', 'no ML'], axis_name=["Generation", "Update Rate"]
        )

        self.ML = ML(self.loss_plot)


    
    def run(self):
        self.is_rule = False
        self.TEST_MODE = self.setting["TEST_MODE"]
        self.PRETRAIN = self.setting["PRETRAIN"]
        self.TRAIN = self.setting["TRAIN"]

        ##### param setting #####
        self.key = self.setting["key"]
        self.key_config = self.config[self.setting["platform"]][self.setting["root"]][self.setting["key"]]
        self.key_data = self.setting[self.setting["root"]][self.setting["key"]]
        self.file_path = self.get_file_path[self.setting["platform"]](self.setting["project_path"], self.key_config["file_path"])

        # config
        self.rule = self.key_config["rule"]
        self.step = self.key_config["step"]
        
        # project setting
        self.platform = self.setting["platform"]
        self.exe_path = self.setting["exe_path"]
        self.project_path = self.setting["project_path"]
        self.bin_name = self.setting["bin_name"]

        # hyperparams
        self.popsize = self.setting['population size']
        self.generations = self.setting['generations']
        self.capture_num = self.setting['capture num']
        # self.Cr_optimiter = HyperOptimizer(init_value=0.8, final_value=0.4, method="exponantial_reverse", rate = 0.03)
        # self.F_optimiter = HyperOptimizer(init_value=0.4, final_value=0.8, method="exponantial_reverse", rate=0.03)
        self.F_optimiter = HyperOptimizer(init_value=0.7, final_value=0.7, method="constant")
        self.Cr_optimiter = HyperOptimizer(init_value=0.5, final_value=0.5, method="constant")
        
        self.bounds = []
        col = sum(self.key_config["col"], [])
        for i in range(len(col)):
            self.bounds.append([self.key_data["coustom_range"][i]]*col[i])
        self.bounds = sum(self.bounds, [])
        print(self.bounds)

        self.param_value = np.array(self.key_data['param_value']) # 所有參數值
        self.dimensions = len(self.param_value)
        self.param_change_idx = self.key_data['param_change_idx'] # 需要tune的參數位置
        self.param_change_num = len(self.param_change_idx) # 需要tune的參數個數
        
        self.trigger_idx = self.setting["trigger_idx"]
        self.trigger_name = self.setting["trigger_name"]
        # test mode 下沒改動的地方為0
        if self.TEST_MODE: self.param_value = np.zeros(self.dimensions)

        # target score
        if self.TEST_MODE:
            self.setting["target_type"]=["TEST", "TEST2", "TEST3"]
            self.setting["target_score"]=[0]*len(self.setting["target_type"])
            self.setting["target_weight"]=[1]*len(self.setting["target_type"])

        self.target_type = np.array(self.setting["target_type"])
        self.target_IQM = np.array(self.setting["target_score"])
        self.weight_IQM = np.array(self.setting["target_weight"])
        self.target_num = len(self.target_type)
        self.std_IQM = np.ones(self.target_num)
        self.loss_plot.setup(self.target_type)

        # target region
        self.roi = self.setting['roi']

        # 退火
        self.T = 0.8
        self.T_rate = 0.95
        if self.TRAIN: self.T=10

        # get the bounds of each parameter
        self.min_b, self.max_b = np.asarray(self.bounds).T
        self.min_b = self.min_b[self.param_change_idx]
        self.max_b = self.max_b[self.param_change_idx]
        self.diff = np.fabs(self.min_b - self.max_b)

        # self.pop = [self.param_value[self.param_change_idx]]*self.popsize
        # for i in range(self.popsize):
        #     self.pop[i] += np.random.uniform(-0.1, 0.2, self.param_change_num)
        #     self.pop[i] = np.clip(self.pop[i], 0, 1)

        # self.pop = np.random.random((self.popsize, self.param_change_num))
        # self.pop = self.min_b + self.pop * self.diff #denorm
        # self.pop = self.round_nearest(self.pop)
        # self.log_info_signal.emit(str(self.pop))
        # self.pop = ((self.pop-self.min_b)/self.diff) #norm
        
        # score
        self.best_score = 1e9
        self.fitness = []  # 計算popsize個individual所產生的影像的分數
        self.IQMs = []

        # update rate
        self.update_count=0
        self.ML_update_count=0
        self.update_rate=0
        self.ML_update_rate=0

        if len(self.setting["target_type"])==0:
            self.alert_info_signal.emit("請先圈ROI", "請先圈ROI")
            self.finish_signal.emit()
            return

        if self.param_change_num==0:
            self.alert_info_signal.emit("請選擇要調的參數", "目前參數沒打勾\n請打勾要調的參數")
            self.finish_signal.emit()
            return

        ##### start tuning #####
        # setup
        self.show_info()
        self.setup()

        ##### Nelder-Mead algorithm #####
        step = 0.1
        max_iter = 50
        no_improve_thr = 1e-5
        no_improv_break = 10
        alpha=1
        gamma=2
        rho=-0.5
        sigma=0.5

        # init
        now_IQM = self.measure_IQM_by_param_value('log/init', self.param_value, train=False)
        self.IQMs.append(now_IQM)
        prev_best = self.cal_score_by_weight(now_IQM)
        
        no_improv = 0
        dim = self.param_change_num
        x_start = (self.param_value[self.param_change_idx]-self.min_b)/self.diff #norm
        res = [[x_start, prev_best]]
        self.update_param_window_signal.emit(0, self.param_value[self.param_change_idx], prev_best, now_IQM)

        for i in range(dim):
            x = copy.copy(x_start)
            x[i] = x[i] + step
            self.param_value[self.param_change_idx] = self.min_b + x * self.diff #denorm
            now_IQM = self.measure_IQM_by_param_value('log/init_{}'.format(i), self.param_value, train=False)
            self.IQMs.append(now_IQM)
            score = self.cal_score_by_weight(now_IQM)
            self.log_info_signal.emit('res: {}'.format([x, score]))
            self.update_param_window_signal.emit(i+1, self.param_value[self.param_change_idx], score, now_IQM)
            res.append([x, score])

        # 更新經由標準差後的分數
        self.std_IQM = np.array(self.IQMs).std(axis=0)
        fitness = []
        for i in range(dim+1):
            res[i][1] = self.cal_score_by_weight(self.IQMs[i])
            fitness.append(res[i][1])
        self.update_param_window_scores_signal.emit(fitness)

        

        # simplex iter
        iters = 0
        while 1:
            # order
            res.sort(key=lambda x: x[1])
            best = res[0][1]

            # break after max_iter
            if max_iter and iters >= max_iter:
                return res[0]
            iters += 1

            # break after no_improv_break iterations with no improvement
            self.log_info_signal.emit('\n...best so far: {}\n'.format(best))
            self.update_best_score(best)

            if best < prev_best - no_improve_thr:
                no_improv = 0
                prev_best = best
            else:
                no_improv += 1

            if no_improv >= no_improv_break:
                self.log_info_signal.emit('...best so far: {}'.format(best))
                print('iters', iters)
                return res[0]

            # centroid
            x0 = [0.] * dim
            for tup in res[:-1]:
                for i, c in enumerate(tup[0]):
                    x0[i] += c / (len(res)-1)

            # reflection
            xr = x0 + alpha*(x0 - res[-1][0])
            self.param_value[self.param_change_idx] = self.min_b + xr * self.diff #denorm
            now_IQM = self.measure_IQM_by_param_value('log/iter_{}_reflection'.format(iters), self.param_value, train=False)
            rscore = self.cal_score_by_weight(now_IQM)
            if res[0][1] <= rscore < res[-2][1]:
                del res[-1]
                res.append([xr, rscore])
                self.update_param_window_signal.emit(self.param_change_num, self.param_value[self.param_change_idx], rscore, now_IQM)
                continue

            # expansion
            if rscore < res[0][1]:
                xe = x0 + gamma*(x0 - res[-1][0])
                self.param_value[self.param_change_idx] = self.min_b + xe * self.diff #denorm
                now_IQM = self.measure_IQM_by_param_value('log/iter_{}_expansion'.format(iters), self.param_value, train=False)
                escore = self.cal_score_by_weight(now_IQM)
                if escore < rscore:
                    del res[-1]
                    res.append([xe, escore])
                    self.update_param_window_signal.emit(self.param_change_num, self.param_value[self.param_change_idx], escore, now_IQM)
                    continue
                else:
                    del res[-1]
                    res.append([xr, rscore])
                    self.update_param_window_signal.emit(self.param_change_num, self.param_value[self.param_change_idx], rscore, now_IQM)
                    continue

            # contraction
            xc = x0 + rho*(x0 - res[-1][0])
            self.param_value[self.param_change_idx] = self.min_b + xc * self.diff #denorm
            now_IQM = self.measure_IQM_by_param_value('log/iter_{}_contraction'.format(iters), self.param_value, train=False)
            cscore = self.cal_score_by_weight(now_IQM)
            if cscore < res[-1][1]:
                del res[-1]
                res.append([xc, cscore])
                self.update_param_window_signal.emit(self.param_change_num, self.param_value[self.param_change_idx], cscore, now_IQM)
                continue

            # reduction
            x1 = res[0][0]
            nres = []
            for tup_i, tup in enumerate(res):
                redx = x1 + sigma*(tup[0] - x1)
                self.param_value[self.param_change_idx] = self.min_b + redx * self.diff #denorm
                now_IQM = self.measure_IQM_by_param_value('log/iter_{}_reduction_{}'.format(iters, tup_i), self.param_value, train=False)
                score = self.cal_score_by_weight(now_IQM)
                nres.append([redx, score])
                self.update_param_window_signal.emit(tup_i, self.param_value[self.param_change_idx], score, now_IQM)
            res = nres
        
        self.finish_signal.emit()

    def update_best_score(self, score):
        # update log score
        self.best_score = np.round(score, 9)
        self.set_score_signal.emit(str(self.best_score))
        

    def measure_IQM_by_param_value(self, path, param_value, train):
        if self.TEST_MODE: 
            if self.TRAIN and train:
                self.start_ML_train()
                self.train_task.join()
            return np.array([self.fobj(param_value)]*len(self.target_type))

        # write param_value to xml
        self.set_param_value[self.platform](self.key, self.key_config, self.file_path, self.trigger_idx, param_value)

        # compile project using bat. push bin code to camera
        self.log_info_signal.emit('push bin to camera...')
        self.run_cmd_signal.emit('adb shell input keyevent = KEYCODE_HOME')
        self.build_and_push[self.platform](self.logger, self.exe_path, self.project_path, self.bin_name)
        self.capture.clear_camera_folder()
        self.log_info_signal.emit('wait for reboot camera...')
        
        if self.TRAIN and train: self.start_ML_train()
        sleep(6)

        # 拍照
        self.capture.capture(path, capture_num=self.capture_num)

        # 計算分數
        now_IQM = self.measure_score_by_multiple_capture(path)

        # 等ML train完再繼續
        if self.TRAIN and train: self.train_task.join()
        
        return now_IQM

    def measure_score_by_multiple_capture(self, path):
        IQM_scores = []
        # print("\n---------IQM score--------\n")
        for i in range(self.capture_num):
            # 讀取image檔案
            if self.capture_num==1: p = str(path+".jpg")
            else: p = path+"_"+str(i)+".jpg"
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            IQM_scores.append(self.calIQM(img))
            # print(i, IQM_scores[i])

        IQM_scores = np.array(IQM_scores)
        zscore = stats.zscore(IQM_scores)
        # print("\n---------zscore--------\n")
        # for i in range(capture_num):
        # print(i, zscore[i])
        select = (np.abs(zscore) < 1).all(axis=1)
        # print(select)

        # print("\n---------after drop outlier => [abs(zscore) > 1]--------\n")
        if (select == True).any():
            IQM_scores = IQM_scores[select]
        # print(IQM_scores)

        # 計算分數
        now_IQM = np.mean(IQM_scores, axis=0)
        return now_IQM

    def calIQM(self, img):
        now_IQM=[]
        for i, roi in enumerate(self.roi):
            if len(roi)==0: continue
            x, y, w, h = roi
            roi_img = img[y: y+h, x:x+w]
            now_IQM.append(self.calFunc[self.target_type[i]](roi_img))
        
        now_IQM = np.array(now_IQM)
        return now_IQM

    def cal_score_by_weight(self, now_IQM):
        if self.TEST_MODE: return np.mean(now_IQM)
        return (np.abs(self.target_IQM-now_IQM)/self.std_IQM).dot(self.weight_IQM.T)

    def mkdir(self, path):
        if self.TEST_MODE: 
            print('mkdir {} return because TEST_MODE'.format(path))
            return
        if not os.path.exists(path):
            os.makedirs(path)
            print("The", path, "dir is created!")

    def setup(self):
        # reset plot
        self.bset_score_plot.reset()
        self.hyper_param_plot.reset()
        self.loss_plot.reset()
        self.update_plot.reset()

        # reset label
        self.set_score_signal.emit("#")
        self.set_generation_signal.emit("#")
        self.set_individual_signal.emit("#")

        self.setup_param_window_signal.emit(self.param_change_num+1, self.param_change_num, self.target_type)

    # Ackley
    # objective function
    def fobj(self, X):
        firstSum = 0.0
        secondSum = 0.0
        for c in X:
            firstSum += c**2.0
            secondSum += math.cos(2.0*math.pi*c)
        n = float(len(X))
        return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e


    def curve_converter(self, x, a):
        if (a==0).all(): return [0.996]*len(x)
        return (1-np.exp(-(x/a)**3.96))*0.996

    def show_info_by_key(self, key, data):
        for k in key:
            self.tab_info.show_info("{}: {}".format(k,data[k]))

    def show_info(self):
        # show info
        self.tab_info.label.setAlignment(Qt.AlignLeft)
        self.tab_info.clear()

        key_config = self.config[self.setting["platform"]][self.setting["root"]][self.setting["key"]]
        key_data = self.setting[self.setting["root"]][self.setting["key"]]

        self.tab_info.show_info("\n###### Target ######")
        self.show_info_by_key(["target_type", "target_score", "target_weight"], self.setting)

        self.tab_info.show_info("\n###### Tuning Block ######")
        self.show_info_by_key(["root", "key"], self.setting)
        self.show_info_by_key(["trigger_idx", "trigger_name"], self.setting)

        self.tab_info.show_info("\n###### Differential evolution ######")
        self.show_info_by_key(["population size","generations","capture num"], self.setting)
        self.show_info_by_key(["coustom_range","param_change_idx"], key_data)
        self.tab_info.show_info("{}: {}".format("param_value", self.param_value))

        self.tab_info.show_info("\n###### Mode ######")
        self.show_info_by_key(["TEST_MODE","PRETRAIN","TRAIN"], self.setting)

        self.tab_info.show_info("\n###### Project Setting ######")
        self.show_info_by_key(["platform", "project_path", "exe_path", "bin_name"], self.setting)




