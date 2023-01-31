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

        self.pop = np.random.random((self.popsize, self.param_change_num))
        self.pop = self.min_b + self.pop * self.diff #denorm
        self.pop = self.round_nearest(self.pop)
        self.log_info_signal.emit(str(self.pop))
        self.pop = ((self.pop-self.min_b)/self.diff) #norm
        


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

        # csv data
        title = ["name", "score"]
        for t in self.target_type: title.append(t)
        title.append(self.key_config["param_names"])
        self.csv_data = [title]
        self.best_csv_data = [title]

        # csv target
        data = ["target", 0]
        for IQM in self.target_IQM: data.append(IQM)
        title.append("")
        self.csv_data.append(data)
        self.best_csv_data.append(data)

        ##### start tuning #####
        # setup
        self.show_info()
        self.setup()
        self.initial_individual()

        # ML
        self.ML.reset(
            TEST_MODE = self.TEST_MODE,
            PRETRAIN=self.PRETRAIN, 
            TRAIN=self.TRAIN, 
            
            target_type = self.target_type,
            std_IQM = self.std_IQM,
            key = self.setting["key"],
            
            input_dim=self.dimensions, 
            output_dim=len(self.target_type)
        )

        # Do Differential Evolution
        for gen_idx in range(self.generations):
            self.run_DE_for_a_generation(gen_idx)
        
        self.finish_signal.emit()


    def initial_individual(self):
        if not self.TEST_MODE:
            # 刪除資料夾
            if os.path.exists('log'): shutil.rmtree('log')
            if os.path.exists('best_photo'): shutil.rmtree('best_photo')
            self.mkdir('log')
            # self.mkdir('log/xml')
            self.mkdir('best_photo')
        
        # self.mkdir('log/img')
        # self.mkdir('log/init')
        self.set_generation_signal.emit("initialize")

        # initial individual
        for ind_idx in range(self.popsize):
            self.set_individual_signal.emit(str(ind_idx))
            self.log_info_signal.emit('\ninitial individual: {}'.format(ind_idx))

            # denormalize to [min_b, max_b]
            trial_denorm = self.min_b + self.pop[ind_idx] * self.diff
            # update param_value
            self.param_value[self.param_change_idx] = trial_denorm
            # measure score
            now_IQM = self.measure_score_by_param_value('log/ind{}_init'.format(ind_idx), self.param_value, train=False)
            self.fitness.append(np.around(self.cal_score_by_weight(now_IQM), 9))
            self.IQMs.append(now_IQM)
            # self.log_info_signal.emit('now IQM {}'.format(now_IQM))
            # self.log_info_signal.emit('now score {}'.format(self.fitness[ind_idx]))

            # update_param_window
            self.update_param_window_signal.emit(ind_idx, trial_denorm, self.fitness[ind_idx], now_IQM)

            
            if not self.TEST_MODE:
                # 儲存xml
            #     des="log/xml/init_ind{}.xml".format(ind_idx)
            #     shutil.copyfile(self.xml_path, des)
                # csv data
                data = ["ind{}_init".format(ind_idx), 0]
                for IQM in now_IQM: data.append(IQM)
                data.append(trial_denorm)
                self.csv_data.append(data)
                self.best_csv_data.append(data)


        self.IQMs = np.array(self.IQMs)
        self.std_IQM = self.IQMs.std(axis=0)
        # 暫時設成1
        # self.std_IQM = np.ones(self.target_num)
        # 依據標準差重新計算
        for ind_idx in range(self.popsize):
            self.fitness[ind_idx] = np.around(self.cal_score_by_weight(self.IQMs[ind_idx]), 5)
            if not self.TEST_MODE:
                self.csv_data[ind_idx+2][1] = self.fitness[ind_idx]
                self.best_csv_data[ind_idx+2][1] = self.fitness[ind_idx]

            # 將圖片複製到best資料夾
            if not self.TEST_MODE:
                for i in range(self.capture_num):
                    if self.capture_num==1:
                        src_img = 'ind{}_init.jpg'.format(ind_idx)
                        des_img = '{}.jpg'.format(ind_idx) 
                        
                    else:
                        src_img = 'ind{}_{}_init.jpg'.format(ind_idx, i)
                        des_img = '{}_{}.jpg'.format(ind_idx, i) 

                    src='log/{}'.format(src_img)
                    des='best_photo/{}'.format(des_img) 

                    if os.path.exists(des): os.remove(des)
                    shutil.copyfile(src, des)

            # 儲存json
            # info = {
            #     "target_type": self.target_type.tolist(),
            #     "target_IQM": self.target_IQM.tolist(),
            #     "now_IQM": self.IQMs[ind_idx].tolist(),
            #     "name": 'init{}'.format(ind_idx),
            #     "param_block": self.key,
            #     "trigger_block": self.trigger_name,
            # }
            # with open('log/{}.json'.format(self.fitness[ind_idx]), "w") as outfile:
            #     outfile.write(json.dumps(info, indent=4))

            if self.fitness[ind_idx] < self.best_score:
                self.update_best_score(ind_idx, self.fitness[ind_idx])

        # 更新經由標準差後的分數
        self.update_param_window_scores_signal.emit(self.fitness)
        # 暫時將std設為1
        # self.std_IQM = self.std_IQM = np.ones(self.target_num)
        print('std_IQM',self.std_IQM)

        if not self.TEST_MODE:
            with open('log/init.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.csv_data)
            with open('best_photo/init.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.best_csv_data)

        # shutil.rmtree("log/init")

    def run_DE_for_a_generation(self, gen_idx):
        self.set_generation_signal.emit(str(gen_idx))

        # create dir
        gen_dir = 'generation{}'.format(gen_idx)
        # self.mkdir(gen_dir)

        # update hyperparam
        F = self.F_optimiter.update(gen_idx)
        Cr = self.Cr_optimiter.update(gen_idx)

        self.update_count=0
        self.ML_update_count=0

        for ind_idx in range(self.popsize):
            self.run_DE_for_a_individual(F, Cr, gen_idx, ind_idx, gen_dir)
        
        self.update_rate=self.update_count/self.popsize
        self.ML_update_rate=self.ML_update_count/self.popsize

        if not self.TEST_MODE:
            with open('log/gen{}.csv'.format(gen_idx), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.csv_data)

            with open('best_photo/gen{}.csv'.format(gen_idx), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.best_csv_data)
        

    def run_DE_for_a_individual(self, F, Cr, gen_idx, ind_idx, gen_dir):
        self.set_individual_signal.emit(str(ind_idx))

        is_return, trial, trial_denorm = self.generate_parameters(ind_idx, F, Cr)
        if(is_return): return
        
        bad_time = 0
        while(self.is_bad_trial(trial_denorm)):
            is_return, trial, trial_denorm = self.generate_parameters(ind_idx, F, Cr)
            if(is_return or bad_time>20): 
                self.log_info_signal.emit("\nreturn because good trial not found\n")
                return
            bad_time+=1

        self.log_info_signal.emit("\nbad_time: {} trial_denorm: {}\n".format(bad_time, trial_denorm))

        # update param_value
        self.param_value[self.param_change_idx] = trial_denorm

        if self.TEST_MODE:
            # mesure score
            # self.log_info_signal.emit("generations:{}, individual:{}".format(gen_idx, ind_idx))
            now_IQM = self.measure_score_by_param_value('log/ind{}_gne{}'.format(ind_idx, gen_idx), self.param_value, train=False)
            f = np.around(self.cal_score_by_weight(now_IQM), 9)
            # self.log_info_signal.emit("now IQM {}".format(now_IQM))
            # self.log_info_signal.emit("now fitness {}".format(f))
            if (self.PRETRAIN or self.TRAIN) and f < self.fitness[ind_idx]: self.update_count+=1

        # use model to predict
        if (self.PRETRAIN or self.TRAIN) and gen_idx>=self.ML.pred_idx:
            trial, trial_denorm = self.get_best_trial(ind_idx, F, Cr, trial, trial_denorm)
            # times = 0
            # while self.is_bad(trial, ind_idx, times) and times<50:
            #     trial, trial_denorm = self.generate_parameters(ind_idx, F, Cr)
            #     times+=1
            # self.log_info_signal.emit("times: {}".format(times))                
            
        # update param_value
        self.param_value[self.param_change_idx] = trial_denorm

        # mesure score
        # self.log_info_signal.emit("generations:{}, individual:{}".format(gen_idx, ind_idx))
        now_IQM = self.measure_score_by_param_value('log/ind{}_gne{}'.format(ind_idx, gen_idx), self.param_value, train=gen_idx>=self.ML.train_idx)
        f = np.around(self.cal_score_by_weight(now_IQM), 9)
        # self.log_info_signal.emit("now IQM {}".format(now_IQM))
        # self.log_info_signal.emit("now fitness {}".format(f))

        data = ['ind{}_gne{}'.format(ind_idx, gen_idx), f]
        for IQM in now_IQM: data.append(IQM)
        data.append(trial_denorm)
        self.csv_data.append(data)

        # update dataset
        if (self.PRETRAIN or self.TRAIN):
            x = np.zeros(self.dimensions)
            x[self.param_change_idx] = trial - self.pop[ind_idx]
            y = now_IQM - self.IQMs[ind_idx]
            self.ML.update_dataset(x, y)

        # 如果突變種比原本的更好
        if f < self.fitness[ind_idx]:
            # update_param_window
            self.update_param_window_signal.emit(ind_idx, trial_denorm, f, now_IQM)
            
            if (self.PRETRAIN or self.TRAIN): self.ML_update_count+=1
            else: self.update_count+=1

            # 替換原本的個體
            self.log_info_signal.emit('replace with better score {}'.format(f))
            self.set_statusbar_signal.emit('generation {} individual {} replace with better score'.format(gen_dir, ind_idx))
            
            self.fitness[ind_idx] = f
            self.IQMs[ind_idx] = now_IQM
            self.pop[ind_idx] = trial

            # 將圖片複製到best資料夾
            if not self.TEST_MODE:
                self.best_csv_data[ind_idx+2] = data
                for i in range(self.capture_num):
                    if self.capture_num==1:
                        src_img = 'ind{}_gne{}.jpg'.format(ind_idx, gen_idx)
                        des_img = '{}.jpg'.format(ind_idx) # 根據量化分數命名
                        
                    else:
                        src_img = 'ind{}_{}_gne{}.jpg'.format(ind_idx, i, gen_idx)
                        des_img = '{}_{}.jpg'.format(ind_idx, i) # 根據量化分數命名

                    src='log/{}'.format(src_img)
                    des='best_photo/{}'.format(des_img) # 根據量化分數命名

                    if os.path.exists(des): os.remove(des)
                    shutil.copyfile(src, des)
            # # 儲存json
            # info = {
            #     "target_type": self.target_type.tolist(),
            #     "target_IQM": self.target_IQM.tolist(),
            #     "now_IQM": now_IQM.tolist(),
            #     "score": f,
            #     "name": 'gne{}_ind{}'.format(gen_idx ,ind_idx),
            #     "param_block": self.key,
            #     "trigger_block": self.trigger_name,
            #     "param_name": self.param_names,
            #     "param_value": self.param_value.tolist(),
            # }
            # with open('log/{}.json'.format(f), "w") as outfile:
            #     outfile.write(json.dumps(info, indent=4))

            # 儲存xml
            # if not self.TEST_MODE:
            #     des="log/xml/gne{}_ind{}.xml".format(gen_idx, ind_idx)
            #     shutil.copyfile(self.xml_path, des)

            # 如果突變種比最優種更好
            if f < self.best_score:
                # 替換最優種
                self.update_best_score(ind_idx, f)
                    
                if f==0:
                    self.finish_signal.emit()
                    sys.exit()

        self.bset_score_plot.update([self.best_score])
        self.hyper_param_plot.update([F, Cr])
        self.update_plot.update([self.ML_update_rate, self.update_rate])

    def start_ML_train(self):
        # 建立一個子執行緒
        self.train_task = threading.Thread(target = lambda: self.ML.train())
        # 當主程序退出，該執行緒也會跟著結束
        self.train_task.daemon = True
        # 執行該子執行緒
        self.train_task.start()

    def get_best_trial(self, ind_idx, F, Cr, trial, trial_denorm):
        best_trial, best_trial_denorm = trial, trial_denorm
        best_good_num = self.target_num/2
        times = 0
        
        for i in range(20):
            is_return, trial, trial_denorm = self.generate_parameters(ind_idx, F, Cr)
            if is_return: break

            x = np.zeros(self.dimensions)
            x[self.param_change_idx] = trial - self.pop[ind_idx] # 參數差
            diff_target_IQM = self.target_IQM - self.IQMs[ind_idx] # 目標差
            # self.log_info_signal.emit(str(x))
            pred_dif_IQM = self.ML.predict(x)
            self.log_info_signal.emit("pred_dif_IQM: {}".format(pred_dif_IQM)) 
            good_num = np.sum(pred_dif_IQM * self.weight_IQM * diff_target_IQM > 0)
            self.log_info_signal.emit("good_num: {}".format(good_num)) 
            if good_num >= best_good_num:
                best_trial, best_trial_denorm = trial, trial_denorm
                self.log_info_signal.emit("get_best_trial times: {}".format(i))  
                return best_trial, best_trial_denorm
                # times = i
                # if good_num==self.target_num:
                #     break

            # bad_time = 0
            # while(self.is_bad_trial(trial_denorm) and bad_time<5):
            #     is_return, trial, trial_denorm = self.generate_parameters(ind_idx, F, Cr)
            #     # self.log_info_signal.emit('trial_denorm'+str(x))
            #     if(is_return): break
            #     bad_time+=1
        self.log_info_signal.emit("get_best_trial times: {}".format(times))  
        return best_trial, best_trial_denorm

    # def is_bad(self, trial, ind_idx, times):
    #     x = np.zeros(self.dimensions)
    #     x[self.param_change_idx] = trial - self.pop[ind_idx] # 參數差
    #     diff_target_IQM = self.target_IQM - self.IQMs[ind_idx] # 目標差
    #     pred_dif_IQM = self.ML.predict(x)
    #     ##### 更改判斷標準(bad大於半數) #####
    #     bad_num = np.sum(pred_dif_IQM * self.weight_IQM * diff_target_IQM <= 0)
    #     if times<10:
    #         return bad_num >= 1
    #     else: 
    #         return bad_num >= np.ceil(self.target_num/2)

    def is_bad_trial(self, trial_denorm):
        # self.log_info_signal.emit("is_bad_trial? {}".format(trial_denorm))
        if self.is_rule:
            for rule in self.rule:
                val = 0
                p0 = trial_denorm[rule["idx"][0]]
                p1 = trial_denorm[rule["idx"][1]]

                # self.log_info_signal.emit("idx0:{},   idx1:{}".format(rule["idx"][0], rule["idx"][1]))
                # self.log_info_signal.emit("p0:{},   p1:{}".format(p0, p1))


                for op in rule["op"]:
                    if op == '-': val = p0-p1
                    elif op == 'abs': val = np.abs(val)

                if not (rule["between"][0]<=val and val<=rule["between"][1]):
                    # self.log_info_signal.emit("not between-> p0:{},   p1:{}".format(p0, p1))
                    if val>rule["between"][1]: dif = -(val-rule["between"][1])
                    if val<rule["between"][0]: dif = (val-rule["between"][0])
                    p = math.exp(dif/self.T) # 差越大p越小，越容易傳True
                    # self.log_info_signal.emit("{},   {}".format(dif, p))
                    if p<np.random.uniform(0.5, 0.8): 
                        self.log_info_signal.emit("bad param")
                        return True # p越小越容易比他大
            
            self.T *= self.T_rate
        
        return False

    def round_nearest(self, x):
        return np.around(self.step*np.around(x/self.step), 2)

    def generate_parameters(self, ind_idx, F, Cr):
        mutant = None
        times = 0
        while (not isinstance(mutant, np.ndarray) or ((trial_denorm-now_denorm)<=1e-5).all()):
            
            # select all pop except j
            idxs = [idx for idx in range(self.popsize) if idx != ind_idx]
            # random select three pop except j
            a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
            vec = F * (b - c)
            # Mutation
            mutant = np.clip(a + vec, 0, 1)

            # random choose the dimensions
            cross_points = np.random.rand(self.param_change_num) < Cr
            # if no dimensions be selected
            if not np.any(cross_points):
                # random choose one dimensions
                cross_points[np.random.randint(0, self.param_change_num)] = True

            # random substitution mutation
            trial = np.where(cross_points, mutant, self.pop[ind_idx])

            # denormalize to [min_b, max_b]
            trial_denorm = self.min_b + trial * self.diff
            trial_denorm = self.round_nearest(trial_denorm)

            now_denorm = self.min_b + self.pop[ind_idx] * self.diff
            now_denorm = self.round_nearest(now_denorm)

            times+=1
            if times>20: return True, [], []

        self.log_info_signal.emit("generate_times: {}".format(times))
        trial = (trial_denorm-self.min_b)/self.diff

        return False, trial, trial_denorm


    def update_best_score(self, idx, score):
        # update log score
        self.best_score = np.round(score, 9)
        self.set_score_signal.emit(str(self.best_score))
        

    def measure_score_by_param_value(self, path, param_value, train):
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

        self.setup_param_window_signal.emit(self.popsize, self.param_change_num, self.target_type)

    # def push_to_phone(self, idx):
    #     if self.TEST_MODE:
    #         QMessageBox.about(self, "info", "TEST_MODE下不可推")
    #         print("TEST_MODE下不可推")
    #     # QMessageBox.about(self, "info", "功能未完善")
    #     # return

    #     if idx>=len(self.fitness):
    #         QMessageBox.about(self, "info", "目前無參數可推")
    #         return

    #     if self.is_run:
    #         QMessageBox.about(self, "info", "程式還在執行中\n請按stop停止\n或等執行完後再推")
    #         return
        
    #     # get normalized parameters
    #     trial = self.pop[idx]
    #     # denormalize to [min_b, max_b]
    #     trial_denorm = self.min_b + trial * self.diff
    #     # update param_value
    #     self.param_value[self.param_change_idx] = trial_denorm

    #     self.setParamToXML(self.param_value)
    #     # 使用bat編譯，將bin code推入手機
    #     self.buildAndPushToCamera()

    #     print('push individual', idx, 'to phone')
    #     print('param_value =', self.param_value)
    #     print("成功推入手機")

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


