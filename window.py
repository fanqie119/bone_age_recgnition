# -*- coding: utf-8 -*-
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import functools
import math
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('骨龄评估系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/tubiao.jpg"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        # # 初始化视频读取线程
        # self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.model = self.model_load(weights="./weights/best.pt",
                                     device=self.device)  # todo 指明模型加载的位置的设备

#############################################################
        self.resize(1363, 708)
        self.horizontalLayoutWidget = QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QRect(970, 630, 371, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.button_bone_age_predict = QPushButton(self.horizontalLayoutWidget)
        self.button_bone_age_predict.setObjectName("button_bone_age_predict")
        self.horizontalLayout.addWidget(self.button_bone_age_predict)
        self.text_bone_age_predict = QLineEdit(self.horizontalLayoutWidget)
        self.text_bone_age_predict.setObjectName("text_bone_age_predict")
        self.horizontalLayout.addWidget(self.text_bone_age_predict)
        self.gridLayoutWidget = QWidget(self)
        self.gridLayoutWidget.setGeometry(QRect(970, 230, 371, 371))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label13 = QLabel(self.gridLayoutWidget)
        self.label13.setObjectName("label13")
        self.gridLayout.addWidget(self.label13, 20, 3, 1, 1)
        self.label_level = QLabel(self.gridLayoutWidget)
        self.label_level.setObjectName("label_level")
        self.gridLayout.addWidget(self.label_level, 0, 2, 1, 1)
        self.label_radius = QLabel(self.gridLayoutWidget)
        self.label_radius.setObjectName("label_radius")
        self.gridLayout.addWidget(self.label_radius, 5, 1, 1, 1)
        self.label_atlas = QLabel(self.gridLayoutWidget)
        self.label_atlas.setObjectName("label_atlas")
        self.gridLayout.addWidget(self.label_atlas, 0, 3, 1, 1)
        self.cbBox_ulna = QComboBox(self.gridLayoutWidget)
        self.cbBox_ulna.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        self.cbBox_ulna.setObjectName("cbBox_ulna")
        self.gridLayout.addWidget(self.cbBox_ulna, 2, 2, 1, 1)
        self.cbBox_first_yaunzhi = QComboBox(self.gridLayoutWidget)
        self.cbBox_first_yaunzhi.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_first_yaunzhi.setObjectName("cbBox_first_yaunzhi")
        self.gridLayout.addWidget(self.cbBox_first_yaunzhi, 17, 2, 1, 1)
        self.cbBox_fifth_yaunzhi = QComboBox(self.gridLayoutWidget)
        self.cbBox_fifth_yaunzhi.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_fifth_yaunzhi.setObjectName("cbBox_fifth_yaunzhi")
        self.gridLayout.addWidget(self.cbBox_fifth_yaunzhi, 20, 2, 1, 1)
        self.label_first_yuanzhi = QLabel(self.gridLayoutWidget)
        self.label_first_yuanzhi.setObjectName("label_first_yuanzhi")
        self.gridLayout.addWidget(self.label_first_yuanzhi, 17, 1, 1, 1)
        self.cbBox_radius = QComboBox(self.gridLayoutWidget)
        self.cbBox_radius.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_radius.setObjectName("cbBox_radius")
        self.gridLayout.addWidget(self.cbBox_radius, 5, 2, 1, 1)
        self.cbBox_third_jinzhi = QComboBox(self.gridLayoutWidget)
        self.cbBox_third_jinzhi.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_third_jinzhi.setObjectName("cbBox_third_jinzhi")
        self.gridLayout.addWidget(self.cbBox_third_jinzhi, 13, 2, 1, 1)
        self.cbBox_third_zhongzhi = QComboBox(self.gridLayoutWidget)
        self.cbBox_third_zhongzhi.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_third_zhongzhi.setObjectName("cbBox_third_zhongzhi")
        self.gridLayout.addWidget(self.cbBox_third_zhongzhi, 15, 2, 1, 1)
        self.cbBox_fifth_jinzhi = QComboBox(self.gridLayoutWidget)
        self.cbBox_fifth_jinzhi.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_fifth_jinzhi.setObjectName("cbBox_fifth_jinzhi")
        self.gridLayout.addWidget(self.cbBox_fifth_jinzhi, 14, 2, 1, 1)
        self.cbBox_third_yaunzhi = QComboBox(self.gridLayoutWidget)
        self.cbBox_third_yaunzhi.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_third_yaunzhi.setObjectName("cbBox_third_yaunzhi")
        self.gridLayout.addWidget(self.cbBox_third_yaunzhi, 18, 2, 1, 1)
        self.label_third_zhongzhi = QLabel(self.gridLayoutWidget)
        self.label_third_zhongzhi.setObjectName("label_third_zhongzhi")
        self.gridLayout.addWidget(self.label_third_zhongzhi, 15, 1, 1, 1)
        self.cbBox_fifth_zhang = QComboBox(self.gridLayoutWidget)
        self.cbBox_fifth_zhang.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_fifth_zhang.setObjectName("cbBox_fifth_zhang")
        self.gridLayout.addWidget(self.cbBox_fifth_zhang, 11, 2, 1, 1)
        self.cbBox_fifth_zhongzhi = QComboBox(self.gridLayoutWidget)
        self.cbBox_fifth_zhongzhi.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_fifth_zhongzhi.setObjectName("cbBox_fifth_zhongzhi")
        self.gridLayout.addWidget(self.cbBox_fifth_zhongzhi, 16, 2, 1, 1)
        self.cbBox_third_zhang = QComboBox(self.gridLayoutWidget)
        self.cbBox_third_zhang.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_third_zhang.setObjectName("cbBox_third_zhang")
        self.gridLayout.addWidget(self.cbBox_third_zhang, 10, 2, 1, 1)
        self.label_fifth_yuanzhi = QLabel(self.gridLayoutWidget)
        self.label_fifth_yuanzhi.setObjectName("label_fifth_yuanzhi")
        self.gridLayout.addWidget(self.label_fifth_yuanzhi, 20, 1, 1, 1)
        self.label_ulna = QLabel(self.gridLayoutWidget)
        self.label_ulna.setObjectName("label_ulna")
        self.gridLayout.addWidget(self.label_ulna, 2, 1, 1, 1)
        self.label3 = QLabel(self.gridLayoutWidget)
        self.label3.setObjectName("label3")
        self.gridLayout.addWidget(self.label3, 8, 3, 1, 1)
        self.label9 = QLabel(self.gridLayoutWidget)
        self.label9.setObjectName("label9")
        self.gridLayout.addWidget(self.label9, 15, 3, 1, 1)
        self.label7 = QLabel(self.gridLayoutWidget)
        self.label7.setObjectName("label7")
        self.gridLayout.addWidget(self.label7, 13, 3, 1, 1)
        self.label12 = QLabel(self.gridLayoutWidget)
        self.label12.setObjectName("label12")
        self.gridLayout.addWidget(self.label12, 18, 3, 1, 1)
        self.label_first_jinzhi = QLabel(self.gridLayoutWidget)
        self.label_first_jinzhi.setObjectName("label_first_jinzhi")
        self.gridLayout.addWidget(self.label_first_jinzhi, 12, 1, 1, 1)
        self.label11 = QLabel(self.gridLayoutWidget)
        self.label11.setObjectName("label11")
        self.gridLayout.addWidget(self.label11, 17, 3, 1, 1)
        self.cbBox_first_zhang = QComboBox(self.gridLayoutWidget)
        self.cbBox_first_zhang.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_first_zhang.setObjectName("cbBox_first_zhang")
        self.gridLayout.addWidget(self.cbBox_first_zhang, 8, 2, 1, 1)
        self.label8 = QLabel(self.gridLayoutWidget)
        self.label8.setObjectName("label8")
        self.gridLayout.addWidget(self.label8, 14, 3, 1, 1)
        self.label4 = QLabel(self.gridLayoutWidget)
        self.label4.setObjectName("label4")
        self.gridLayout.addWidget(self.label4, 10, 3, 1, 1)
        self.label1 = QLabel(self.gridLayoutWidget)
        self.label1.setObjectName("label1")
        self.gridLayout.addWidget(self.label1, 2, 3, 1, 1)
        self.label_first_zhang = QLabel(self.gridLayoutWidget)
        self.label_first_zhang.setObjectName("label_first_zhang")
        self.gridLayout.addWidget(self.label_first_zhang, 8, 1, 1, 1)
        self.label_5 = QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 11, 1, 1, 1)
        self.label6 = QLabel(self.gridLayoutWidget)
        self.label6.setObjectName("label6")
        self.gridLayout.addWidget(self.label6, 12, 3, 1, 1)
        self.label_fifth_jinzhi = QLabel(self.gridLayoutWidget)
        self.label_fifth_jinzhi.setObjectName("label_fifth_jinzhi")
        self.gridLayout.addWidget(self.label_fifth_jinzhi, 14, 1, 1, 1)
        self.label_fifth_zhongzhi = QLabel(self.gridLayoutWidget)
        self.label_fifth_zhongzhi.setObjectName("label_fifth_zhongzhi")
        self.gridLayout.addWidget(self.label_fifth_zhongzhi, 16, 1, 1, 1)
        self.label_third_jinzhi = QLabel(self.gridLayoutWidget)
        self.label_third_jinzhi.setObjectName("label_third_jinzhi")
        self.gridLayout.addWidget(self.label_third_jinzhi, 13, 1, 1, 1)
        self.label_third_yuanzhi = QLabel(self.gridLayoutWidget)
        self.label_third_yuanzhi.setObjectName("label_third_yuanzhi")
        self.gridLayout.addWidget(self.label_third_yuanzhi, 18, 1, 1, 1)
        self.label5 = QLabel(self.gridLayoutWidget)
        self.label5.setObjectName("label5")
        self.gridLayout.addWidget(self.label5, 11, 3, 1, 1)
        self.label10 = QLabel(self.gridLayoutWidget)
        self.label10.setObjectName("label10")
        self.gridLayout.addWidget(self.label10, 16, 3, 1, 1)
        self.label_third_zhang = QLabel(self.gridLayoutWidget)
        self.label_third_zhang.setObjectName("label_third_zhang")
        self.gridLayout.addWidget(self.label_third_zhang, 10, 1, 1, 1)
        self.label2 = QLabel(self.gridLayoutWidget)
        self.label2.setObjectName("label2")
        self.gridLayout.addWidget(self.label2, 5, 3, 1, 1)
        self.cbBox_first_jinzhi = QComboBox(self.gridLayoutWidget)
        self.cbBox_first_jinzhi.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        self.cbBox_first_jinzhi.setObjectName("cbBox_first_jinzhi")
        self.gridLayout.addWidget(self.cbBox_first_jinzhi, 12, 2, 1, 1)
        self.label_epiphysis = QLabel(self.gridLayoutWidget)
        self.label_epiphysis.setObjectName("label_epiphysis")
        self.gridLayout.addWidget(self.label_epiphysis, 0, 1, 1, 1)
        self.gridLayoutWidget_2 = QWidget(self)
        self.gridLayoutWidget_2.setGeometry(QRect(40, 30, 911, 651))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.right_img = QLabel(self.gridLayoutWidget_2)
        self.right_img.setObjectName("right_img")
        self.gridLayout_2.addWidget(self.right_img, 0, 2, 1, 1)
        self.det_img_button = QPushButton(self.gridLayoutWidget_2)
        self.det_img_button.setObjectName("det_img_button")
        self.gridLayout_2.addWidget(self.det_img_button, 2, 2, 1, 1)
        self.left_img = QLabel(self.gridLayoutWidget_2)
        self.left_img.setObjectName("left_img")
        self.gridLayout_2.addWidget(self.left_img, 0, 1, 1, 1)
        self.up_img_button = QPushButton(self.gridLayoutWidget_2)
        self.up_img_button.setObjectName("up_img_button")
        self.gridLayout_2.addWidget(self.up_img_button, 2, 1, 1, 1)
        self.gridLayoutWidget_3 = QWidget(self)
        self.gridLayoutWidget_3.setGeometry(QRect(970, 90, 371, 101))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_father_height = QLabel(self.gridLayoutWidget_3)
        self.label_father_height.setObjectName("label_father_height")
        self.gridLayout_3.addWidget(self.label_father_height, 3, 0, 1, 1)
        self.text_father_height = QLineEdit(self.gridLayoutWidget_3)
        self.text_father_height.setObjectName("text_father_height")
        self.gridLayout_3.addWidget(self.text_father_height, 3, 2, 1, 1)
        self.label_mother_height = QLabel(self.gridLayoutWidget_3)
        self.label_mother_height.setObjectName("label_mother_height")
        self.gridLayout_3.addWidget(self.label_mother_height, 3, 3, 1, 1)
        self.label_name = QLabel(self.gridLayoutWidget_3)
        self.label_name.setObjectName("label_name")
        self.gridLayout_3.addWidget(self.label_name, 1, 0, 1, 1)
        self.label_gender = QLabel(self.gridLayoutWidget_3)
        self.label_gender.setObjectName("label_gender")
        self.gridLayout_3.addWidget(self.label_gender, 1, 3, 1, 1)
        self.text_name = QLineEdit(self.gridLayoutWidget_3)
        self.text_name.setEnabled(True)
        self.text_name.setObjectName("text_name")
        self.gridLayout_3.addWidget(self.text_name, 1, 2, 1, 1)
        self.comboBox_gender = QComboBox(self.gridLayoutWidget_3)
        self.comboBox_gender.addItems(['男', '女'])
        self.comboBox_gender.setObjectName("comboBox_gender")
        self.gridLayout_3.addWidget(self.comboBox_gender, 1, 4, 1, 1)
        self.dateEdit = QDateEdit(self.gridLayoutWidget_3)
        self.dateEdit.setObjectName("dateEdit")
        self.gridLayout_3.addWidget(self.dateEdit, 2, 4, 1, 1)
        self.text_age = QLineEdit(self.gridLayoutWidget_3)
        self.text_age.setObjectName("text_age")
        self.gridLayout_3.addWidget(self.text_age, 2, 2, 1, 1)
        self.label_date = QLabel(self.gridLayoutWidget_3)
        self.label_date.setObjectName("label_date")
        self.gridLayout_3.addWidget(self.label_date, 2, 3, 1, 1)
        self.label_age = QLabel(self.gridLayoutWidget_3)
        self.label_age.setObjectName("label_age")
        self.gridLayout_3.addWidget(self.label_age, 2, 0, 1, 1)
        self.text_mother_height = QLineEdit(self.gridLayoutWidget_3)
        self.text_mother_height.setObjectName("text_mother_height")
        self.gridLayout_3.addWidget(self.text_mother_height, 3, 4, 1, 1)
        self.text_predict_height = QLineEdit(self.gridLayoutWidget_3)
        self.text_predict_height.setObjectName("text_predict_height")
        self.gridLayout_3.addWidget(self.text_predict_height, 4, 2, 1, 1)
        self.button_predict_height = QPushButton(self.gridLayoutWidget_3)
        self.button_predict_height.setObjectName("button_predict_height")
        self.gridLayout_3.addWidget(self.button_predict_height, 4, 0, 1, 1)
        font_main = QFont('楷体', 14)
        _translate = QCoreApplication.translate
        self.button_bone_age_predict.setFont(font_main)
        self.button_bone_age_predict.setText(_translate("Form", "骨龄评估"))
        self.button_bone_age_predict.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(2,110,180);}"
                                          "QPushButton{background-color:rgb(48,124,208)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:5px 5px}")
        # self.label13.setText(_translate("Form", "TextLabel"))
        self.label_level.setText(_translate("Form", "等级"))
        self.label_radius.setText(_translate("Form", "02桡骨"))
        # self.label_atlas.setText(_translate("Form", "图谱"))
        self.label_first_yuanzhi.setText(_translate("Form", "07第一远节指骨"))
        self.label_third_zhongzhi.setText(_translate("Form", "09第三中节指骨"))
        self.label_fifth_yuanzhi.setText(_translate("Form", "13第五远节指骨"))
        self.label_ulna.setText(_translate("Form", "01尺骨"))
        # self.label3.setText(_translate("Form", "TextLabel"))
        # self.label9.setText(_translate("Form", "TextLabel"))
        # self.label7.setText(_translate("Form", "TextLabel"))
        # self.label12.setText(_translate("Form", "TextLabel"))
        self.label_first_jinzhi.setText(_translate("Form", "06第一近节指骨"))
        # self.label11.setText(_translate("Form", "TextLabel"))
        # self.label8.setText(_translate("Form", "TextLabel"))
        # self.label4.setText(_translate("Form", "TextLabel"))
        # self.label1.setText(_translate("Form", "TextLabel"))
        self.label_first_zhang.setText(_translate("Form", "03第一掌骨"))
        self.label_5.setText(_translate("Form", "05第五掌骨"))
        # self.label6.setText(_translate("Form", "TextLabel"))
        self.label_fifth_jinzhi.setText(_translate("Form", "11第五近节指骨"))
        self.label_fifth_zhongzhi.setText(_translate("Form", "12第五中节指骨"))
        self.label_third_jinzhi.setText(_translate("Form", "08第三近节指骨"))
        self.label_third_yuanzhi.setText(_translate("Form", "10第三远节指骨"))
        # self.label5.setText(_translate("Form", "TextLabel"))
        # self.label10.setText(_translate("Form", "TextLabel"))
        self.label_third_zhang.setText(_translate("Form", "04第三掌骨"))
        # self.label2.setText(_translate("Form", "TextLabel"))
        self.label_epiphysis.setText(_translate("Form", "骨骺"))
        self.right_img.setText(_translate("Form", "右边图片"))
        self.det_img_button.setText(_translate("Form", "开始检测"))
        self.left_img.setText(_translate("Form", "左边图片"))
        self.left_img.setPixmap(QPixmap("images/UI/img1.jpg"))
        self.right_img.setPixmap(QPixmap("images/UI/img2.jpg"))
        self.up_img_button.setText(_translate("Form", "上传图片"))
        self.up_img_button.setFont(font_main)
        self.det_img_button.setFont(font_main)
        self.up_img_button.clicked.connect(self.upload_img)
        self.det_img_button.clicked.connect(self.detect_img)
        self.up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        self.det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        self.label_father_height.setText(_translate("Form", "父亲身高："))
        self.label_mother_height.setText(_translate("Form", "母亲身高："))
        self.label_name.setText(_translate("Form", "姓名："))
        self.label_gender.setText(_translate("Form", "性别："))
        self.label_date.setText(_translate("Form", "时间"))
        self.label_age.setText(_translate("Form", "年龄："))
        self.button_predict_height.setText(_translate("Form", "遗传身高预测"))
        font_predict_height = QFont('楷体', 12)
        self.button_predict_height.setFont(font_predict_height)
        self.button_predict_height.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(2,110,180);}"
                                          "QPushButton{background-color:rgb(48,124,208)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                                 )

        # 遗传身高
        self.button_predict_height.clicked.connect(self.calcGeneHeight)

        # 预测骨龄
        self.button_bone_age_predict.clicked.connect(self.calc_boneAge)

        self.initUI()

        self.LabelIndexName = {
            '01': '01',  # 尺骨
            '02': '02',  # 桡骨
            '61': '03',  # 第一掌骨
            '62': '04',  # 第三掌骨
            '63': '05',  # 第五掌骨
            '11': '06',  # 第一近指骨
            '12': '07',  # 第一远指骨
            '31': '08',  # 第三近指骨
            '32': '09',  # 第三中指骨
            '33': '10',  # 第三远指骨
            '51': '11',  # 第五近指骨
            '52': '12',  # 第五中指骨
            '53': '13'  # 第五远指骨
            }

    '''
    ***模型初始化***
    '''
    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = torch.device('cpu')
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = torch.device('cpu')
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model

    '''
    ***界面初始化***
    '''
    def initUI(self):
        # 图片检测子界面
        # font_title = QFont('楷体', 16)
        # font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        mid_img_layout.addStretch(0)
        mid_img_widget.setLayout(mid_img_layout)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)

        img_detection_widget.setLayout(img_detection_layout)

        self.addTab(img_detection_widget, '图片检测')
        self.setTabIcon(0, QIcon('images/UI/tubiao.jpg'))

    #遗传身高计算
    def calcGeneHeight(self):

        faHeight = float(self.text_father_height.text())
        maHeight = float(self.text_mother_height.text())
        sex = self.comboBox_gender.currentText()

        if sex == '男':
            result = (faHeight + maHeight + 13) / 2
        else:
            result = (faHeight + maHeight - 13) / 2

        self.text_predict_height.setText(str(result))


    #计算骨龄发育总分，调用骨龄计算函数，返回具体骨龄值
    def calc_boneAge(self):

        # 骨发育分评分表
        gjScore = {'女': {
            '01': [0, 30, 33, 37, 45, 74, 118, 173],
            '02': [0, 23, 30, 44, 56, 78, 114, 160, 218],
            '61': [0, 8, 12, 18, 24, 31, 43, 53, 67],
            '62': [0, 5, 8, 12, 16, 23, 37, 47, 53],
            '63': [0, 6, 9, 12, 17, 23, 35, 48, 52],
            '11': [0, 9, 11, 14, 20, 31, 44, 56, 67],
            '12': [0, 7, 9, 15, 22, 33, 48, 51, 68],
            '31': [0, 5, 7, 12, 19, 27, 37, 44, 54],
            '32': [0, 6, 8, 12, 18, 27, 36, 45, 52],
            '33': [0, 7, 8, 11, 15, 22, 33, 37, 49],
            '51': [0, 6, 7, 12, 18, 26, 35, 42, 51],
            '52': [0, 7, 8, 12, 18, 28, 35, 43, 49],
            '53': [0, 7, 8, 11, 15, 22, 32, 36, 47]
        },
            '男': {
                '01': [0, 27, 30, 32, 40, 58, 107, 181],  # 尺骨
                '02': [0, 16, 21, 30, 39, 59, 87, 138, 213],  # 桡骨
                '61': [0, 6, 9, 14, 21, 26, 36, 49, 67],  # 第一掌骨
                '62': [0, 4, 5, 9, 12, 19, 31, 43, 52],  # 第三掌骨
                '63': [0, 4, 6, 9, 14, 18, 29, 43, 52],  # 第五掌骨
                '11': [0, 7, 8, 11, 17, 26, 38, 52, 67],  # 第一近指骨
                '12': [0, 5, 6, 11, 17, 26, 38, 46, 66],  # 第一远指骨
                '31': [0, 4, 4, 9, 15, 23, 31, 40, 53],  # 第三近指骨
                '32': [0, 4, 6, 9, 15, 22, 32, 43, 52],  # 第三中指骨
                '33': [0, 4, 6, 8, 13, 18, 28, 34, 49],  # 第三远指骨
                '51': [0, 4, 5, 9, 15, 21, 30, 39, 51],  # 第五近指骨
                '52': [0, 6, 7, 9, 15, 23, 32, 42, 49],  # 第五中指骨
                '53': [0, 5, 6, 9, 13, 18, 27, 34, 48]  # 第五远指骨
            }
        }

        sex = self.comboBox_gender.currentText()
        #获取每个骨关节的骨发育等级
        cg = ord(self.cbBox_ulna.currentText()) - 65
        rg = ord(self.cbBox_radius.currentText()) - 65
        dyz = ord(self.cbBox_first_zhang.currentText()) - 65
        dszh = ord(self.cbBox_third_zhang.currentText()) - 65
        dwzh = ord(self.cbBox_fifth_zhang.currentText()) - 65
        dyj = ord(self.cbBox_first_jinzhi.currentText()) - 65
        dsj = ord(self.cbBox_third_jinzhi.currentText()) - 65
        dwj = ord(self.cbBox_fifth_jinzhi.currentText()) - 65
        dsz = ord(self.cbBox_third_zhongzhi.currentText()) - 65
        dwz = ord(self.cbBox_fifth_zhongzhi.currentText()) - 65
        dyy = ord(self.cbBox_first_yaunzhi.currentText()) - 65
        dsy = ord(self.cbBox_third_yaunzhi.currentText()) - 65
        dwy = ord(self.cbBox_fifth_yaunzhi.currentText()) - 65

        # 将骨发育等级转成数值
        value = [cg, rg, dyz, dszh, dwzh, dyj, dsj, dwj, dsz, dwz, dyy, dsy, dwy]
        finally_results = {'01': value[0], '02': value[1], '61': value[2], '62': value[3],
                           '63': value[4], '11': value[5], '31': value[6], '51': value[7],
                           '32': value[8], '52': value[9], '12': value[10], '33': value[11],
                           '53': value[12]}
        score = 0
        for key, value in finally_results.items():
            # 根据每个关节的等级，计算骨发育得分
            score += gjScore[sex][key][value]


        # 计算骨龄
        list1 = ['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5',
                 '7.0', '7.5', '8.0', '8.5', '9,0', '9.5', '10.0', '10.5', '11.0', '11.5', '12.0', '12.5',
                 '13.0', '13.5', '14.0', '14.5', '15.0', '15.5', '16.0', '16.5', '17.0', '17.5', '18.0']
        # 男生
        list2 = ['74', '80', '85', '91', '98', '105', '112', '121', '130', '139', '150', '161', '173',
                 '186', '201', '216', '232', '250', '270', '291', '313', '338', '369', '431', '492',
                 '550', '605', '659', '710', '760', '807', '853', '897', '940', '980']
        list3 = ['139', '146', '152', '160', '168', '177', '188', '199', '211', '225', '240', '257',
                 '275', '296', '318', '343', '370', '401', '434', '483', '560', '627', '686', '738', '784',
                 '824', '859', '890', '917', '940', '961', '979', '995']

        if sex == '男':
            if score <= 980:
                res = 0
                for i, item in enumerate(list2):
                    int(item)
                    if int(item) >= score:
                        res = i
                        break
                # print(list2[res])
                x = score - int(list2[res - 1])
                y = (int(list2[res]) - int(list2[res - 1]))//6
                z = x // y
                year = ((res-1)//2) + 1
                if (res-1) % 2 == 1:
                    month = 6 + z
                else:
                    month = z
                value = "{0}岁{1}个月".format(year, month)
                self.text_bone_age_predict.setText(value)
            else:
                self.text_bone_age_predict.setText("18岁")
        else:
            if score <= 995:
                res = 0
                for i, item in enumerate(list3):
                    int(item)
                    if int(item) >= score:
                        res = i
                        break
                # print(list3[res])
                x = score - int(list3[res - 1])
                y = (int(list3[res]) - int(list3[res - 1]))//6
                z = x // y
                year = ((res-1)//2) + 1
                if (res-1) % 2 == 1:
                    month = 6 + z
                else:
                    month = z
                value = "{0}岁{1}个月".format(year, month)
                self.text_bone_age_predict.setText(value)
            else:
                self.text_bone_age_predict.setText("18岁")



    '''
    ***上传图片***
    '''
    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/img2.jpg"))
            self.cbBox_ulna.setCurrentText('A')
            self.cbBox_radius.setCurrentText('A')
            self.cbBox_first_zhang.setCurrentText("A")
            self.cbBox_third_zhang.setCurrentText("A")
            self.cbBox_fifth_zhang.setCurrentText("A")
            self.cbBox_first_jinzhi.setCurrentText("A")
            self.cbBox_first_yaunzhi.setCurrentText("A")
            self.cbBox_third_jinzhi.setCurrentText("A")
            self.cbBox_third_zhongzhi.setCurrentText("A")
            self.cbBox_third_yaunzhi.setCurrentText("A")
            self.cbBox_fifth_jinzhi.setCurrentText("A")
            self.cbBox_fifth_zhongzhi.setCurrentText("A")
            self.cbBox_fifth_yaunzhi.setCurrentText("A")
            self.text_bone_age_predict.setText("")



    def showCombBoxResult(self, source):
        for key, value in source.items() :
            if key == '01' :
                self.cbBox_ulna.setCurrentText(value)
            elif key == '02' :
                self.cbBox_radius.setCurrentText(value)
            elif key == '03':
                self.cbBox_first_zhang.setCurrentText(value)
            elif key == '04' :
                self.cbBox_third_zhang.setCurrentText(value)
            elif key == '05' :
                self.cbBox_fifth_zhang.setCurrentText(value)
            elif key == '06' :
                self.cbBox_first_jinzhi.setCurrentText(value)
            elif key == '07' :
                self.cbBox_first_yaunzhi.setCurrentText(value)
            elif key == '08' :
                self.cbBox_third_jinzhi.setCurrentText(value)
            elif key == '09' :
                self.cbBox_third_zhongzhi.setCurrentText(value)
            elif key == '10' :
                self.cbBox_third_yaunzhi.setCurrentText(value)
            elif key == '11' :
                self.cbBox_fifth_jinzhi.setCurrentText(value)
            elif key == '12' :
                self.cbBox_fifth_zhongzhi.setCurrentText(value)
            else :
                self.cbBox_fifth_yaunzhi.setCurrentText(value)


    '''
    ***获取排序标签***
    '''
    def getIndexAndName(self, source):
        if source in self.LabelIndexName.keys():
            return self.LabelIndexName.get(source)


    '''
    ***检测图片***
    '''
    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [640,640]  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            device = torch.device('cpu')
            webcam = False
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            # Dataloader
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs
            # Run inference
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1
                # Inference
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3
                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        results = {} # 图片检测结果
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                # with open(txt_path + '.txt', 'a') as f:
                                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                label = label.split(' ')[0]
                                index = self.getIndexAndName(label[:2])
                                value = label[2:]
                                label = index + value
                                results[index] = value
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                # if save_crop:
                                #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                #                  BGR=True)

                        #文本显示检测结果
                        # print(results)
                        self.showCombBoxResult(results)

                    # Print time (inference-only)
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                    # Stream results
                    im0 = annotator.result()
                    # if view_img:
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 millisecond
                    # Save results (image with detections)
                    resize_scale = output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result.jpg", im0)
                    # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
                    self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

    '''
    ### 界面关闭事件 ### 
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    # ui = Ui_Form()  # ui是Ui_MainWindow()类的实例化对象
    # ui.setupUi(mainWindow)  # 执行类中的setupUi方法，方法的参数是第二步中创建的QMainWindow
    mainWindow.show()
    sys.exit(app.exec_())
