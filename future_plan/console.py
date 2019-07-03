from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os
import sys
import pprint
from IBnet import SaveActivations
from qt_plot import ShowImgHelper, ShowJsonHelper, ProgressBarHelper

import cv2 as cv

class Console(QWidget):
    StartSignal = pyqtSignal(dict)
    previewSignal = pyqtSignal(dict)
    def __init__(self):
        super().__init__()        
        self.initUI()

    def initUI(self):           
        self.img_folder   = ShowImgHelper()
        self.json_folder  = ShowJsonHelper()
        self.progress_bar = ProgressBarHelper(self.json_folder)

        self.creatControls("parameter setting")
        self.creatResult("parameter preview")

        layout = QHBoxLayout()
        layout.addWidget(self.controlsGroup)
        layout.addWidget(self.resultGroup)
        self.setLayout(layout)

        self.StartSignal.connect(self.onStartButtonClick)
        self.previewSignal.connect(self.onPreviewButtonClick)

        self.startButton.clicked.connect(self.emitStartSignal)
        self.previewButton.clicked.connect(self.emitPreviewSignal)

        self.InfoPlanButton.clicked.connect(self.img_folder.onShowPlanSignal)
        self.Mean_and_STDButton.clicked.connect(self.img_folder.onShowMean_and_STDSignal)
        self.imageButton.clicked.connect(self.img_folder.onMainWindowClicked)

        self.computeMIButton.clicked.connect(self.progress_bar.onMainWindowClicked)
        self.computeMIButton.clicked.connect(self.json_folder.onComputeMISignal)
        self.jsonButton.clicked.connect(self.json_folder.onMainWindowClicked)


        self.setGeometry(300, 300, 360, 250)
        self.setWindowTitle('DeepInfoFlow Console')
        self.show()

    def configButtons(self):
        self.startButton = QPushButton("start train")
        self.previewButton  = QPushButton("preview")
        self.computeMIButton  = QPushButton("compute MI")
        self.InfoPlanButton = QPushButton("InfoPlan")
        self.Mean_and_STDButton  = QPushButton("Mean_and_STD")
        self.imageButton = QPushButton("image folders")
        self.jsonButton = QPushButton("json folders")

    def configLineEdit(self):
        self.learningRateLabel = QLabel("learning rate: ")
        self.learningRateText = QLineEdit(self)
        self.learningRateText.setText("0.0001")

        self.maxEpochLabel = QLabel("max epoch: ")
        self.maxEpochText = QLineEdit(self)
        self.maxEpochText.setText("2")

    def configCombo(self):
        self.datasetLabel = QLabel("dataset: ")
        self.datasetCombo = QComboBox(self)
        self.datasetCombo.addItem("IBNet")
        self.datasetCombo.addItem("MNIST")

        self.activationLabel = QLabel("activation: ")
        self.activationCombo = QComboBox(self)
        self.activationCombo.addItem("tanh")
        self.activationCombo.addItem("relu")

    def configCheckBox(self):
        self.isLogMean = QCheckBox("mean")
        self.isLogStd = QCheckBox("std")
        self.isLogL2n = QCheckBox("l2n")
        self.isLogMean.setCheckState(2)
        self.isLogStd.setCheckState(2)
        self.isLogL2n.setCheckState(2)

    def configControlers(self):
        self.configButtons()
        self.configLineEdit()
        self.configCombo()
        self.configCheckBox()


    def layoutControls(self):
        controlsLayout = QGridLayout()
        controlsLayout.addWidget(self.learningRateLabel, 0, 0)
        controlsLayout.addWidget(self.learningRateText, 0, 1)

        controlsLayout.addWidget(self.maxEpochLabel, 1, 0)
        controlsLayout.addWidget(self.maxEpochText, 1, 1)

        controlsLayout.addWidget(self.datasetLabel, 0, 2)
        controlsLayout.addWidget(self.datasetCombo, 0, 3)

        controlsLayout.addWidget(self.activationLabel, 1, 2)
        controlsLayout.addWidget(self.activationCombo, 1, 3)

        controlsLayout.addWidget(self.previewButton, 0, 4)
        controlsLayout.addWidget(self.startButton, 1, 4)
        
        controlsLayout.addWidget(self.isLogMean, 2, 0)
        controlsLayout.addWidget(self.isLogStd, 2, 1)
        controlsLayout.addWidget(self.isLogL2n, 2, 2)

        controlsLayout.addWidget(self.jsonButton, 3, 0)
        controlsLayout.addWidget(self.computeMIButton, 3, 1)

        controlsLayout.addWidget(self.imageButton, 4, 0)
        controlsLayout.addWidget(self.InfoPlanButton, 4, 1)
        controlsLayout.addWidget(self.Mean_and_STDButton, 4, 2)
        
        return controlsLayout

    def creatControls(self, title):
        self.controlsGroup = QGroupBox(title)
        self.configControlers()
        controlsLayout = self.layoutControls()
        self.controlsGroup.setLayout(controlsLayout)

    def creatResult(self, title):
        self.resultGroup = QGroupBox(title)
        self.resultLabel = QLabel("")
        layout = QHBoxLayout()
        layout.addWidget(self.resultLabel)
        self.resultGroup.setLayout(layout)

    def getConfigFromConsole(self):
        lr          = float(self.learningRateText.text())
        maxEpoch    = int(self.maxEpochText.text())
        dataset     = self.datasetCombo.currentText()
        activation  = self.activationCombo.currentText()
        mean        = True if self.isLogMean.isChecked() else False
        std         = True if self.isLogStd.isChecked() else False
        l2n         = True if self.isLogL2n.isChecked() else False
        keys = ["lr", "max_epoch", "dataset", "activation", "mean", "std", "l2n"]
        vals = (lr, maxEpoch, dataset, activation, mean, std, l2n)
        config = dict(zip(keys, vals))
        return config

    def emitPreviewSignal(self):
        config = self.getConfigFromConsole()
        self.previewSignal.emit(config)

    def emitStartSignal(self):
        config = self.getConfigFromConsole()
        self.StartSignal.emit(config)

    def onPreviewButtonClick(self, config):
        pretty_dict_str = pprint.pformat(config)
        self.resultLabel.setText(pretty_dict_str)    

    def onStartButtonClick(self, config):
        test = SaveActivations()
        test._update_opt(config)
        test.training_model()     

    def showImg(self):
        img_path = os.path.join(self.opt.plot_path, self.opt.model_name, self.opt.timestamp, self.opt.type, "test.jpg")
        if os.path.exists(img_path):
            src = cv.imread(img_path)   
            cv.namedWindow(self.opt.type, cv.WINDOW_AUTOSIZE)
            cv.imshow(self.opt.type, src)
            cv.waitKey(0)
        else:
            print("Can't find img!")
    

if __name__ == '__main__':

    app = QApplication(sys.argv)
    dispatch = Console()
    sys.exit(app.exec_())

