from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import pprint
from IBnet import SaveActivations

class SignalEmit(QWidget):
    StartSignal = pyqtSignal(dict)
    previewSignal = pyqtSignal(dict)
    def __init__(self):
        super().__init__()        
        self.initUI()

    def initUI(self):           

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

        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('DeepInfoFlow Console')
        self.show()

    def configControls(self):
        self.startButton = QPushButton("start train")
        self.previewButton  = QPushButton("preview")

        self.learningRateLabel = QLabel("learning rate: ")
        self.learningRateText = QLineEdit(self)
        self.learningRateText.setText("0.0001")

        self.maxEpochLabel = QLabel("max epoch: ")
        self.maxEpochText = QLineEdit(self)
        self.maxEpochText.setText("2")

        self.datasetLabel = QLabel("dataset: ")
        self.datasetCombo = QComboBox(self)
        self.datasetCombo.addItem("IBNet")
        self.datasetCombo.addItem("MNIST")

        self.isLogMean = QCheckBox("mean")
        self.isLogStd = QCheckBox("std")
        self.isLogL2n = QCheckBox("l2n")
        self.isLogMean.setCheckState(2)
        self.isLogStd.setCheckState(2)
        self.isLogL2n.setCheckState(2)

    def layoutControls(self):
        controlsLayout = QGridLayout()
        controlsLayout.addWidget(self.learningRateLabel, 0, 0)
        controlsLayout.addWidget(self.learningRateText, 0, 1)
        controlsLayout.addWidget(self.maxEpochLabel, 1, 0)
        controlsLayout.addWidget(self.maxEpochText, 1, 1)

        controlsLayout.addWidget(self.datasetLabel, 0, 2)
        controlsLayout.addWidget(self.datasetCombo, 0, 3)

        controlsLayout.addWidget(self.previewButton, 0, 4)
        controlsLayout.addWidget(self.startButton, 1, 4)
        
        controlsLayout.addWidget(self.isLogMean, 2, 0)
        controlsLayout.addWidget(self.isLogStd, 2, 1)
        controlsLayout.addWidget(self.isLogL2n, 2, 2)
        return controlsLayout

    def creatControls(self, title):
        self.controlsGroup = QGroupBox(title)
        self.configControls()
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
        mean        = True if self.isLogMean.isChecked() else False
        std         = True if self.isLogStd.isChecked() else False
        l2n         = True if self.isLogL2n.isChecked() else False
        keys = ["lr", "max_epoch", "dataset", "mean", "std", "l2n"]
        vals = (lr, maxEpoch, dataset, mean, std, l2n)
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
        
if __name__ == '__main__':

    app = QApplication(sys.argv)
    dispatch = SignalEmit()
    sys.exit(app.exec_())

