import os
import sys
from numpy import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from plot_utils import *

class QtPlot(QWidget):

    def __init__(self, opt):
        super().__init__()
        self.title = opt.type
        self.left = [100, 1000]
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left[0] if opt.type == 'Mean_and_STD' else  self.left[1], \
                        self.top, self.width, self.height)
    
        label = QLabel(self)
        img = self.load_img()
        label.setPixmap(img)
        self.resize(img.width(),img.height())
        
        self.show()

    def load_img(self):
        img_path = os.path.join(opt.plot_dir, opt.model_name, opt.timestamp, opt.type, "test.jpg")
        img = QPixmap(img_path)
        return img


if __name__ == '__main__':

    C = type('type_C', (object,), {})
    opt = C()

    opt.plot_dir = './plots'
    opt.model_name = 'testdrawing'
    opt.timestamp = '19050310'
    opt.type = 'InfoPlan'

    app = QApplication(sys.argv)
    ex = QtPlot(opt)
    sys.exit(app.exec_())
    
    