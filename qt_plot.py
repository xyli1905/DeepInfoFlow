import os
import sys
from numpy import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2 as cv


import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class ShowImgHelper(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        hlay = QHBoxLayout(self)
        self.treeview = QTreeView()
        self.listview = QListView()
        hlay.addWidget(self.treeview)
        hlay.addWidget(self.listview)

        path = ".\\results"
        
        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(QDir.rootPath())
        self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

        self.fileModel = QFileSystemModel()
        self.fileModel.setFilter(QDir.NoDotAndDotDot |  QDir.Files)

        self.treeview.setModel(self.dirModel)
        self.listview.setModel(self.fileModel)

        self.treeview.setRootIndex(self.dirModel.index(path))
        self.listview.setRootIndex(self.fileModel.index(path))

        self.treeview.clicked.connect(self.onClicked)

    def onMainWindowClicked(self):
        if not self.isVisible():
            self.show()

    def onShowPlanSignal(self):
        name = 'InfoPlan'
        img_path = self.path + "//InfoPlan.jpg"
        if os.path.exists(img_path):
            self.showImg(name, img_path)

    def onShowMean_and_STDSignal(self):
        name = 'Mean_and_STD'
        img_path = self.path + "//Mean_and_STD.jpg"
        if os.path.exists(img_path):
            self.showImg(name, img_path)

    def onClicked(self, index):
        self.path = self.dirModel.fileInfo(index).absoluteFilePath()
        self.listview.setRootIndex(self.fileModel.setRootPath(self.path))

    def showImg(self, name, img_path):
        if os.path.exists(img_path):
            src = cv.imread(img_path)   
            cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
            cv.imshow(name, src)
            cv.waitKey(0)
        else:
            print("Can't find img!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = ShowImgHelper()
    w.show()
    sys.exit(app.exec_())
