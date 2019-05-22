import os
import sys
from numpy import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2 as cv
from ComputeMI import *
import time
import threading

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
        self.treeview.setFixedSize(720, 480)
        self.treeview.setColumnWidth(0, 500)

    def onMainWindowClicked(self):
        if not self.isVisible():
            self.show()
            self.setWindowTitle("Img folders")

    def onShowPlanSignal(self):
        name = 'InfoPlan'
        try:
            img_path = self.path + "//InfoPlan.jpg"
            if os.path.exists(img_path):
                self.showImg(name, img_path)
        except Exception as e:
            print ("You should firstly specify a folder")
        

    def onShowMean_and_STDSignal(self):
        name = 'Mean_and_STD'
        try:
            img_path = self.path + "//Mean_and_STD.jpg"
            if os.path.exists(img_path):
                self.showImg(name, img_path)
        except Exception as e:
            print ("You should firstly specify a folder")
        

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


class ShowJsonHelper(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.c = ComputeMI()
        path = ".\\results"
        hlay = QHBoxLayout(self)
        self.treeview = QTreeView()
        self.listview = QListView()
        hlay.addWidget(self.treeview)
        hlay.addWidget(self.listview)
        
        self.dirModel = QtWidgets.QFileSystemModel()
        self.dirModel.setRootPath(QDir.rootPath())
        self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

        self.fileModel = QtWidgets.QFileSystemModel()
        self.fileModel.setFilter(QDir.NoDotAndDotDot |  QDir.Files)

        self.treeview.setModel(self.dirModel)
        self.listview.setModel(self.fileModel)

        self.treeview.setRootIndex(self.dirModel.index(path))
        self.listview.setRootIndex(self.fileModel.index(path))
        self.treeview.clicked.connect(self.onClicked)

        self.treeview.setFixedSize(720, 480)
        self.treeview.setColumnWidth(0, 500)

    def onComputeMISignal(self):
        self.thread_1 = ShowJsonWorker(self.c)
        self.thread_1.startComputeMI.connect(self.doComputeMI)
        self.thread_1.start()

    def doComputeMI(self):
        try:
            self.c.path = os.path.join('results', self.path)
            self.c.launch_computeMI_Thread()
        except Exception as e:
            print (e)
            print ("You should firstly specify a folder")


    def onClicked(self, index):
        self.path = self.dirModel.fileInfo(index).absoluteFilePath()
        self.listview.setRootIndex(self.fileModel.setRootPath(self.path))


    def onMainWindowClicked(self):
        if not self.isVisible():
            self.show()
            self.setWindowTitle("Json folders")



class ProgressBarHelper(QWidget):

    def __init__(self, show_json):
        super(ProgressBarHelper, self).__init__()
        self.c = show_json.c
        self.initUi()

    def initUi(self):
        self.setFixedSize(500, 90)
        self.setObjectName("Form")
        self.main_widget = QtWidgets.QWidget(self)
        self.progressBar = QtWidgets.QProgressBar(self.main_widget)
        self.progressBar.setGeometry(QtCore.QRect(20, 20, 450, 50))
        self.thread_1 = ProgressBarWorker(self.c)
        self.thread_1.progressBarValue.connect(self.update_progress_bar)
        

    def update_progress_bar(self, i):
        self.progressBar.setValue(i)

    def onMainWindowClicked(self):
        if not self.isVisible():
            self.show()
            self.setWindowTitle("ComputeMI Progress")
            self.thread_1.start()



class ShowJsonWorker(QThread):

    startComputeMI = pyqtSignal() 

    def __init__(self, c):
        super(ShowJsonWorker, self).__init__()
        self.c = c

    def run(self):
        self.startComputeMI.emit()


class ProgressBarWorker(QThread):

    progressBarValue = pyqtSignal(int) 

    def __init__(self, c):
        super(ProgressBarWorker, self).__init__()
        self.c = c

    def run(self):
        QApplication.processEvents()
        while True:
            time.sleep(0.5)
            value = self.c.progress_bar
            QApplication.processEvents()
            self.progressBarValue.emit(value) 
            if value >= 100:
                break



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = ShowImgHelper()
    w.show()
    sys.exit(app.exec_())
