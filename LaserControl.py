# -*- coding: utf-8 -*-
"""
GUI for laser control

@author: samg
"""

import sip
sip.setapi('QString', 2)
import time
import numpy as np
from PyQt4 import QtGui,QtCore
import serial
import nidaq


def start():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    w = LaserControlGUI(app)
    app.exec_()


class LaserControlGUI():
    
    def __init__(self,app):
        self.app = app
        
        winWidth = 300
        winHeight = 300
        self.mainWin = QtGui.QMainWindow()
        self.mainWin.setWindowTitle('LaserControl')
        self.mainWin.closeEvent = self.mainWinCloseEvent
        self.mainWin.resize(winWidth,winHeight)
        screenCenter = QtGui.QDesktopWidget().availableGeometry().center()
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(screenCenter)
        self.mainWin.move(mainWinRect.topLeft())
        
        self.nidaqDigOut = nidaq.DigitalOutputs(device='Dev1',port=1,initialState='high')
        self.nidaqDigOut.StartTask()
        self.nidaqDigOut.Write(self.nidaqDigOut.lastOut)
        
        self.blueLaser = LaserControlObj('Blue Laser','COM4')
        
        self.orangeLaser =LaserControlObj('Orange Laser','COM5',shutterControl='digital',nidaqDigOut=self.nidaqDigOut)
        
        
        setLayoutGridSpacing(self.blueLaser.layout,winHeight/2,winWidth,5,3)
        setLayoutGridSpacing(self.orangeLaser.layout,winHeight/2,winWidth,5,3)
        
        self.mainWidget = QtGui.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtGui.QGridLayout()
        self.mainWidget.setLayout(self.mainLayout)
        setLayoutGridSpacing(self.mainLayout,winHeight,winWidth,9,1)
        self.mainLayout.addLayout(self.blueLaser.layout,0,0,4,1)
        self.mainLayout.addLayout(self.orangeLaser.layout,5,0,4,1)
        self.mainWin.show()
        
    def mainWinCloseEvent(self,event):
        self.blueLaser.serialPort.close()
        self.blueLaser.nidaqAnalogOut.StopTask()
        self.blueLaser.nidaqAnalogOut.ClearTask()
        self.orangeLaser.serialPort.close()
        self.orangeLaser.nidaqDigOut.StopTask()
        self.orangeLaser.nidaqDigOut.ClearTask()
        event.accept()
    

class LaserControlObj():
    
    def __init__(self,label,port,shutterControl='analog',nidaqDigOut=None):
        self.label = QtGui.QLabel(label)
        self.label.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.serialPort = serial.Serial()
        self.serialPort.port = port
        self.serialPort.baudrate = 115200
        self.serialPort.bytesize = serial.EIGHTBITS
        self.serialPort.stopbits = serial.STOPBITS_ONE
        self.serialPort.parity = serial.PARITY_NONE
        self.serialPort.open()
        if shutterControl=='analog':
            self.serialPort.write('cp\r')
            self.serialPort.write('p 0\r')
        
        self.shutterControl = shutterControl
        if shutterControl=='digital':
            self.nidaqDigOut = nidaqDigOut
            self.nidaqDigOutCh = 1
        else:
            self.nidaqAnalogOut = nidaq.AnalogOutput(device='Dev1',channel=0,voltageRange=(0,5))
            self.nidaqAnalogOut.StartTask()
        
        self.powerControl = QtGui.QDoubleSpinBox()
        self.powerControl.setPrefix('Power:  ')
        self.powerControl.setSuffix(' mW')
        self.powerControl.setDecimals(1)
        self.powerControl.setRange(0,100)
        self.powerControl.setSingleStep(0.5)
        self.powerControl.setValue(100)
        self.powerControl.valueChanged.connect(self.powerControlChanged)
        
        self.modeMenu = QtGui.QComboBox()
        self.modeMenu.addItems(('Continuous','Pulse'))
        self.modeMenu.currentIndexChanged.connect(self.modeMenuChanged)
        
        self.pulseNumControl = QtGui.QSpinBox()
        self.pulseNumControl.setPrefix('# Pulses:  ')
        self.pulseNumControl.setRange(1,1000)
        self.pulseNumControl.setSingleStep(1)
        self.pulseNumControl.setValue(1)
        self.pulseNumControl.setEnabled(False)
        self.pulseNumControl.valueChanged.connect(self.pulseNumChanged)
        
        self.pulseDurControl = QtGui.QDoubleSpinBox()
        self.pulseDurControl.setPrefix('Pulse Duration:  ')
        self.pulseDurControl.setSuffix(' s')
        self.pulseDurControl.setDecimals(3)
        self.pulseDurControl.setRange(0.001,60)
        self.pulseDurControl.setSingleStep(0.1)
        self.pulseDurControl.setValue(1)
        self.pulseDurControl.valueChanged.connect(self.pulseDurChanged)
        self.pulseDurControl.setEnabled(False)
        
        self.pulseIntervalControl = QtGui.QDoubleSpinBox()
        self.pulseIntervalControl.setPrefix('Pulse Interval:  ')
        self.pulseIntervalControl.setSuffix(' s')
        self.pulseDurControl.setDecimals(3)
        self.pulseIntervalControl.setRange(0.001,60)
        self.pulseIntervalControl.setSingleStep(0.1)
        self.pulseIntervalControl.setValue(1)
        self.pulseIntervalControl.valueChanged.connect(self.pulseIntChanged)
        self.pulseIntervalControl.setEnabled(False)
        
        self.onOffButton = QtGui.QPushButton('Laser On',checkable=True)
        self.onOffButton.clicked.connect(self.onOffButtonPress)
        
        self.layout = QtGui.QGridLayout()
        self.layout.addWidget(self.label,0,1,1,1)
        self.layout.addWidget(self.powerControl,1,0,1,1)
        self.layout.addWidget(self.modeMenu,1,1,1,1)
        self.layout.addWidget(self.pulseNumControl,2,1,1,1)
        self.layout.addWidget(self.pulseDurControl,3,1,1,1)
        self.layout.addWidget(self.pulseIntervalControl,4,1,1,1)
        self.layout.addWidget(self.onOffButton,1,2,1,1)
        
    def powerControlChanged(self,val):
        if self.shutterControl=='digital':
            self.serialPort.write('p '+str(val/1e3)+'\r')
    
    def modeMenuChanged(self,ind):
        if ind==0:
            self.pulseNumControl.setEnabled(False)
            self.pulseIntervalControl.setEnabled(False)
            self.pulseDurControl.setEnabled(False)
            if self.shutterControl=='analog':
                self.powerControl.setSuffix(' mW')
                self.powerControl.setDecimals(1)
                self.powerControl.setRange(0,100)
                self.powerControl.setSingleStep(0.5)
                self.powerControl.setValue(100)
                self.serialPort.write('cp\r') # constant power mode
                self.serialPort.write('p 0\r')
        else:
            self.pulseNumControl.setEnabled(True)
            self.pulseIntervalControl.setEnabled(True)
            self.pulseDurControl.setEnabled(True)
            if self.shutterControl=='analog':
                self.powerControl.setSuffix(' V')
                self.powerControl.setDecimals(2)
                self.powerControl.setRange(0,1.5)
                self.powerControl.setSingleStep(0.05)
                self.powerControl.setValue(1)
                self.serialPort.write('em\r sdmes 0\r sames 1\r') # analog modulation mode
    
    def pulseNumChanged(self,val):
        pass
    
    def pulseDurChanged(self,val):
        pass
    
    def pulseIntChanged(self,val):
        pass
    
    def onOffButtonPress(self,val):
        if self.onOffButton.isChecked():
            self.onOffButton.setText('Laser Off')
            if self.modeMenu.currentIndex()==0:
                if self.shutterControl=='digital':
                    self.nidaqDigOut.WriteBit(self.nidaqDigOutCh,0)
                else:
                    self.serialPort.write('p '+str(self.powerControl.value()/1e3)+'\r')
            else:
                if self.shutterControl=='digital':
                    for i in range(self.pulseNumControl.value()):
                        if i>0:
                            time.sleep(self.pulseIntervalControl.value())
                        self.nidaqDigOut.WriteBit(self.nidaqDigOutCh,0)
                        time.sleep(self.pulseDurControl.value())
                        self.nidaqDigOut.WriteBit(self.nidaqDigOutCh,1)
                else:
                    pulseSamples = round(self.nidaqAnalogOut.sampRate*self.pulseDurControl.value())
                    intervalSamples = round(self.nidaqAnalogOut.sampRate*self.pulseIntervalControl.value())
                    pulseTrain = np.zeros((self.pulseNumControl.value(),pulseSamples+intervalSamples))
                    pulseTrain[:,:pulseSamples] = self.powerControl.value()
                    self.nidaqAnalogOut.Write(pulseTrain.ravel()[:-intervalSamples+1])
                self.onOffButton.click()
        else:
            self.onOffButton.setText('Laser On')
            if self.shutterControl=='digital':
                self.nidaqDigOut.WriteBit(self.nidaqDigOutCh,1)
            elif self.modeMenu.currentIndex()==0:
                self.serialPort.write('p 0\r')
    

def setLayoutGridSpacing(layout,height,width,rows,cols):
    for row in range(rows):
        layout.setRowMinimumHeight(row,height/rows)
        layout.setRowStretch(row,1)
    for col in range(cols):
        layout.setColumnMinimumWidth(col,width/cols)
        layout.setColumnStretch(col,1)
        

if __name__=="__main__":
    start()