# -*- coding: utf-8 -*-
"""
GUI for laser control

@author: samg
"""

import sip
sip.setapi('QString', 2)
import time
import serial
import numpy as np
import nidaq
from PyQt4 import QtGui, QtCore
import VisStimControl


def start():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    w = LaserControl(app)
    app.exec_()


class LaserControl():
    
    def __init__(self,app):
        self.app = app
        
        self.serialPort = None
        self.nidaqDigitalOut = None
        self.nidaqAnalogOut = None
        self.visControl = None
        
        winWidth = 500
        winHeight = 300
        self.mainWin = QtGui.QMainWindow()
        self.mainWin.setWindowTitle('LaserControl')
        self.mainWin.closeEvent = self.mainWinCloseEvent
        self.mainWin.resize(winWidth,winHeight)
        screenCenter = QtGui.QDesktopWidget().availableGeometry().center()
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(screenCenter)
        self.mainWin.move(mainWinRect.topLeft())
        
        self.devices = ('LED','Blue Laser','Orange Laser')
        self.deviceMenu = QtGui.QComboBox()
        self.deviceMenu.addItems(self.devices)
        self.deviceMenu.currentIndexChanged.connect(self.deviceMenuCallback)
        
        self.powerControl = QtGui.QDoubleSpinBox()
        self.powerControl.setPrefix('Power:  ')
        self.powerControl.valueChanged.connect(self.powerControlCallback)
        
        self.zeroOffsetControl = QtGui.QDoubleSpinBox()
        self.zeroOffsetControl.setPrefix('Zero Offset:  ')
        self.zeroOffsetControl.setSuffix(' V')
        self.zeroOffsetControl.setDecimals(2)
        self.zeroOffsetControl.setRange(0,1)
        self.zeroOffsetControl.setSingleStep(0.05)
        
        self.rampDurControl = QtGui.QDoubleSpinBox()
        self.rampDurControl.setPrefix('Ramp:  ')
        self.rampDurControl.setSuffix(' s')
        self.rampDurControl.setDecimals(3)
        self.rampDurControl.setRange(0,1)
        self.rampDurControl.setSingleStep(0.05)
        self.rampDurControl.setValue(0.25)
        
        self.modeMenu = QtGui.QComboBox()
        self.modeMenu.addItems(('Continuous','Pulse'))
        self.modeMenu.currentIndexChanged.connect(self.modeMenuCallback)
        
        self.pulseNumControl = QtGui.QSpinBox()
        self.pulseNumControl.setPrefix('# Pulses:  ')
        self.pulseNumControl.setRange(1,1000)
        self.pulseNumControl.setSingleStep(1)
        self.pulseNumControl.setValue(1)
        self.pulseNumControl.setEnabled(False)
        
        self.pulseDurControl = QtGui.QDoubleSpinBox()
        self.pulseDurControl.setPrefix('Pulse Duration:  ')
        self.pulseDurControl.setSuffix(' s')
        self.pulseDurControl.setDecimals(3)
        self.pulseDurControl.setRange(0.001,60)
        self.pulseDurControl.setSingleStep(0.1)
        self.pulseDurControl.setValue(2.5)
        self.pulseDurControl.valueChanged.connect(self.pulseDurControlCallback)
        self.pulseDurControl.setEnabled(False)
        
        self.pulseIntervalControl = QtGui.QDoubleSpinBox()
        self.pulseIntervalControl.setPrefix('Pulse Interval:  ')
        self.pulseIntervalControl.setSuffix(' s')
        self.pulseDurControl.setDecimals(3)
        self.pulseIntervalControl.setRange(0.001,60)
        self.pulseIntervalControl.setSingleStep(0.1)
        self.pulseIntervalControl.setValue(5)
        self.pulseIntervalControl.setEnabled(False)
        
        self.onOffButton = QtGui.QPushButton('Start',checkable=True)
        self.onOffButton.clicked.connect(self.onOffButtonCallback)
        
        self.grayScreenCheckbox = QtGui.QCheckBox('Gray Screen')
        self.grayScreenCheckbox.clicked.connect(self.grayScreenCheckboxCallback)
        
        self.mainWidget = QtGui.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtGui.QGridLayout()
        self.mainWidget.setLayout(self.mainLayout)
        setLayoutGridSpacing(self.mainLayout,winHeight,winWidth,5,3)
        self.mainLayout.addWidget(self.deviceMenu,0,1,1,1)
        self.mainLayout.addWidget(self.grayScreenCheckbox,0,2,1,1)
        self.mainLayout.setAlignment(self.grayScreenCheckbox,QtCore.Qt.AlignHCenter)
        self.mainLayout.addWidget(self.powerControl,1,0,1,1)
        self.mainLayout.addWidget(self.zeroOffsetControl,2,0,1,1)
        self.mainLayout.addWidget(self.rampDurControl,3,0,1,1)
        self.mainLayout.addWidget(self.modeMenu,1,1,1,1)
        self.mainLayout.addWidget(self.pulseNumControl,2,1,1,1)
        self.mainLayout.addWidget(self.pulseDurControl,3,1,1,1)
        self.mainLayout.addWidget(self.pulseIntervalControl,4,1,1,1)
        self.mainLayout.addWidget(self.onOffButton,1,2,1,1)
        
        self.deviceMenuCallback(0)
        self.mainWin.show()
        
    def deviceMenuCallback(self,ind):
        self.selectedDevice = self.devices[ind]
        if self.selectedDevice!='LED': 
            self.serialPort = serial.Serial()
            self.serialPort.port = 'COM4' if self.selectedDevice=='Blue Laser' else 'COM5'
            self.serialPort.baudrate = 115200
            self.serialPort.bytesize = serial.EIGHTBITS
            self.serialPort.stopbits = serial.STOPBITS_ONE
            self.serialPort.parity = serial.PARITY_NONE
            self.serialPort.open()
            if self.selectedDevice=='Blue Laser':
                self.serialPort.write('em\r sdmes 0\r sames 1\r') # analog modulation mode
        if self.selectedDevice=='Orange Laser':
            self.controlType = 'digital'
            self.nidaqDigitalOut = nidaq.DigitalOutputs(device='Dev1',port=1,initialState='high')
            self.nidaqDigitalOut.StartTask()
            self.nidaqDigitalOut.Write(self.nidaqDigitalOut.lastOut)
            self.nidaqDigitalOutCh = 1
            self.powerControl.setSuffix(' mW')
            self.powerControl.setDecimals(1)
            self.powerControl.setRange(0,100)
            self.powerControl.setSingleStep(0.5)
            self.powerControl.setValue(100)
            self.zeroOffsetControl.setEnabled(False)
            self.rampDurControl.setEnabled(False)
        else:
            self.controlType = 'analog'
            self.nidaqAnalogOut = nidaq.AnalogOutput(device='Dev1',channel=0,voltageRange=(0,5))
            self.nidaqAnalogOut.StartTask()
            self.nidaqAnalogOut.Write(np.array([0.0]))
            self.powerControl.setSuffix(' V')
            self.powerControl.setDecimals(2)
            self.powerControl.setSingleStep(0.05)
            if self.selectedDevice=='LED':
                self.powerControl.setRange(0,5)
                self.powerControl.setValue(2)
                self.zeroOffsetControl.setValue(0.25)
            else:
                self.powerControl.setRange(0,1.3)
                self.powerControl.setValue(1)
                self.zeroOffsetControl.setValue(0.6)
            self.zeroOffsetControl.setEnabled(True)
            self.rampDurControl.setEnabled(True)
        
    def powerControlCallback(self,val):
        if self.controlType=='digital':
            self.serialPort.write('p '+str(val/1e3)+'\r')
        else:
            if val<self.zeroOffsetControl.value():
                self.zeroOffsetControl.setValue(val)
    
    def modeMenuCallback(self,ind):
        if ind==0:
            self.pulseNumControl.setEnabled(False)
            self.pulseIntervalControl.setEnabled(False)
            self.pulseDurControl.setEnabled(False)
        else:
            self.pulseNumControl.setEnabled(True)
            self.pulseIntervalControl.setEnabled(True)
            self.pulseDurControl.setEnabled(True)
            
    def pulseDurControlCallback(self,val):
        if self.controlType=='analog':
            if val<self.rampDurControl.value():
                self.rampDurControl.setValue(val)
    
    def onOffButtonCallback(self,val):
        if self.onOffButton.isChecked():
            self.onOffButton.setText('Stop')
            if self.controlType=='analog':
                power = self.powerControl.value()
                rampDur = self.rampDurControl.value()
                if rampDur>0:
                    ramp = np.linspace(self.zeroOffsetControl.value(),power,round(rampDur*self.nidaqAnalogOut.sampRate))
            if self.modeMenu.currentIndex()==0:
                if self.controlType=='digital':
                    self.nidaqDigitalOut.WriteBit(self.nidaqDigitalOutCh,0)
                else:
                    if rampDur>0:
                        self.nidaqAnalogOut.Write(ramp)
                    else:
                        self.nidaqAnalogOut.Write(np.array([power]))
            else:
                pulseDur = self.pulseDurControl.value()
                pulseInt = self.pulseIntervalControl.value()
                for i in range(self.pulseNumControl.value()):
                    if i>0:
                        time.sleep(pulseInt)
                    self.app.processEvents()
                    if not self.onOffButton.isChecked():
                        return
                    if self.controlType=='digital':
                        self.nidaqDigitalOut.WriteBit(self.nidaqDigitalOutCh,0)
                        time.sleep(pulseDur)
                        self.nidaqDigitalOut.WriteBit(self.nidaqDigitalOutCh,1)
                    else:
                        if rampDur>0:
                            t = time.clock()
                            self.nidaqAnalogOut.Write(ramp)
                            while time.clock()-t<pulseDur:
                                time.sleep(1/self.nidaqAnalogOut.sampRate)
                        else:
                            self.nidaqAnalogOut.Write(np.array([power]))
                            time.sleep(pulseDur)
                        self.nidaqAnalogOut.Write(np.array([0.0]))
                self.onOffButton.click()
        else:
            self.onOffButton.setText('Start')
            if self.controlType=='digital':
                self.nidaqDigitalOut.WriteBit(self.nidaqDigitalOutCh,1)
            else:
                self.nidaqAnalogOut.Write(np.array([0.0]))
                
# pulseSamples = round(self.nidaqAnalogOut.sampRate*self.pulseDurControl.value())
# intervalSamples = round(self.nidaqAnalogOut.sampRate*self.pulseIntervalControl.value())
# pulseTrain = np.zeros((self.pulseNumControl.value(),pulseSamples+intervalSamples))
# pulseTrain[:,:pulseSamples] = self.powerControl.value()
# self.nidaqAnalogOut.Write(pulseTrain.ravel()[:-intervalSamples+1])
                
    def grayScreenCheckboxCallback(self):
        if self.visControl is None:
            self.visControl = VisStimControl.VisStimControl()
            self.visControl.prepareWindow()
            self.visControl._win.flip()
        else:
            self.visControl._win.close()
            self.visControl = None
                
    def closeDevice(self):
        if self.serialPort is not None:
            self.serialPort.close()
        if self.nidaqDigitalOut is not None:
            self.nidaqDigitalOut.StopTask()
            self.nidaqDigitalOut.ClearTask()
        if self.nidaqAnalogOut is not None:
            self.nidaqAnalogOut.StopTask()
            self.nidaqAnalogOut.ClearTask()
        
    def mainWinCloseEvent(self,event):
        if self.visControl is not None:
            self.visControl._win.close()
        self.closeDevice()
        event.accept()
        

def setLayoutGridSpacing(layout,height,width,rows,cols):
    for row in range(rows):
        layout.setRowMinimumHeight(row,height/rows)
        layout.setRowStretch(row,1)
    for col in range(cols):
        layout.setColumnMinimumWidth(col,width/cols)
        layout.setColumnStretch(col,1)
        

if __name__=="__main__":
    start()