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
        self.nidaqSecondaryAnalogOut = None
        self.visControl = None
        self.dualPower = [0.86,2.3]
        
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
        
        self.analogOut0Button = QtGui.QRadioButton('0')
        self.analogOut0Button.setChecked(True)
        self.analogOut1Button = QtGui.QRadioButton('1')
        self.analogOutDualButton = QtGui.QRadioButton('Dual')
        self.analogOutDualButton.setEnabled(False)
        self.analogOutGroupLayout = QtGui.QHBoxLayout()
        for button in (self.analogOut0Button,self.analogOut1Button,self.analogOutDualButton):
            button.clicked.connect(self.analogOutButtonCallback)
            self.analogOutGroupLayout.addWidget(button)
        self.analogOutGroupBox = QtGui.QGroupBox('Analog Out Channel')
        self.analogOutGroupBox.setLayout(self.analogOutGroupLayout)
        
        self.powerControl = QtGui.QDoubleSpinBox()
        self.powerControl.setPrefix('Power:  ')
        self.powerControl.valueChanged.connect(self.powerControlCallback)
        
        self.secondaryPowerControl = QtGui.QDoubleSpinBox()
        self.secondaryPowerControl.setPrefix('Secondary Power:  ')
        self.secondaryPowerControl.setSuffix(' V')
        self.secondaryPowerControl.setDecimals(2)
        self.secondaryPowerControl.setSingleStep(0.05)
        self.secondaryPowerControl.setRange(0,5)
        self.secondaryPowerControl.setEnabled(False)
        
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
        self.rampDurControl.setValue(0.1)
        
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
        self.mainLayout.addWidget(self.deviceMenu,0,0,1,1)
        self.mainLayout.addWidget(self.analogOutGroupBox,0,1,1,1)
        self.mainLayout.addWidget(self.grayScreenCheckbox,0,2,1,1)
        self.mainLayout.setAlignment(self.grayScreenCheckbox,QtCore.Qt.AlignHCenter)
        self.mainLayout.addWidget(self.powerControl,1,0,1,1)
        self.mainLayout.addWidget(self.zeroOffsetControl,2,0,1,1)
        self.mainLayout.addWidget(self.rampDurControl,3,0,1,1)
        self.mainLayout.addWidget(self.secondaryPowerControl,4,0,1,1)
        self.mainLayout.addWidget(self.modeMenu,1,1,1,1)
        self.mainLayout.addWidget(self.pulseNumControl,2,1,1,1)
        self.mainLayout.addWidget(self.pulseDurControl,3,1,1,1)
        self.mainLayout.addWidget(self.pulseIntervalControl,4,1,1,1)
        self.mainLayout.addWidget(self.onOffButton,1,2,1,1)
        
        self.deviceMenuCallback(0)
        self.mainWin.show()
        
    def deviceMenuCallback(self,ind):
        self.closeDevice()
        self.analogOutDualButton.setEnabled(False)
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
            self.analogOutGroupBox.setEnabled(False)
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
            self.analogOutGroupBox.setEnabled(True)
            self.controlType = 'analog'
            self.powerControl.setSuffix(' V')
            self.powerControl.setDecimals(2)
            self.powerControl.setSingleStep(0.01)
            if self.selectedDevice=='LED':
                self.analogOut1Button.setChecked(True)
                self.powerControl.setRange(0,5)
                self.powerControl.setValue(2)
                self.zeroOffsetControl.setValue(0.25)
            else:
                self.analogOut0Button.setChecked(True)
                self.analogOutDualButton.setEnabled(True)
                self.powerControl.setRange(0,1.3)
                self.powerControl.setValue(1)
                self.zeroOffsetControl.setValue(0.6)
            self.zeroOffsetControl.setEnabled(True)
            self.rampDurControl.setEnabled(True)
            self.initAnalogOut()
            
    def analogOutButtonCallback(self):
        self.initAnalogOut()
        
    def initAnalogOut(self):
        self.closeAnalogOut()
        ch = 1 if self.analogOut1Button.isChecked() else 0
        self.nidaqAnalogOut = nidaq.AnalogOutput(device='Dev1',channel=ch,voltageRange=(0,5))
        self.nidaqAnalogOut.StartTask()
        self.nidaqAnalogOut.Write(np.array([0.0]))
        if self.analogOutDualButton.isChecked():
            self.nidaqSecondaryAnalogOut = nidaq.AnalogOutput(device='Dev1',channel=1,voltageRange=(0,5))
            self.nidaqSecondaryAnalogOut.StartTask()
            self.nidaqSecondaryAnalogOut.Write(np.array([0.0]))
            self.powerControl.setValue(self.dualPower[0])
            self.secondaryPowerControl.setValue(self.dualPower[1])
            self.secondaryPowerControl.setEnabled(True)
        else:
            self.secondaryPowerControl.setEnabled(False)
        
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
                if self.nidaqSecondaryAnalogOut is not None:
                    secondaryPower = self.secondaryPowerControl.value()
                rampDur = self.rampDurControl.value()
                if rampDur>0:
                    ramp = np.linspace(self.zeroOffsetControl.value(),power,round(rampDur*self.nidaqAnalogOut.sampRate))
                    if self.nidaqSecondaryAnalogOut is not None:
                        secondaryRamp = np.linspace(0.25,secondaryPower,round(rampDur*self.nidaqAnalogOut.sampRate))
            if self.modeMenu.currentIndex()==0:
                if self.controlType=='digital':
                    self.nidaqDigitalOut.WriteBit(self.nidaqDigitalOutCh,0)
                else:
                    if rampDur>0:
                        self.nidaqAnalogOut.Write(ramp)
                        if self.nidaqSecondaryAnalogOut is not None:
                            self.nidaqSecondaryAnalogOut.Write(secondaryRamp)
                    else:
                        self.nidaqAnalogOut.Write(np.array([power]))
                        if self.nidaqSecondaryAnalogOut is not None:
                            self.nidaqSecondaryAnalogOut.Write(np.array([secondaryPower]))
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
                            if self.nidaqSecondaryAnalogOut is not None:
                                self.nidaqSecondaryAnalogOut.Write(secondaryRamp)
                            while time.clock()-t<pulseDur:
                                time.sleep(1/self.nidaqAnalogOut.sampRate)
                        else:
                            self.nidaqAnalogOut.Write(np.array([power]))
                            if self.nidaqSecondaryAnalogOut is not None:
                                self.nidaqSecondaryAnalogOut.Write(np.array([secondaryPower]))
                            time.sleep(pulseDur)
                        self.nidaqAnalogOut.Write(np.array([0.0]))
                        if self.nidaqSecondaryAnalogOut is not None:
                            self.nidaqSecondaryAnalogOut.Write(np.array([0.0]))
                self.onOffButton.click()
        else:
            self.onOffButton.setText('Start')
            if self.controlType=='digital':
                self.nidaqDigitalOut.WriteBit(self.nidaqDigitalOutCh,1)
            else:
                self.nidaqAnalogOut.Write(np.array([0.0]))
                if self.nidaqSecondaryAnalogOut is not None:
                    self.nidaqSecondaryAnalogOut.Write(np.array([0.0]))
                
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
            self.serialPort = None
        if self.nidaqDigitalOut is not None:
            self.nidaqDigitalOut.WriteBit(self.nidaqDigitalOutCh,1)
            self.nidaqDigitalOut.StopTask()
            self.nidaqDigitalOut.ClearTask()
            self.nidaqDigitalOut = None
        self.closeAnalogOut()
        
    def closeAnalogOut(self):
        if self.nidaqAnalogOut is not None:
            self.nidaqAnalogOut.Write(np.array([0.0]))
            self.nidaqAnalogOut.StopTask()
            self.nidaqAnalogOut.ClearTask()
            self.nidaqAnalogOut = None
        if self.nidaqSecondaryAnalogOut is not None:
            self.nidaqSecondaryAnalogOut.Write(np.array([0.0]))
            self.nidaqSecondaryAnalogOut.StopTask()
            self.nidaqSecondaryAnalogOut.ClearTask()
            self.nidaqSecondaryAnalogOut = None
        
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