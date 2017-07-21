# -*- coding: utf-8 -*-
"""
Superclass for visual stimulus control

"""

import os, time, random
import h5py
import numpy as np
from psychopy import monitors, visual
#from psychopy.visual.windowwarp
import ProjectorWindow
import nidaq
import serial

class VisStimControl():
    
    def __init__(self):
        self.rig = 'dome' # 'dome' or '2-photon'
        self._save = True # if True, saves all attributes not starting with underscore
        self._saveFrameIntervals = True
        self.frameRate = 60.0
        self._screen = 1 # monitor to present stimuli on
        self.drawDiodeBox = True
        self.nidaqDevice = 'USB-6009'
        self.wheelRotDir = -1 # 1 or -1
        self.wheelRadius = 7.6 # cm
        self.wheelSpeedGain = 1 # arbitrary scale factor
        self.rewardDur = 0.02 # duration in seconds of analog output pulse controlling reward size
        self.laser = None # 'Blue', 'Orange', 'Blue LED', 'Red LED', or None
        self.laserPower = [0] # 0-100 mW for orange laser, 0-1.5 V for blue laser, or 0-5 V for LED
        self.laserRandom = False
        self.laserPreFrames = 60 # initLaser() sets laser pre and post frames to 0 if laser is None
        self.laserPostFrames = 6
        self.laserRampFrames = 6
        self.blueLaserZeroOffset = 0.6
        self.blueLedZeroOffset = 0.25
        self.redLedZeroOffset = 0.0
        if self.rig=='dome':
            self._saveDir = 'C:\Users\SVC_CCG\Desktop\Data' # path where parameters and data saved
            self.monWidth = None # cm
            self.monDistance = None # cm
            self.monGamma = 1.42 # float or None
            self.monPix = (2560,720)
            self._flipScreenHorz = True
            self.pixelsPerDeg = 425.0/90.0
            self.fieldWidth = 1150 # visual field width in pixels
            self.fieldHeight = 700 # visual field height in pixels
            self.lowerFieldHeight = 200 # pixels from bottom of visual field to zero elevation
            self.horizontalOffset = 0 # pixels from monitor horizontal center to center azimuth
            self.verticalOffset = 0 # pixels from monitor vertical center to center elevation
            self.warp = 'Warpfile' # one of ('Disabled','Spherical','Cylindrical','Curvilinear','Warpfile')
            self.warpFile = 'C:\Users\SVC_CCG\Documents\Python Scripts\warp05272016.data'
            self.warpWarp = True
            self.warpShift = [0,0,0,0] # [left x, left y, right x, right y]
            self.warpAngle = [0,0] # [left, right]
            self.diodeBoxSize = 30
            self.diodeBoxPosition = (123,-47) # in degrees
        elif self.rig=='2-photon':
            self._saveDir = 'C:\Users\svc_ccg\Desktop' # path where parameters and data saved
            self.monWidth = 38.1 # cm
            self.monDistance = 15.24 # cm
            self.monGamma = None # float or None
            self.monPix = (1280,1024)
            self._flipScreenHorz = False
            self.pixelsPerDeg = 1280.0/90.0
            self.fieldWidth = 1280 # visual field width in pixels
            self.fieldHeight = 1024 # visual field height in pixels
            self.lowerFieldHeight = 512 # distance from bottom of visual field to zero elevation in pixels
            self.warp = 'Disabled' # one of ('Disabled','Spherical','Cylindrical','Curvilinear','Warpfile')
            self.warpFile = None
            self.warpWarp = None
            self.warpShift = None
            self.warpAngle = None
            self.diodeBoxSize = 30
            self.diodeBoxPosition = (123,-47) # in degrees
            
    def visStimFlip(self):
        if self.drawDiodeBox:
            self.diodeBox.fillColor = -self.diodeBox.fillColor
            self.diodeBox.lineColor = self.diodeBox.fillColor
            self.diodeBox.draw()
        self.setAcquisitionSignal(0)
        self._win.flip()
        self.setAcquisitionSignal(1)
        
    def setDefaultImageSizeAndPosition(self):
        # use with ImageStimNumpyuByte
        verticalOffset = (self.fieldHeight/2-self.lowerFieldHeight)/2
        self.imageSize = [self.fieldWidth,self.fieldHeight-2*verticalOffset]
        self.imagePosition = [0,verticalOffset]
        
    def prepareRun(self):
        self.startTime = time.strftime('%Y%m%d_%H%M%S')
        
        self.prepareWindow()

        self.diodeBox = visual.Rect(self._win,units='pix', width=self.pixelsPerDeg*self.diodeBoxSize, height=self.pixelsPerDeg*self.diodeBoxSize, lineColor=0,
                                    fillColor=1, pos=(self.pixelsPerDeg*self.diodeBoxPosition[0], self.pixelsPerDeg*self.diodeBoxPosition[1]))
        
        self.numpyRandomSeed = random.randint(0,2**32)
        self._numpyRandom = np.random.RandomState(self.numpyRandomSeed)
        
        self.startNidaqDevice()
        self.initLaser()
        
    def prepareWindow(self):
        self._mon = monitors.Monitor('monitor1',width=self.monWidth,distance=self.monDistance,gamma=self.monGamma)
        self._mon.setSizePix(self.monPix)
        self._mon.saveMon()
        self._win =  ProjectorWindow.ProjectorWindow(monitor=self._mon,screen=self._screen,fullscr=True,flipHorizontal=self._flipScreenHorz,
                                                     warp=getattr(ProjectorWindow.Warp,self.warp),warpfile=self.warpFile,warpWarp=self.warpWarp,
                                                     warpShift=self.warpShift,warpAngle=self.warpAngle, units='pix')      
#        self._win = visual.Window(size = self.monPix, screen = self._screen, gamma = self.monGamma, useFBO=True, fullscr = True, units='pix')
#        self.warper = Warper(self._win, warp = 'warpfile', warpfile = self.warpFile, 
#                                               flipHorizontal = flipScreenHorz)
        self._win.viewPos = [self.horizontalOffset, self.verticalOffset]
        self._win.setRecordFrameIntervals(self._saveFrameIntervals)
                                               
    def completeRun(self):
        self._win.close()
        self.closeLaser()
        self.stopNidaqDevice()
        if self._save:
            fileOut = h5py.File(os.path.join(self._saveDir,'VisStim_'+self.__class__.__name__+'_'+self.startTime+'.hdf5'),'w')
            saveParameters(fileOut,self.__dict__)
            if self._saveFrameIntervals:
                fileOut.create_dataset('frameIntervals',data=self._win.frameIntervals)
            fileOut.close()
        
    def startNidaqDevice(self):
        # rotary encoder: AI0
        self._rotEncoderInput = nidaq.AnalogInput(device='Dev1',channel=0,voltageRange=(0,5),sampRate=1000.0,bufferSize=8)
        self._rotEncoderInput.StartTask()
        
        # blue laser analog modulation: AO0
        self._laserAnalogControl = nidaq.AnalogOutput(device='Dev1',channel=0,voltageRange=(0,5))
        self._laserAnalogControl.StartTask()
        
        # reward delivery: AO1
        self._rewardOut = nidaq.AnalogOutput(device='Dev1',channel=1,voltageRange=(0,5))
        self._rewardOut.StartTask()
        
        # digital inputs (port 0)
        # line 0: lick sensor
        self._digInputs = nidaq.DigitalInputs(device='Dev1',port=0)
        self._digInputs.StartTask()
        
        # digital outputs (port 1)
        # line 0: psychopy acquisition signal
        # line 1: orange laser shutter
        self._digOutputs = nidaq.DigitalOutputs(device='Dev1',port=1,initialState='high')
        self._digOutputs.StartTask()
        self._digOutputs.Write(self._digOutputs.lastOut)
    
    def stopNidaqDevice(self):
        for task in ['_rotEncoderInput','_laserAnalogControl','_rewardOut','_digInputs','_digOutputs']:
            getattr(self,task).StopTask()
            getattr(self,task).ClearTask()
        
    def readRotaryEncoder(self):
        return self._rotEncoderInput.data[:]
        
    def deliverReward(self):
        self._rewardOut.Write(np.concatenate((self._rewardOut.voltageRange[1]*np.ones(round(self._rewardOut.sampRate*self.rewardDur)),[0])))
        
    def getLickInput(self):
        return self._digInputs.Read()[0]
        
    def setAcquisitionSignal(self,level):
        self._digOutputs.WriteBit(0,level)
        
    def initLaser(self):
        if self.laser is None:
            self.laserPower = [0]
            self.laserPreFrames = 0
            self.laserPostFrames = 0
        elif self.laser=='Orange':
            if min(self.laserPower)<0 or max(self.laserPower)>100:
                raise ValueError('orange laser power must be 0 to 100 mW')
            self._laserPort = serial.Serial()
            self._laserPort.port = 'COM5'
            self._laserPort.baudrate = 115200
            self._laserPort.bytesize = serial.EIGHTBITS
            self._laserPort.stopbits = serial.STOPBITS_ONE
            self._laserPort.parity = serial.PARITY_NONE
            self._laserPort.open()
        elif self.laser=='Blue':
            if min(self.laserPower)<0 or max(self.laserPower)>1.5:
                raise ValueError('blue laser power must be 0 to 1.5 V')
            self._laserPort = serial.Serial()
            self._laserPort.port = 'COM4'
            self._laserPort.baudrate = 115200
            self._laserPort.bytesize = serial.EIGHTBITS
            self._laserPort.stopbits = serial.STOPBITS_ONE
            self._laserPort.parity = serial.PARITY_NONE
            self._laserPort.open()
            self._laserPort.write('em\r sdmes 0\r sames 1\r') # analog modulation mode
            self._laserZeroOffset = self.blueLaserZeroOffset
        elif self.laser=='Blue LED':
            self._laserZeroOffset = self.blueLedZeroOffset
        elif self.laser=='Red LED':
            self._laserZeroOffset = self.redLedZeroOffset
        else:
            raise ValueError('laser must be Blue, Orange, Blue LED, Red LED, or None')
        if self.laser is not None:
            self.setLaserOff()
    
    def closeLaser(self):
        if self.laser is not None:
            self.setLaserOff()
            if 'LED' not in self.laser:
                self._laserPort.close()
            
    def setTrialLaserPower(self,trialTypes):
        if len(self.laserPower)>1 and not self.laserRandom:
            for trials in trialTypes:
                random.shuffle(trials)
            shuffledTrials = []
            for trialInd,_ in enumerate(trialTypes[0]):
                for pwrInd,_ in enumerate(self.laserPower):
                    shuffledTrials.append(trialTypes[pwrInd][trialInd])
            return shuffledTrials
        else:
            random.shuffle(trialTypes)
            return trialTypes
            
    def setLaserOn(self,power):
        if power>0:
            if self.laser=='Orange':
                self._laserPort.write('p '+str(power/1e3)+'\r')
                self._digOutputs.WriteBit(1,0)
            else:
                if self.laserRampFrames>0:
                    rampSamples = round(self.laserRampFrames/self.frameRate*self._laserAnalogControl.sampRate)
                    self._laserAnalogControl.Write(np.linspace(self._laserZeroOffset,power,rampSamples))
                else:
                    self._laserAnalogControl.Write(np.array([float(power)]))
        
    def setLaserOff(self):
        if self.laser=='Orange':
            self._digOutputs.WriteBit(1,1)
        else:
            if self.laserRampFrames>0:
                if self._laserAnalogControl.lastOut>self._laserZeroOffset:
                    rampSamples = round(self.laserRampFrames/self.frameRate*self._laserAnalogControl.sampRate)
                    self._laserAnalogControl.Write(np.linspace(self._laserAnalogControl.lastOut,self._laserZeroOffset,rampSamples))
            self._laserAnalogControl.Write(np.array([0.0]))


def saveParameters(fileOut,paramDict,dictName=None):
    for key,val in paramDict.items():
        if key[0] != '_':
            if dictName is None:
                paramName = key
            else:
                paramName = dictName+'_'+key
            if isinstance(val,dict):
                saveParameters(fileOut,val,paramName)
            else:
                try:
                    fileOut.create_dataset(paramName,data=val)
                except:
                    print 'could not save ' + key
                    

if __name__ == "__main__":
    pass