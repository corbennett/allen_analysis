# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:08:52 2017

@author: svc_ccg
"""

from psychopy import visual, event
import numpy as np
from VisStimControl import VisStimControl
import itertools

class loom(VisStimControl):
    
    def __init__(self):
        VisStimControl.__init__(self)        
        self.lvratio = [10, 20, 40, 80]              
        self.colors = [-1, 1]
        self.prestimtime = 360                      #number of gray refreshes show before boxes begin
        self.interTrialInterval = 120                 #frequency to display boxes        
        self.xpos = [60]                             # x and y        
        self.ypos = [0, 60]
        self.startRadius = 0.5                        #degrees
        self.maxRadius = 80                         
        self.pauseFrames = 30                       #how long stimulus will remain on screen at max size


    def run(self):
        
        # clear history        
        self.stimStartFrames = []
        
        #setup monitor
        self.prepareRun()

        #check params
        for param in [self.lvratio, self.xpos, self.ypos, self.colors]:
            if type(param) is not list:
                param = [param]
        
        #show pre stim gray
        for _ in range(self.prestimtime):
            self._win.flip()
        
        self.lvHistory = []
        self.posHistory = []
        self.colorHistory = []
        self.laserPowerHistory = []
        
        #initialize stim      
        numConditions = len(self.lvratio) * len(self.colors) * len(self.xpos) * len(self.ypos) * len(self.laserPower)
        laserPwr = self.laserPower if self.laserRandom else [self.laserPower[0]]
        paramVect = list(itertools.product(self.lvratio, self.colors, self.xpos, self.ypos, laserPwr))
        if len(self.laserPower)>1 and not self.laserRandom:
            paramVect = [paramVect]
            for pwr in self.laserPower[1:]:
                trials = [list(params) for params in paramVect[0]]
                for params in trials:
                    params[-1] = pwr
                paramVect.append(trials)
        numTrials=0
        numCycles = 0
        nframes=0
        stim = visual.Circle(self._win,units='pix')
        breakFlag=False
        while not breakFlag:
            if numTrials%numConditions==0:
                print('Starting loop: ' + str(numCycles+1))
                numCycles += 1
                shuffledTrials = self.setTrialLaserPower(paramVect)
            numTrials += 1
            trialFrames = 0
            
            trialInd = numTrials%numConditions
            trialLV = shuffledTrials[trialInd][0]
            trialColor = shuffledTrials[trialInd][1]
            trialX = shuffledTrials[trialInd][2]*self.pixelsPerDeg
            trialY = shuffledTrials[trialInd][3]*self.pixelsPerDeg
            trialLaserPwr = shuffledTrials[trialInd][4]

            self.lvHistory.append(trialLV)
            self.posHistory.append([trialX, trialY])
            self.colorHistory.append(trialColor)
            self.laserPowerHistory.append(trialLaserPwr)

            stim.fillColor = trialColor
            stim.lineColor = trialColor
            stim.pos = [trialX, trialY]
            stim.radius=0
            stimRadiusVector = self.calculateLoomRadius(trialLV)
            
            # Run trial
            self.setLaserOn(trialLaserPwr)
            while trialFrames < self.laserPreFrames + stimRadiusVector.size + self.pauseFrames + self.laserPostFrames + self.interTrialInterval:
                if trialFrames == self.laserPreFrames:
                    self.stimStartFrames.append(nframes)
                if self.laserPreFrames <= trialFrames < self.laserPreFrames + stimRadiusVector.size + self.pauseFrames:
                    if trialFrames < self.laserPreFrames + stimRadiusVector.size:
                        stim.radius = stimRadiusVector[trialFrames-self.laserPreFrames]
                    elif trialFrames == self.laserPreFrames + stimRadiusVector.size:
                        stim.radius = self.maxRadius*self.pixelsPerDeg
                    stim.draw()
                if trialFrames == self.laserPreFrames + stimRadiusVector.size + self.pauseFrames + self.laserPostFrames:
                    self.setLaserOff()
                self.visStimFlip()
                trialFrames += 1
                nframes += 1
                if len(event.getKeys())>0:
                    breakFlag = True                                        
                    event.clearEvents()
                    break

        self.completeRun()
            
    def calculateLoomRadius(self, lvratio):
        time = np.arange(0, 360000, 16.6667)
        halfTheta = np.arctan(lvratio/time)[::-1] * (180./np.pi)
        start = np.where(halfTheta>=self.startRadius)[0][0]
        end = np.where(halfTheta<=self.maxRadius)[0][-1]
        return halfTheta[start:end]*self.pixelsPerDeg

if __name__ == "__main__":
    pass