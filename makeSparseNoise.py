# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 12:36:46 2016

@author: corbettb
"""

from psychopy import visual, event
import numpy as np
import copy
from VisStimControl import VisStimControl
import itertools, random


class makeSparseNoise(VisStimControl):
    
    def __init__(self):
        VisStimControl.__init__(self)        
        self.boxSize = [5,10,20,1000]               #degrees: sizes >= 1000 will become full field flashes only shown once per cycle
        self.boxColors = [1, -1]
        self.prestimtime = 360                      #number of gray refreshes show before boxes begin
        self.interBoxInterval = 0                   #frequency to display boxes
        self.boxDuration = 6                        #number of frames to show each box
        self.trialBoxPosition = [0, 0]              # x and y
        self.trialBoxColor = 1        
        self.gridBoundaries = [-20, -30, 120, 90]   #Screen coordinates define presentation window (degrees): x1 y1 for lower left, x2 y2 for upper right corner
        self.gridSpacing = 10                       #Spacing between grid nodes (degrees)
        self.laserBlockTrials = 25                  # number of consecutive trials at a given laser power
        self.postLaserBlockFrames = 240             # frames to wait after laser off
        
     
    def run(self):
        # clear history 
        self.stimStartFrames = []       
        self.boxPositionHistory = []
        self.boxColorHistory = []
        self.boxSizeHistory = []
        self.laserPowerHistory = []
        
        #setup monitor
        self.prepareRun()
        
        #establish grid for stim boxes
        self.gridX = np.linspace(self.gridBoundaries[0], self.gridBoundaries[2], 1 + (self.gridBoundaries[2] - self.gridBoundaries[0])/self.gridSpacing)
        self.gridY = np.linspace(self.gridBoundaries[1], self.gridBoundaries[3], 1 + (self.gridBoundaries[3] - self.gridBoundaries[1])/self.gridSpacing)
        
        stim = visual.Rect(self._win,units='pix')
        
        #initialize stim parameters
        self._parameterCombos = None
        shuffledTrials = self.set_params()
        trialsPerCycle = len(shuffledTrials)
        
        self.trialDuration = self.boxDuration + self.interBoxInterval
        
        #show pre stim gray
        for _ in range(self.prestimtime):
            self._win.flip()
        
        #begin noise stimulus
        breakFlag = False
        self.totalFrames = 0
        numTrials = 0
        while True and not(breakFlag):
            #reset params every full cycle
            if numTrials % trialsPerCycle == 0:
                shuffledTrials = self.set_params()
                print 'starting cycle:', numTrials/trialsPerCycle + 1
                
            #set stim for this trial
            trialInd = numTrials%trialsPerCycle
            stim.pos = shuffledTrials[trialInd][:2]
            stim.fillColor = shuffledTrials[trialInd][2]
            stim.lineColor = shuffledTrials[trialInd][2]
            stim.height = self.pixelsPerDeg * shuffledTrials[trialInd][3]
            stim.width = self.pixelsPerDeg * shuffledTrials[trialInd][3]
            
            # turn laser on before block of laser trials
            trialLaserPwr = shuffledTrials[trialInd][4]
            if trialLaserPwr>0 and (numTrials==0 or self.laserPowerHistory[-1]!=trialLaserPwr):
                self.setLaserOn(trialLaserPwr)
                self.waitWithFlips(self.laserPreFrames)
            
            # run trial
            numTrials += 1
            ntrialframes = 0
            self.stimStartFrames.append(self.totalFrames)
            self.boxPositionHistory.append(list(stim.pos))
            self.boxColorHistory.append(stim.fillColor)
            self.boxSizeHistory.append(stim.width)
            self.laserPowerHistory.append(trialLaserPwr)
            while (ntrialframes < self.trialDuration):            
                if (ntrialframes < self.boxDuration):
                    stim.draw()
                self.visStimFlip()
                ntrialframes += 1
                self.totalFrames += 1
                if len(event.getKeys())>0:
                    breakFlag = True                                        
                    event.clearEvents()
                    break
                
            # turn laser off after block of laser trials
            if any(self.laserPower) and (trialInd==trialsPerCycle-1 or shuffledTrials[trialInd+1][4]!=trialLaserPwr):
                if trialLaserPwr>0:
                    self.waitWithFlips(self.laserPostFrames)
                    self.setLaserOff()
                self.waitWithFlips(self.postLaserBlockFrames)
        
        self.completeRun()
        
        
    def waitWithFlips(self,frames):
        for _ in range(frames):
            self.visStimFlip()
            self.totalFrames += 1
    
    
    def set_params(self):
        if self._parameterCombos is None:
            boxSizes = np.array(copy.copy(self.boxSize))
            boxSizes = boxSizes[boxSizes<1000]
            laserPwr = self.laserPower if self.laserRandom else [self.laserPower[0]]
            self._parameterCombos = list(itertools.product(self.pixelsPerDeg * self.gridX, self.pixelsPerDeg * self.gridY, self.boxColors, boxSizes, laserPwr))
            
            if max(self.boxSize) >= 1000:
                for color in [1, -1]:
                    for pwr in laserPwr:
                        fullField = [0,0,0,300,0]
                        fullField[2] = color
                        fullField[4] = pwr
                        self._parameterCombos.append(fullField)
            
            if len(self.laserPower)>1 and not self.laserRandom:
                self._parameterCombos = [self._parameterCombos]
                for pwr in self.laserPower[1:]:
                    trials = [list(params) for params in self._parameterCombos[0]]
                    for params in trials:
                        params[-1] = pwr
                    self._parameterCombos.append(trials)
                        
        if len(self.laserPower)>1 and not self.laserRandom:
            for trials in self._parameterCombos:
                random.shuffle(trials)
            shuffledTrials = []
            for ind in range(0,len(self._parameterCombos[0]),self.laserBlockTrials):
                for pwrInd,_ in enumerate(self.laserPower):
                    blockTrials = self._parameterCombos[pwrInd][ind:ind+self.laserBlockTrials]
                    for trial in blockTrials:
                        shuffledTrials.append(trial)
            return shuffledTrials
        else:
            random.shuffle(self._parameterCombos)
            return self._parameterCombos
            
            
    def setLaserParams(self,laser,power):
        self.laser = laser
        self.laserPower = power
 
        
if __name__ == "__main__":
    pass