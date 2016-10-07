# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 12:36:46 2016

@author: corbettb
"""

from psychopy import visual, event
import numpy as np
from VisStimControl import VisStimControl
import itertools, random
import time

class makeSparseNoise(VisStimControl):
    
    def __init__(self):
        VisStimControl.__init__(self)        
        self.boxSize = [5,10,20]                         #degrees
        self.boxColors = [1, -1]
        self.prestimtime = 60                       #number of gray refreshes show before boxes begin
        self.interBoxInterval = 0                 #frequency to display boxes
        self.boxDuration = 6                      #number of frames to show each box
        self.trialBoxPosition = [0, 0]              # x and y
        self.trialBoxColor = 1        
        self.gridBoundaries = [-20, -30, 120, 90]    #Screen coordinates define presentation window (degrees): x1 y1 for lower left, x2 y2 for upper right corner
        self.gridSpacing = 10                        #Spacing between grid nodes (degrees)
        
     
    def run(self):
        # clear history        
        self.boxPositionHistory = []
        self.boxColorHistory = []
        self.boxSizeHistory = []
        self.stimStartFrames = []
        #setup monitor
        self.prepareRun()
        
        #establish grid for stim boxes
        self.gridX = np.linspace(self.gridBoundaries[0], self.gridBoundaries[2], 1 + (self.gridBoundaries[2] - self.gridBoundaries[0])/self.gridSpacing)
        self.gridY = np.linspace(self.gridBoundaries[1], self.gridBoundaries[3], 1 + (self.gridBoundaries[3] - self.gridBoundaries[1])/self.gridSpacing)
        
        stim = visual.Rect(self._win,units='pix')
 
        self.trialDuration = self.boxDuration + self.interBoxInterval
        ntotalframes = 0
        
        #show pre stim gray
        while ntotalframes < self.prestimtime:
            self._win.flip()
            ntotalframes += 1
        
        #begin noise stimulus
        breakFlag = False
        trialsPerCycle = len(self.gridX) * len(self.gridY) * len(self.boxSize) * len(self.boxColors)
        ntotalframes = 0
        numTrials = 0
        while True and not(breakFlag):
            
            #reset params every full cycle
            if numTrials % trialsPerCycle == 0:
                self.set_params()
                print 'starting cycle:', numTrials/trialsPerCycle + 1
                
            #set stim for this trial
            stim.pos = self._parameterCombos[numTrials%trialsPerCycle][0:2]
            stim.fillColor = self._parameterCombos[numTrials%trialsPerCycle][2]
            stim.lineColor = self._parameterCombos[numTrials%trialsPerCycle][2]
            stim.height = self.pixelsPerDeg * self._parameterCombos[numTrials%trialsPerCycle][3]
            stim.width = self.pixelsPerDeg * self._parameterCombos[numTrials%trialsPerCycle][3]
            
            
            ntrialframes = 0
            self.stimStartFrames.append(ntotalframes)
            while (ntrialframes < self.trialDuration):
                
                if (ntrialframes < self.boxDuration):
                    stim.draw()
                
                self.visStimFlip()
                
                ntrialframes += 1
                ntotalframes += 1
                if len(event.getKeys())>0:
                    breakFlag = True                                        
                    event.clearEvents()
                    break
            if not(breakFlag):    
                self.boxPositionHistory.append(list(stim.pos))
                self.boxColorHistory.append(stim.fillColor)
                self.boxSizeHistory.append(stim.width)
                numTrials += 1

        print ntotalframes
        print numTrials
        self.totalFrames = ntotalframes
        self.completeRun()
        
    def set_params(self):
        self._parameterCombos = list(itertools.product(self.pixelsPerDeg * self.gridX, self.pixelsPerDeg * self.gridY, self.boxColors, self.boxSize))
        random.shuffle(self._parameterCombos)
 
        
if __name__ == "__main__":
    pass