# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:50:20 2016

@author: SVC_CCG
"""

"""
Created on Mon Jan 18 13:36:05 2016

@author: corbettb

"""

import random
from psychopy import visual, event
from VisStimControl import VisStimControl
import numpy as np

class fullFieldFlash(VisStimControl):
    
    def __init__(self):
        VisStimControl.__init__(self)
        self.stimDur = 30
        self.grayDur = 30
        self.grayLevels = [-1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0]
        self.numCycles = 15
        self.prestimtime = 30
       
    
    def run(self):
        
        #setup monitor
        self.prepareRun()
        self._win.setRecordFrameIntervals(True)
        
        self.stimHistory = []
        self.stimStartFrames = []
        
        stim = visual.Rect(self._win,units='pix')   
        stim.width = self.monPix[0]        
        stim.height = self.monPix[1]         
        grayLevels = np.array(self.grayLevels)
        
        #show pre stim gray
        nframes = 0
        while nframes < self.prestimtime:
            self._win.flip()
            nframes += 1

        #run stimulus
        trialDur = self.stimDur + self.grayDur
        trial = 0
        nframes = 0
        ncycles = 0  
        while True:
            cycleIndex = nframes%(trialDur*len(self.grayLevels))
            if cycleIndex==0:
                ncycles+=1
                print 'starting cycle:', ncycles    
                np.random.shuffle(grayLevels)
            
            trialframe = nframes%trialDur
            if trialframe >= self.grayDur:
                if trialframe == self.grayDur:
                    stim.fillColor = grayLevels[trial%grayLevels.size]                
                    stim.lineColor = grayLevels[trial%grayLevels.size]     
                    self.stimHistory.append(stim.fillColor[0])
                    self.stimStartFrames.append(nframes)
                    trial+=1
                stim.draw()                

            
            self.visStimFlip()
            nframes += 1
            
            if len(event.getKeys())>0:                           
                event.clearEvents()
                break
            
        self.completeRun()
        
if __name__ == "__main__":
    pass