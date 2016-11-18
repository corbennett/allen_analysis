# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 12:36:46 2016

@author: corbettb
"""

from psychopy import visual, event
import numpy as np
from VisStimControl import VisStimControl


class manualMapper(VisStimControl):
    
    def __init__(self):
        VisStimControl.__init__(self)        
        self.boxSize = 50                  #degrees?
        self.boxColors = (-1, 1)
        self.boxDuration = 10              #number of frames to show each box
        self.trialBoxPosition = [0, 0]     # x and y   
        self.boxPositionHistory = []
        self.boxColorHistory = []
        self.toggleColor = False
        self.togglePeriod = 30           #frames between toggles
        self._save = False
        
    def run(self):
        
        #setup monitor
        self.prepareRun()
        self._win.units = 'pix'
        
        event.clearEvents()
        mouse = event.Mouse(win=self._win)
        mouse.setVisible(0)

        stim = visual.Rect(self._win)
        stim.height = self.boxSize
        stim.width = self.boxSize
        stim.pos = (self.trialBoxPosition[0], self.trialBoxPosition[1])
        stim.fillColor = self.boxColors[0]
        stim.lineColor = self.boxColors[0]
        lastMouseWheelButton = 0
        numFrames = 0        
        mousePos = np.array([0,0])
        while True:

#            mousePos = mouse.getPos()
            mousePos += mouse.getRel()
            stim.pos = (mousePos[0], mousePos[1])
            if self.toggleColor and (numFrames % self.togglePeriod == 0):
                stim.fillColor = -stim.fillColor
                stim.lineColor = -stim.lineColor
                
            stim.draw()
            
            self.visStimFlip()
            keys = event.getKeys(['space', 'escape', 'period', 'comma', 'up', 'down', 'p'])
            mouseWheel = mouse.getWheelRel()
            mouseButtons = mouse.getPressed()
            stepSize = 10
            if max(mouseButtons[0], mouseButtons[2])==0:
                stim.height +=  stepSize*mouseWheel[1]
                stim.width +=  stepSize*mouseWheel[1]
            else:
                if [mouseButtons[0], mouseButtons[2]] == [1, 0]:
                    stim.height += stepSize*mouseWheel[1]
                if [mouseButtons[0], mouseButtons[2]] == [0, 1]:
                    stim.width += stepSize*mouseWheel[1]
                if [mouseButtons[0], mouseButtons[2]] == [1, 1]:
                    stim.height = (stim.height + stim.width)/2
                    stim.width = (stim.height + stim.width)/2
            if mouseButtons[1] - lastMouseWheelButton == 1:
                stim.fillColor = -stim.fillColor
                stim.lineColor = -stim.lineColor
            
            if len(keys) > 0:
                if 'space' in keys:
                    self.toggleColor = False if self.toggleColor else True
                if 'comma' in keys:
                    self.togglePeriod -= 5
                    self.togglePeriod = 1 if self.togglePeriod <= 0 else self.togglePeriod
                if 'period' in keys:
                    self.togglePeriod += 5
                if 'up' in keys:
                    stim.ori += 5
                if 'down' in keys:
                    stim.ori -= 5
                if 'escape' in keys:
                    break
                if 'p' in keys:
                    print [pos/self.pixelsPerDeg for pos in stim.pos]
            lastMouseWheelButton = mouseButtons[1]
            numFrames += 1
                
        event.clearEvents()
        self.completeRun()
        
#    def set_params(self):
#        self.trialBoxColor = np.random.choice(self.boxColors)
#        self.trialBoxPosition[0] = np.random.randint(-self.monPix[0]/2, self.monPix[0]/2)
#        self.trialBoxPosition[1] = np.random.randint(-self.monPix[1]/2, self.monPix[1]/2)      
 
        
if __name__ == "__main__":
    pass