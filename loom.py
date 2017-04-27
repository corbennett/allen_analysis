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
        
        nframes=0
        #show pre stim gray
        while nframes < self.prestimtime:
            self._win.flip()
            nframes += 1
        
        self.lvHistory = []
        self.posHistory = []
        self.colorHistory = []
        
        #initialize stim      
        numConditions = len(self.lvratio) * len(self.colors) * len(self.xpos) * len(self.ypos)
        numTrials=0
        paramVect = list(itertools.product(self.lvratio, self.colors, self.xpos, self.ypos))
        numCycles = 0
        stim = visual.Circle(self._win,units='pix')
        nframes=0
        breakFlag=False
        while not breakFlag:
            if numTrials%numConditions==0:
                print('Starting loop: ' + str(numCycles+1))
                numCycles += 1
                paramVect = np.random.permutation(paramVect)
            trialFrames = 0
            trialLV = paramVect[numTrials%numConditions][0]
            trialColor = paramVect[numTrials%numConditions][1]
            trialX = paramVect[numTrials%numConditions][2]*self.pixelsPerDeg
            trialY = paramVect[numTrials%numConditions][3]*self.pixelsPerDeg

            numTrials += 1
            
            self.lvHistory.append(trialLV)
            self.posHistory.append([trialX, trialY])
            self.colorHistory.append(trialColor)
            self.stimStartFrames.append(nframes)

            stim.fillColor = trialColor
            stim.lineColor = trialColor

            stim.pos = [trialX, trialY]
            stim.radius=0
            stimRadiusVector = self.calculateLoomRadius(trialLV)
            #Run Loom
            for rad in stimRadiusVector:
                stim.radius = rad
                stim.draw()
                self.visStimFlip()
                trialFrames += 1
                nframes += 1
                
                if len(event.getKeys())>0:
                    breakFlag = True                                        
                    event.clearEvents()
                    break
            
            #Pause Loom at end position
            trialFrames = 0
            stim.radius = self.maxRadius*self.pixelsPerDeg
            while trialFrames <= self.pauseFrames and not breakFlag:                
                stim.draw()
                self.visStimFlip()
                trialFrames += 1
                nframes += 1
            
                if len(event.getKeys())>0:
                    breakFlag = True                                        
                    event.clearEvents()
                    break
            
            #Intertrial interval
            trialFrames = 0
            while trialFrames <= self.interTrialInterval and not breakFlag:
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