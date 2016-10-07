# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:39:29 2016

@author: samg
"""

import itertools, random
import numpy as np
from psychopy import visual,event
from VisStimControl import VisStimControl


class MovingSpotGrid(VisStimControl):

    def __init__(self):
        VisStimControl.__init__(self)
        self.gridBoundaries = [40,0,80,40] # [left,bottom,right,top]
        self.gridSpacing = 40 # degrees
        self.spotColor = [-1] # -1 (black) to 1 (white)
        self.spotSize = [4,8,16,32] # degerees
        self.spotSpeed = [10,30,90] # degrees/s
        self.spotDir = [0,90,180,270] # list containing any of [0,90,180,270]
        self.interTrialInterval = [0.25,0.25] # [min,max] seconds
        self.numLoops = 10
          
    def checkParameterValues(self):
        for param in ('spotSize','spotSpeed','spotDir'):
            if isinstance(getattr(self,param),(int,float)):
                setattr(self,param,[getattr(self,param)])
        if not isinstance(self.gridBoundaries,list) or len(self.gridBoundaries)!=4 or self.gridBoundaries[0]>self.gridBoundaries[2] or self.gridBoundaries[1]>self.gridBoundaries[3]:
            raise ValueError('gridBoundaries must be list [left,bottom,right,top]')
        if any((True for color in self.spotColor if not -1<=color<=1)):
            raise ValueError('spotColor must be -1 to 1')
        if any((True for dir in self.spotDir if dir not in (0,90,180,270))):
            raise ValueError('spotDir must be 0, 90 ,180 and/or 270')
        if not isinstance(self.interTrialInterval,list) or len(self.interTrialInterval)!=2 or self.interTrialInterval[0]>self.interTrialInterval[1]:
            raise ValueError('interTrialInterval must be a list of [min,max] duration in seconds')
     
    def run(self):
        self.checkParameterValues()
        
        self.prepareRun()
        
        # create spot stim
        spot = visual.Rect(self._win,units='pix')
        
        # get trialTypes
        self.elevation = np.arange(self.gridBoundaries[1],self.gridBoundaries[3]+0.1,self.gridSpacing)
        self.azimuth = np.arange(self.gridBoundaries[0],self.gridBoundaries[2]+0.1,self.gridSpacing)
        trialTypes = list(itertools.product(self.laserPower,self.elevation,self.spotColor,self.spotSize,self.spotSpeed,[dir for dir in self.spotDir if dir in (0,180)]))
        trialTypes.extend(itertools.product(self.laserPower,self.azimuth,self.spotColor,self.spotSize,self.spotSpeed,[dir for dir in self.spotDir if dir in (90,270)]))
        random.shuffle(trialTypes) 
        
        # run
        frame = 0
        loop = 0
        trial = -1
        trialFrame = 0
        trialInterval = self.getInterTrialInterval()
        self.trialStartFrame = []
        self.trialNumFrames = []
        self.trialLaserPower  = []
        self.trialSpotPos = []
        self.trialSpotColor = []
        self.trialSpotSize = []
        self.trialSpotSpeed = []
        self.trialSpotDir = []
        while True:
            if trialFrame==trialInterval-1:
                if trial==len(trialTypes)-1:
                    loop += 1
                    print('loops completed = '+str(loop))
                    if loop==self.numLoops:
                        break
                    else:
                        trial = 0
                        random.shuffle(trialTypes)
                else:
                    trial += 1
                trialFrame = -1
                self.trialStartFrame.append(frame+1)
                self.trialLaserPower.append(trialTypes[trial][0])
                self.trialSpotPos.append(trialTypes[trial][1])
                self.trialSpotColor.append(trialTypes[trial][2])
                self.trialSpotSize.append(trialTypes[trial][3])
                self.trialSpotSpeed.append(trialTypes[trial][4])
                self.trialSpotDir.append(trialTypes[trial][5])
                spotSize = self.trialSpotSize[-1]*self.pixelsPerDeg
                if self.trialSpotDir[-1] in (0,180):
                    startPos,endPos = -spotSize/2,self.fieldWidth/2+spotSize/2
                    if self.trialSpotDir[-1]==180:
                        startPos,endPos = endPos,startPos
                    spot.pos = [startPos,self.trialSpotPos[-1]*self.pixelsPerDeg] 
                else:
                    startPos,endPos = -self.lowerFieldHeight-spotSize/2,self.fieldHeight-self.lowerFieldHeight+spotSize/2
                    if self.trialSpotDir[-1]==270:
                        startPos,endPos = endPos,startPos
                    spot.pos = [self.trialSpotPos[-1]*self.pixelsPerDeg,startPos]
                spot.lineColor = self.trialSpotColor[-1]
                spot.fillColor = self.trialSpotColor[-1]
                spot.height = spotSize
                spot.width = spotSize
                movePerFrame = round(self.trialSpotSpeed[-1]*self.pixelsPerDeg/self.frameRate)
                self.trialNumFrames.append(round(abs(endPos-startPos)/movePerFrame))
                trialInterval = self.laserPreFrames+self.trialNumFrames[-1]+self.laserPostFrames+self.getInterTrialInterval()
            elif trial>-1:
                if trialFrame==0:
                    self.setLaserOn(self.trialLaserPower[-1])
                if self.laserPreFrames<=trialFrame<self.laserPreFrames+self.trialNumFrames[-1]:
                    if self.trialSpotDir[-1]==0:
                        spot.pos[0] += movePerFrame
                    elif self.trialSpotDir[-1]==180:
                        spot.pos[0] -= movePerFrame
                    elif self.trialSpotDir[-1]==90:
                        spot.pos[1] += movePerFrame
                    else:
                        spot.pos[1] -= movePerFrame
                    spot.pos = spot.pos
                    spot.draw()
                if trialFrame==self.laserPreFrames+self.trialNumFrames[-1]+self.laserPostFrames:
                    self.setLaserOff()
            self.visStimFlip()
            frame += 1
            trialFrame += 1
            if len(event.getKeys())>0:                           
                event.clearEvents()
                break
        self.completeRun()
        
    def getInterTrialInterval(self):
        return round((random.random()*(self.interTrialInterval[1]-self.interTrialInterval[0])+self.interTrialInterval[0])*self.frameRate)
        

if __name__=="__main__":
    pass