# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import itertools, random, traceback
from psychopy import visual, event
from ImageStimNumpyuByte import ImageStimNumpyuByte
from TaskControl import TaskControl
import numpy as np


class CategoryMaskTask(TaskControl):
    
    def __init__(self):
        TaskControl.__init__(self)
        self.taskType = 'noise' # gratings or noise
        self.interTrialFrames = 120
        self.maxTrialFrames = 240
        self.rewardDistance = 6 # degrees to move stim for reward
        
        # mouse can move target stimulus with wheel for early training
        # varying stimulus duration and/or masking not part of this stage
        self.moveStim = True
        self.preMoveFrames = 30 # number of frames after stimulus onset before stimulus moves
        
        self.preStimMaskFrames = 0 # duration of forward mask
        self.maskToStimFrames = 0 # frames between forward mask offset and target stimulus
        self.stimToMaskFrames = 0 # frames between target stimulus offset and backward mask
        self.postStimMaskFrames = 0 # duration of backward mask
        
        self.stimSize = 10 # degrees
        self.gratingsSF = [0.02,0.08] # cycles/deg
        self.gratingsOri = [0,90]
        self.noiseSquareSize = [0.1,0.3,1,3] # degrees

    def checkParameterValues(self):
        pass # todo
    
    def run(self):
        self.checkParameterValues()
        
        self.prepareRun()
        
        try:
            # create stim
            stimSizePix = int(self.stimSize*self.pixelsPerDeg)
            if self.taskType=='gratings':
                categoryParams = (self.gratingsSF,self.gratingsOri)
                stim = visual.GratingStim(win=self._win,
                                          units='pix',
                                          mask='none',
                                          tex='sin',
                                          size=stimSizePix, 
                                          pos=(0,0))  
            elif self.taskType=='noise':
                categoryParams = (self.noiseSquareSize,)*2
                self._stimImage = np.zeros((stimSizePix,)*2,dtype=np.uint8)
                stim  = ImageStimNumpyuByte(self._win,image=self._stimImage,size=self._stimImage.shape[::-1],pos=(0,0))
            
            assert(len(categoryParams)==2)
            assert(len(categoryParams[0])==len(categoryParams[1]))
            
            # create list of trial parameter combinations
            trialTypes = [list(i) for i in itertools.product(categoryParams[0],categoryParams[1])]
            # add reward side to off diagonal elements of category parameter matrix
            # and remove combinations on diagonal
            for params in trialTypes[:]: # looping through shallow copy of trialTypes
                i,j = (categoryParams[n].index(params[n])+1 for n in (0,1))
                if i==j:
                    trialTypes.remove(params)
                elif i/j>1:
                    params.append(1)
                else:
                    params.append(-1)
            
            # run session
            sessionFrame = 0
            trialFrame = 0
            self.trialStartFrame = []
            self.trialRewardSide = []
            self.trialRewarded = []
            self.trialSF = []
            self.trialOri = []
            self.trialSquareSize = []
            
            while True: # each loop is a frame flip
                # start new trial
                if trialFrame==0:
                    stimPos = 0
                    stim.pos = (stimPos,0)
                    trialTypeIndex = len(self.trialStartFrame) % len(trialTypes)
                    if trialTypeIndex==0:
                        random.shuffle(trialTypes)
                    if self.taskType=='gratings':
                        sf,ori,rewardSide = trialTypes[trialTypeIndex]
                        self.trialSF.append(sf)
                        self.trialOri.append(ori)
                        stim.sf = sf/self.pixelsPerDeg
                        stim.ori = ori
                    elif self.taskType=='noise':
                        squareSize,_,rewardSide = trialTypes[trialTypeIndex]
                        self.trialSquareSize.append(squareSize)
                        self._squareSizePix = squareSize*self.pixelsPerDeg
                        self.updateStimImage(random=True)
                        stim.setReplaceImage(self._stimImage)
                    self.trialRewardSide.append(rewardSide)
                    self.trialStartFrame.append(sessionFrame)
                
                # update stimulus after intertrial gray screen period is complete
                if trialFrame > self.interTrialFrames:
                    # update stim position according to wheel encoder change
                    if self.moveStim and (trialFrame > self.interTrialFrames+self.preMoveFrames):
                        stimPos += self.translateEndoderChange()
                        stim.pos = (stimPos,0)
                    
                    # forward mask
                    if trialFrame <= self.interTrialFrames+self.preStimMaskFrames:
                        self.showMask()
                    
                    # target
                    elif trialFrame <= self.interTrialFrames+self.preStimMaskFrames+self.maskToStimFrames:
                        self.showTarget()
                        
                    # backward mask
                    elif trialFrame <= self.interTrialFrames+self.preStimMaskFrames+self.maskToStimFrames+targetFrames:
                        self.showMask()
                    
                    stim.draw()
                self.visStimFlip()
                trialFrame += 1
                sessionFrame += 1
                
                # end trial if reward earned or at max trial duration
                if ((rewardSide<0 and stimPos<-self.rewardDistance*self.pixelsPerDeg) or
                    (rewardSide>0 and stimPos>self.rewardDistance*self.pixelsPerDeg)):
                       self.deliverReward()
                       self.trialRewarded.append(True)
                       trialFrame = 0
                elif trialFrame==self.maxTrialFrames:
                    self.trialRewarded.append(False)
                    trialFrame = 0
                
                # check for keyboard events to end session
                if len(event.getKeys()) > 0:                  
                    event.clearEvents()
                    break
                
        except:
            traceback.print_exc()
            
        finally:
            self.completeRun()
            
        
    def updateStimImage(self):
        numSquares = int(round(self._stimImage.shape[0]/self._squareSizePix))
        self._stimImage = self._numpyRandom.randint(0,2,(numSquares,)*2).astype(np.uint8)*255
        self._stimImage = np.repeat(self._stimImage,self._squareSizePix,0)
        self._stimImage = np.repeat(self._stimImage,self._squareSizePix,1)


if __name__ == "__main__":
    pass