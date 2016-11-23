"""
Created on Mon Jan 18 13:36:05 2016

@author: corbettb

"""

import random, itertools
from psychopy import visual, event
from VisStimControl import VisStimControl
import numpy as np

class makeGratings(VisStimControl):
    
    def __init__(self):
        VisStimControl.__init__(self)
        self.experimentType = 'stf'
        self.gratingType = 'sqr' # or sin
        self.phase = [0]
        self.posx = [80]                                     #x position of center of grating patch, in degrees
        self.posy = [0]                                     #y position of center of grating patch, in degrees
        self.size = [self.fieldWidth/self.pixelsPerDeg]     #size of grating patch in degrees
        self.numCycles = 15                                #number of times to cycle through parameters
        self.shuffle = True
        self.closedloop = False
        self.preTime = 30                                   #num frames of gray before gratings
        self.postTime = 30                                  #num frames of gray after gratings
        self.stimTime = 120                                 #num frames of gratings
        self.mask = 'none'                                  #determines mask shape for grating patches, for instance 'gauss'
        self.ori = [0, 90]        
        self.tf = [0.5, 1, 2, 4, 8]                         #cycles per second
        self.sf = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]            #cycLes per degree
        self.contrast = [1]
        self.interleavedGrayScreen = True
        
       
    def checkParameterValues(self):
        for param in ['ori', 'tf', 'sf', 'contrast', 'phase', 'size', 'posx', 'posy']:
            if not isinstance(getattr(self,param),(tuple,list)):
                setattr(self,param,(getattr(self,param),)) 
     
     
    def run(self):

            
        #make sure all params are lists 
        self.checkParameterValues()
       
        #setup monitor
        self.prepareRun()
        self._win.setRecordFrameIntervals(True)
        
        #setup grating object         
        self._grating = visual.GratingStim(win=self._win, mask=self.mask, tex=self.gratingType, size=self.pixelsPerDeg*self.size[0], 
                                           pos=[self.pixelsPerDeg*self.posx[0],self.pixelsPerDeg*self.posy[0]], sf=3, units='pix')
        
        #initialize stim parameters (shuffle if necessary)
        self.set_params()
        self.numSweeps = len(self._parameterCombos)
        
        #run stimulus
        ncycles = 0  
        self.stimulusHistory = []
        stimHist = []
        self.stimStartFrames = []
        breakFlag = False
        nTotalFrames = 0
        while (ncycles<self.numCycles) and not(breakFlag):
            print 'starting cycle:', ncycles + 1            
            nsweeps = 0
            while (nsweeps < self.numSweeps) and not(breakFlag): 
          
                self.set_grating(nsweeps)
                nframes = 0
                
                while nframes < self.preTime + self.laserPreFrames + self.stimTime + self.laserPostFrames + self.postTime:
                    if nframes == self.preTime:
                        self.setLaserOn(self._parameterCombos[nsweeps][-1])
                    if nframes == self.preTime + self.laserPreFrames:
                        self.stimStartFrames.append(nTotalFrames)
                    if (self.preTime + self.laserPreFrames <= nframes < self.preTime + self.laserPreFrames + self.stimTime):
                        self._grating.draw()
                        phase_advance = self._grating.tf/self.frameRate #fractions of a cycle to advance every refresh
                        self._grating.setPhase(phase_advance, '+')
                    if nframes == self.preTime + self.laserPreFrames + self.stimTime + self.laserPostFrames:
                        self.setLaserOff()
                    
                    self.visStimFlip()
                    
                    nframes += 1
                    nTotalFrames += 1
                    
                    if len(event.getKeys())>0: 
                        breakFlag = True
                        event.clearEvents()                        
                        break
                
                #save stimulus parameters for this sweep
                stimHist.append(self._parameterCombos[nsweeps])
                nsweeps += 1

            ncycles += 1
            self.set_params()
            
        self.stimulusHistory = dict(zip(self.stimulusParams, list(np.array(stimHist).T)))
        self.stimulusHistory['sf'] *= self.pixelsPerDeg
        self.stimulusHistory['size'] /= self.pixelsPerDeg
        self.completeRun()
        
    def set_params(self):
        self.stimulusParams = ['ori', 'tf', 'sf', 'contrast', 'phase', 'size', 'laserPower'] #the first six of these should be the grating parameters
        laserPwr = self.laserPower if self.laserRandom else [self.laserPower[0]]
        self._parameterCombos = list(itertools.product(self.ori, self.tf, [sf/self.pixelsPerDeg for sf in self.sf], self.contrast, self.phase, [size * self.pixelsPerDeg for size in self.size], laserPwr))
        if self.interleavedGrayScreen:
            grayCombo = list(self._parameterCombos[-1])
            contrastIndex = [index for index,p in enumerate(self.stimulusParams) if p=='contrast'][0]
            grayCombo[contrastIndex] = 0
            self._parameterCombos.append(tuple(grayCombo))
        if self.shuffle:
            random.shuffle(self._parameterCombos)
        if len(self.laserPower)>1 and not self.laserRandom:
            self._parameterCombos *= len(self.laserPower)
            self._parameterCombos = [list(params) for params in self._parameterCombos]
        self.setTrialLaserPower(self._parameterCombos)
        print self._parameterCombos
        
    def set_grating(self, nsweeps):
        paramVector = self._parameterCombos[nsweeps]
        gratingParams = ['ori', 'tf', 'sf', 'contrast', 'phase', 'size']
        for params in gratingParams:
            paramIndex =  [ind for ind,p in enumerate(self.stimulusParams) if p==params][0]           
            setattr(self._grating, params, paramVector[paramIndex])
    
    
    def set_experiment_type(self, expType):
        if expType == 'stf':
            self.ori = [0, 90]        
            self.tf = [0.5, 1, 2, 4, 8]                     #cycles per second
            self.sf = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]        #cycLes per degree
            self.contrast = [1]
        
        elif expType == 'ori':
            self.ori = range(0, 360, 45)        
            self.tf = [1, 4]                                   #cycles per second
            self.sf = [0.02, 0.16]                                #cycLes per degree
            self.contrast = [1]
        else:
            print 'Did not recognize requested experiment type'
        
if __name__ == "__main__":
    pass