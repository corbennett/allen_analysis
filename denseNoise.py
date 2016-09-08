
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 12:36:46 2016

@author: corbettb
"""

from psychopy import visual, event
import numpy as np
from VisStimControl import VisStimControl
from ImageStimNumpyuByte import ImageStimNumpyuByte


class denseNoise(VisStimControl):
    
    def __init__(self):
        VisStimControl.__init__(self)        
        self.boxSize = 10.0                          #degrees
        self.prestimtime = 60                       #number of gray frames before boxes begin
        self.stimDuration = 3                       #number of frames to show each noise stimulus
        self.gridBoundaries = [10, -30, 0.5*self.fieldWidth/self.pixelsPerDeg, 90]    #Screen coordinates define presentation window (degrees): x1 y1 for lower left, x2 y2 for upper right corner
        self.setDefaultImageSizeAndPosition()
        
     
    def run(self):
        # clear history        
        self.noiseStartFrames = []
        self.boxSize = float(self.boxSize)
        #convert grid and box to pix
        self._gridBoundariesPix = np.array(self.gridBoundaries)*self.pixelsPerDeg
        self._gridExtentPix = [(self._gridBoundariesPix[2] - self._gridBoundariesPix[0] + 1),(self._gridBoundariesPix[3] - self._gridBoundariesPix[1] + 1)]
        self.imageSize = np.round(self._gridExtentPix)
        self.imagePosition[0] = np.round(self._gridBoundariesPix[0] + self._gridExtentPix[0]/2)
        self.imagePosition[1] = np.round(self._gridBoundariesPix[1] + self._gridExtentPix[1]/2)        
        
        
        #setup monitor
        self.prepareRun()

        #make noise stimulus
        noiseImage = self.makeNoise()
        noiseStim = ImageStimNumpyuByte(self._win,image=noiseImage, size=self.imageSize,pos=self.imagePosition)
        

        ntotalframes = 0
        #show pre stim gray
        while ntotalframes < self.prestimtime:
            self.visStimFlip()
            ntotalframes += 1
        
        #begin noise stimulus
        while True:

            ntrialframes = 0
            while (ntrialframes < self.stimDuration):
                if ntrialframes == 0:
                    self.noiseStartFrames.append(ntotalframes)
                    
                noiseStim.draw()
                self.visStimFlip()
                ntrialframes += 1
                ntotalframes += 1
            
            #reset noise stim for next trial
            noiseImage = self.makeNoise()
            noiseStim.setReplaceImage(noiseImage)
            
            if len(event.getKeys())>0:
                event.clearEvents()
                break
 

        self.totalFrames = ntotalframes
        self.completeRun()
        
    
    def makeNoise(self):
        gridXlength = np.ceil((self.gridBoundaries[2] - self.gridBoundaries[0])/self.boxSize)
        gridYlength = np.ceil((self.gridBoundaries[3] - self.gridBoundaries[1])/self.boxSize)
        noiseImage = self._numpyRandom.randint(0,2, [gridYlength, gridXlength]).astype(np.uint8)*255
        noiseImage = np.repeat(noiseImage, self.boxSize, 0)
        noiseImage = np.repeat(noiseImage, self.boxSize, 1)
        return noiseImage

        
if __name__ == "__main__":
    pass