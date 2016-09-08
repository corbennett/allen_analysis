# -*- coding: utf-8 -*-
"""
Created on Thu May  5 00:02:20 2016

@author: samg
"""

import math, random, cv2
import numpy as np
from psychopy import event
from VisStimControl import VisStimControl
from ImageStimNumpyuByte import ImageStimNumpyuByte


class VirtualTunnel(VisStimControl):
    
    def __init__(self):
        VisStimControl.__init__(self)
        self.setDefaultImageSizeAndPosition()
        self.domeRadius = 30.0 # cm; mouse assumed to be at center of dome
        self.domeBehind = 15.0 #cm; distance from mouse eye to back of dome
        self.domeUnder = 15.0 # cm; distance from mouse eye to bottom of dome
        self.tunnelRadius = 15.0 # cm; actual radius adjusted to fit whole number of squares
        self.squareSize = 2.0 # cm
        self.drawCurvature = False # determines whether warped square edges are curved
        self.bckgndSpeed = 'rotary encoder' # cm/s or 'rotary encoder'
        self.minRunSpeed = 10.0 # cm/s; min speed to move tunnel
        self.patchInterval = [240,300] # [min,max] frames, or set None for no patch trials
        self.patchCenterDist = [30.0] # cm; from back of tunnel
        self.patchCenterPhi = [22.5] # degrees
        self.patchSize = [6.0] # cm
        self.patchSpeed = [-5.0,5.0] # cm/s; positive forward
        self.patchMoveDur = [120] # frames

    def checkParameterValues(self):
        for param in ('patchCenterDist','patchCenterPhi','patchSize','patchSpeed','patchMoveDur'):
            if isinstance(getattr(self,param),(int,float)):
                setattr(self,param,[getattr(self,param)]) 
        if isinstance(self.bckgndSpeed,str) and self.bckgndSpeed!='rotary encoder':
            raise ValueError('background speed must be "rotary encoder" or float cm/s')
        if self.patchInterval is not None:
            if (not isinstance(self.patchInterval,list) or len(self.patchInterval)!=2
                or self.patchInterval[0]>self.patchInterval[1]
                or not isinstance(self.patchInterval[0],int) or not isinstance(self.patchInterval[1],int)):
                raise ValueError('patchInterval must be a list of [min,max] duration in frames')
            if any((True for dist in self.patchCenterDist if not 0<dist<self.domeRadius+self.domeBehind)):
                raise ValueError('patchCenterDist must be between 0 and tunnel length')
            if any((True for phi in self.patchCenterPhi if not 0<=phi<360)):
                raise ValueError('patchCenterPhi must be between 0 and 360 degrees')
            for size in self.patchSize:
                if size>2*math.pi*self.tunnelRadius:
                    raise ValueError('patchSize must be smaller than tunnel circumference')
                elif size>self.domeRadius+self.domeBehind:
                    raise ValueError('patchSize must be smaller than tunnel length')
                elif size<self.squareSize:
                    raise ValueError('patchSize must be larger than squareSize')
            if self.patchInterval[0]<max(self.patchMoveDur):
                raise ValueError('min patchInterval must be > patchMovDur')
        
    def run(self):
        self.checkParameterValues()
        
        self.prepareRun()
        
        # create tunnel stim
        # bckgndPhi only covers 0 to pi/2 to take advantage of symmetry
        # fullPhi used to align patches to bckgndPhi
        self._tunnelLength = self.domeRadius+self.domeBehind
        bckgndOffset = self.squareSize
        bckgndDist = np.concatenate((np.arange(self._tunnelLength,0,-self.squareSize),[0]))
        self.adjustedTunnelRadius = round(2*math.pi*self.tunnelRadius/self.squareSize)*self.squareSize/(2*math.pi)
        bckgndPhi = np.linspace(0,math.pi/2,round(math.pi/2*self.adjustedTunnelRadius/self.squareSize))
        fullPhi = np.concatenate((np.arange(-2*math.pi,0,bckgndPhi[1]),np.arange(0,4*math.pi,bckgndPhi[1])))
        bckgndColors = self._numpyRandom.randint(0,2,(bckgndDist.size+1,4*(bckgndPhi.size-1)))*255
        self._tunnelImage = np.zeros((self.imageSize[1],self.imageSize[0]),dtype=np.uint8)
        self.updateTunnelImage(bckgndDist,bckgndPhi,bckgndColors,True)
        tunnelStim = ImageStimNumpyuByte(self._win,image=self._tunnelImage,size=self.imageSize,pos=self.imagePosition)
        
        # run
        frame = 0
        patchTrial = -1
        if self.patchInterval is not None:
            patchInterval = random.randint(self.patchInterval[0],self.patchInterval[1])
            patchFrame = 0
            self.patchStartFrame = []
            self.trialPatchCenterDist = []
            self.trialPatchCenterPhi = []
            self.trialPatchSize = []
            self.trialPatchSpeed = []
            self.trialPatchMoveDur = []
        if self.bckgndSpeed=='rotary encoder':
            self.rotaryEncoderRadians = []
        while True:
            if self.bckgndSpeed=='rotary encoder':
                encoderAngle = self.readRotaryEncoder()*2*math.pi/5
                self.rotaryEncoderRadians.append(np.arctan2(np.mean(np.sin(encoderAngle)),np.mean(np.cos(encoderAngle))))
                if frame>1:
                    angleChange = self.rotaryEncoderRadians[-1]-self.rotaryEncoderRadians[-2]
                    if angleChange<-np.pi:
                        angleChange += 2*math.pi
                    elif angleChange>np.pi:
                        angleChange -= 2*math.pi
                    # cm/frame = angle/frame * rotation/2*pi * cm/rotation * arbitrary gain
                    #          = angleChange * rotation/2*pi * 2*pi*wheelRadius * arbitrary gain
                    #          = angleChange * wheelRadius * arbitrary gain
                    bckgndMov = self.wheelRotDir*angleChange*self.wheelRadius*self.wheelSpeedGain
                    if bckgndMov<self.minRunSpeed/self.frameRate:
                        bckgndMov = 0
                else:
                    bckgndMov = 0
            else:
                bckgndMov = self.bckgndSpeed/self.frameRate
            if bckgndMov>0 or (patchTrial>-1 and patchFrame<self.patchMoveDur):
                if bckgndMov>0:
                    bckgndOffset += bckgndMov
                    if bckgndOffset>self.squareSize:
                        newSquares = bckgndOffset//self.squareSize
                        bckgndColors[newSquares:,:] = bckgndColors[:-newSquares,:]
                        bckgndColors[0:newSquares,:] = self._numpyRandom.randint(0,2,(newSquares,bckgndColors.shape[1]))*255
                        bckgndOffset %= self.squareSize
                        if bckgndOffset==0:
                            bckgndOffset = self.squareSize
                    bckgndDist = np.concatenate(([self._tunnelLength],np.arange(self._tunnelLength-bckgndOffset,0,-self.squareSize),[0]))
                self._tunnelImage[:,:] = 0
                self.updateTunnelImage(bckgndDist,bckgndPhi,bckgndColors,True)
            if self.patchInterval is not None and patchFrame==patchInterval-1:
                patchTrial += 1
                patchInterval = random.randint(self.patchInterval[0],self.patchInterval[1])
                patchFrame = -1
                self.patchStartFrame.append(frame+1)
                self.trialPatchCenterDist.append(random.choice(self.patchCenterDist))
                self.trialPatchCenterPhi.append(random.choice(self.patchCenterPhi))
                self.trialPatchSize.append(random.choice(self.patchSize))
                self.trialPatchSpeed.append(random.choice(self.patchSpeed))
                self.trialPatchMoveDur.append(random.choice(self.patchMoveDur))
                patchSquaresPerSide = self.trialPatchSize[-1]/self.squareSize
                iminus,iplus = patchSquaresPerSide//2,round(patchSquaresPerSide/2)
                jminus,jplus = iminus,iplus
                i = np.argmin(abs(bckgndDist-self.trialPatchCenterDist[-1]))
                if i-iminus<0:
                    iminus = i
                elif i+iplus>bckgndDist.size:
                    iplus = bckgndDist.size-1-i
                patchDist = np.copy(bckgndDist[i-iminus:i+iplus+1])
                j = np.argmin(abs(fullPhi-self.trialPatchCenterPhi[-1]*math.pi/180))
                patchPhi = fullPhi[j-jminus:j+jplus+1]
                patchColors = np.tile(bckgndColors,(1,3))[i-iminus:i+iplus,j-jminus:j+jplus]
            elif patchTrial>-1 and patchFrame<self.trialPatchMoveDur[-1] and patchDist is not None:
                patchDist += self.trialPatchSpeed[-1]/self.frameRate
                inTunnel = patchDist>=0
                if any(inTunnel):
                    if not all(inTunnel):
                        patchDist = np.concatenate((patchDist[inTunnel],[0]))
                        patchColors = patchColors[:patchDist.size-1,:]
                    inTunnel = patchDist<=self._tunnelLength
                    if any(inTunnel):
                        if not all(inTunnel):
                            patchDist = np.concatenate(([self._tunnelLength],patchDist[inTunnel]))
                            patchColors = patchColors[-patchDist.size+1:,:]
                        self.updateTunnelImage(patchDist,patchPhi,patchColors,False)
                    else:
                        patchDist = None
                else:
                    patchDist = None
            tunnelStim.setReplaceImage(self._tunnelImage)
            tunnelStim.draw()
            self.visStimFlip()
            frame += 1
            if self.patchInterval is not None:
                patchFrame += 1
            if len(event.getKeys())>0:                                     
                event.clearEvents()
                break
        self.completeRun()
     
    def updateTunnelImage(self,dist,phi,colors,bckgnd):
        # calculate angle of tunnel radius at squareSize intervals from back of tunnel
        theta = np.arctan(self.adjustedTunnelRadius/dist)
        # get virtual tunnel radii in pixels
        radii = np.round(theta*self.imageSize[0]/math.pi)
        if dist[-1]==0:
            radii = np.concatenate((radii,[self.imageSize[0]/2]))
        # use angle of square edges along tunnel cicumference (phi) to calculate corresponding screen coordinates
        xvertices = np.round(radii[:,np.newaxis]*np.cos(phi))
        yvertices = np.round(radii[:,np.newaxis]*np.sin(phi))
        # fill between vertices of inner and outer radii for each white square
        if not self.drawCurvature:
            vertices = np.zeros((4,2),dtype=np.int32)
        offset = [self.imageSize[0]/2-self.imagePosition[0],self.imageSize[1]/2-self.imagePosition[1]]
        for j in xrange(phi.size-1):
            if self.drawCurvature:
                outerX = None
            for i in xrange(radii.size-1):
                if radii[i+1]>radii[i]:
                    if self.drawCurvature:
                        if outerX is None:
                            innerX,innerY = getCurveVertices(xvertices[i,j:j+2],yvertices[i,j:j+2],radii[i])
                        else:
                            innerX,innerY = outerX,outerY
                        outerX,outerY = getCurveVertices(xvertices[i+1,j:j+2],yvertices[i+1,j:j+2],radii[i+1])
                        ind = innerX.size
                        vertices = np.zeros((ind+outerX.size,2),dtype=np.int32)
                        vertices[:ind,0] = innerX
                        vertices[:ind,1] = innerY
                        vertices[:ind-1:-1,0] = outerX
                        vertices[:ind-1:-1,1] = outerY
                    else:
                        vertices[:2,0] = xvertices[i,j:j+2]
                        vertices[:2,1] = yvertices[i,j:j+2]
                        vertices[:1:-1,0] = xvertices[i+1,j:j+2]
                        vertices[:1:-1,1] = yvertices[i+1,j:j+2]
                    if colors[i,j]>0 or not bckgnd:
                        cv2.fillConvexPoly(self._tunnelImage,vertices+offset,colors[i,j])
                    if bckgnd:
                        # use same (flipped) vertices but unique colors for other three quadrants
                        if colors[i,2*(phi.size-1)-1-j]>0:
                            cv2.fillConvexPoly(self._tunnelImage,vertices*np.array((-1,1),dtype=np.int32)+offset,255)
                        if colors[i,j+(phi.size-1)*2]>0:
                            cv2.fillConvexPoly(self._tunnelImage,-vertices+offset,255)
                        if colors[i,4*(phi.size-1)-1-j]>0:
                            cv2.fillConvexPoly(self._tunnelImage,vertices*np.array((1,-1),dtype=np.int32)+offset,255)
  
  
def getCurveVertices(xvertices,yvertices,radius):
    if xvertices[0]<xvertices[1]:
        x = np.arange(xvertices[0],xvertices[1]+1)
        y = -np.round(np.sqrt(radius**2-x**2))
        y[[0,-1]] = yvertices
    elif xvertices[0]>xvertices[1]:
        x = np.arange(xvertices[1],xvertices[0]+1)
        y = np.round(np.sqrt(radius**2-x**2))
        y[[-1,0]] = yvertices
    else:
        x = xvertices
        y = yvertices
    return x,y

                
if __name__=="__main__":
    pass