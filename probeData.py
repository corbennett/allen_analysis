# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:19:20 2016

@author: SVC_CCG
"""

from __future__ import division
import datetime, h5py, itertools, json, math, ntpath, os, shelve, shutil
import numpy as np
import scipy.ndimage.filters
import scipy.optimize
import scipy.signal
import scipy.stats
from matplotlib import pyplot as plt 
import matplotlib.gridspec as gridspec
from PyQt4 import QtGui


dataDir = 'C:\Users\SVC_CCG\Desktop\Data'

class probeData():
    
    def __init__(self):
        self.recording = 0
        self.TTLChannelLabels = ['VisstimOn', 'CamExposing', 'CamSaving', 'OrangeLaserShutter']
        self.channelMapFile = r'C:\Users\SVC_CCG\Documents\PythonScripts\imec_channel_map.prb'
        self.wheelChannel = 134
        self.diodeChannel = 135
        self.sampleRate = 30000     


    def loadKwd(self, filePath):
                
        f = h5py.File(filePath, 'r')
        datDict = {}
        datDict['info'] = f['recordings'][str(self.recording)].attrs
        
        datDict['data'] = f['recordings'][str(self.recording)]['data']
        datDict['gains'] = f['recordings'][str(self.recording)]['application_data']['channel_bit_volts'][:]
        
        datDict['sampleRate'] = datDict['info']['sample_rate']
        datDict['startTime'] = datDict['info']['start_time']
        
        datDict['firstAnalogSample'] = f['recordings'][str(self.recording)]['application_data']['timestamps'][0][0]
        
        return datDict
        
        
    def loadExperiment(self, loadRunningData=False):
        self.kwdFileList, nsamps = getKwdInfo()
        filelist = self.kwdFileList
        filePaths = [os.path.dirname(f) for f in filelist]        
        
        self.d = []
        for index, f in enumerate(filelist):
            ff = os.path.basename(os.path.dirname(f))
            ff = ff.split('_')[-1]  
            datDict = self.loadKwd(f)
            datDict['protocolName'] = ff
            datDict['numSamples'] = nsamps[index]
            self.d.append(datDict)
                    
            
        self.getSingleUnits(fileDir=os.path.dirname(filePaths[0]))
        self.mapChannels()
        self.visstimData = {}
        self.behaviorData = {}
        self.TTL = {}
        for pro, proPath in enumerate(filePaths):
            files = os.listdir(proPath)
            
            visStimFound = False
            eyeDataFound = False
            self.behaviorData[str(pro)] = {}
            for f in files:
                if 'VisStim' in f:
                    self.getVisStimData(os.path.join(proPath, f), protocol=pro)
                    visStimFound = True
                    continue
            
                #load eye tracking data
                if 'MouseEyeTracker' in f:  
                    self.getEyeTrackData(os.path.join(proPath, f), protocol=pro)
                    eyeDataFound = True
                    continue
                
            ttlFile = [f for f in files if f.endswith('kwe')][0]             
            self.getTTLData(filePath=os.path.join(proPath, ttlFile), protocol=pro)
            
            if loadRunningData:
                self.behaviorData[str(pro)]['running'] = self.decodeWheel(self.d[pro]['data'][::500, self.wheelChannel]*self.d[pro]['gains'][self.wheelChannel])
            
            if not visStimFound:
                print('No vis stim data found for ' + os.path.basename(proPath))
            if not eyeDataFound:
                print('No eye tracking data found for ' + os.path.basename(proPath))
            
        
    def getTTLData(self, filePath=None, protocol=0):
        
        if filePath is None:
            ttlFileDir = self.filePath[:self.filePath.rfind('/')]
            filelist = os.listdir(ttlFileDir)
            filePath = ttlFileDir + '/' + [f for f in filelist if f.endswith('kwe')][0]
        
        
        f = h5py.File(filePath, 'r')
        recordingID = f['event_types']['TTL']['events']['recording'][:]
        eventChannels = f['event_types']['TTL']['events']['user_data']['event_channels'][recordingID==self.recording]        
        edges = f['event_types']['TTL']['events']['user_data']['eventID'][recordingID==self.recording]
        timeSamples = f['event_types']['TTL']['events']['time_samples'][recordingID==self.recording]
        
        self.TTLChannels = np.unique(eventChannels)        
        self.TTL[str(protocol)] = {}
        for chan in self.TTLChannels:
            eventsForChan = np.where(eventChannels == chan)
            self.TTL[str(protocol)][self.TTLChannelLabels[chan]] = {}
            self.TTL[str(protocol)][self.TTLChannelLabels[chan]]['rising'] = timeSamples[np.intersect1d(eventsForChan, np.where(edges == 1))] - self.d[protocol]['firstAnalogSample']
            self.TTL[str(protocol)][self.TTLChannelLabels[chan]]['falling'] = timeSamples[np.intersect1d(eventsForChan, np.where(edges ==0))] - self.d[protocol]['firstAnalogSample']
        
        if str(protocol) in self.visstimData.keys():
            if not hasattr(self, 'frameSamples'):
                self.alignFramesToDiode(protocol=protocol)
    
    
    def getVisStimData(self, filePath=None, protocol=0):
        if filePath is None:        
            filePath = getFile()
        
        dataFile = h5py.File(filePath)        
        self.visstimData[str(protocol)] = {}
        for params in dataFile.keys():
            if dataFile[params].size > 1:
                self.visstimData[str(protocol)][params] = dataFile[params][:]
            else:
                self.visstimData[str(protocol)][params] = dataFile[params][()]   
    
    
    def getEyeTrackData(self, filePath=None, protocol=0):
        if filePath is None:        
            filePath = getFile()
        
        dataFile = h5py.File(filePath)
        frameTimes = dataFile['frameTimes'][:]
        frameInt = 1/60.0
        nRealFrames = round(frameTimes[-1]/frameInt)+1
        frameInd = np.round(frameTimes/frameInt).astype(int)
        
        eyeDict = {}
        firstFrameIndex = np.where(self.TTL[str(protocol)]['CamExposing']['rising'][:] < self.TTL[str(protocol)]['CamSaving']['rising'][0])[0][-1]
        eyeTime = self.TTL[str(protocol)]['CamExposing']['rising'][firstFrameIndex:firstFrameIndex + nRealFrames] - self.d[protocol]['startTime']
        eyeDict['samples'] = eyeTime[eyeTime<self.d[protocol]['numSamples']]         
                
        for param in ('pupilArea','pupilX','pupilY'):
            eyeDict[param] = np.full(nRealFrames,np.nan)       
            eyeDict[param][frameInd] = dataFile[param][:]
            eyeDict[param] = eyeDict[param][0:eyeDict['samples'].size]
        
        self.behaviorData[str(protocol)]['eyeTracking'] = eyeDict
    
    
    def alignFramesToDiode(self, frameSampleAdjustment = None, plot = False, protocol=0):
        if frameSampleAdjustment is None:
            frameSampleAdjustment = np.round((4.5/60.0) * 30000) 
        self.visstimData[str(protocol)]['frameSamples'] =  (self.TTL[str(protocol)]['VisstimOn']['falling'] + frameSampleAdjustment).astype(int)
        self.visstimData[str(protocol)]['frameSamples'] = (self.TTL[str(protocol)]['VisstimOn']['falling'] + frameSampleAdjustment).astype(int)
        self._frameSampleAdjustment = frameSampleAdjustment
        
        if plot:
            plt.figure()
            plt.plot(self.data[str(protocol)]['data'][:self.visstimData[str(protocol)]['frameSamples'][10], self.diodeChannel])
            plt.plot(self.visstimData[str(protocol)]['frameSamples'][:10], np.ones(10) * np.max(self.data[str(protocol)]['data'][:self.visstimData[str(protocol)]['frameSamples'][10], self.diodeChannel]), 'go')
            
            plt.figure()
            plt.plot(self.data[str(protocol)]['data'][self.visstimData[str(protocol)]['frameSamples'][-10]:, self.diodeChannel])
            plt.plot(self.visstimData[str(protocol)]['frameSamples'][-10:] - self.visstimData[str(protocol)]['frameSamples'][-10], np.ones(10) * np.max(self.data[str(protocol)]['data'][self.visstimData[str(protocol)]['frameSamples'][-10]:, self.diodeChannel]), 'go')
    
    
    def mapChannels(self):
        f = open(self.channelMapFile, 'r') 
        fdict = json.load(f)
        self.channelMapping = np.array(fdict['0']['mapping'])
        self.channelMapping = self.channelMapping[np.where(self.channelMapping > 0)] - 1
    
    
    def decodeWheel(self, wheelData, kernelLength = 0.5, wheelSampleRate = 60.0):
    
        sampleRate = wheelSampleRate
        wheelData = wheelData - np.min(wheelData)
        wheelData = 2*np.pi*wheelData/np.max(wheelData)
          
        smoothFactor = sampleRate/60.0       
        angularWheelData = np.arctan2(np.sin(wheelData), np.cos(wheelData))
        angularWheelData = np.convolve(angularWheelData, np.ones(smoothFactor), 'same')/smoothFactor
        
        artifactThreshold = (100.0/sampleRate)/7.6      #reasonable bound for how far (in radians) a mouse could move in one sample point (assumes top speed of 100 cm/s)
        angularDisplacement = (np.diff(angularWheelData) + np.pi)%(2*np.pi) - np.pi
        angularDisplacement[np.abs(angularDisplacement) > artifactThreshold ] = 0
        wheelData = np.convolve(angularDisplacement, np.ones(kernelLength*sampleRate), 'same')/(kernelLength*sampleRate)
        wheelData *= 7.6*sampleRate
        wheelData = np.insert(wheelData, 0, wheelData[0])
        
        return wheelData
        
    def filterChannel(self, chan, cutoffFreqs, protocol=0):
        Wn = np.array(cutoffFreqs)/(self.sampleRate/2)        
        b,a = scipy.signal.butter(4, Wn, btype='bandpass')
        return scipy.signal.filtfilt(b, a, self.data[str(protocol)]['data'][:, chan])
        
    
    def thresholdChannel(self, chan, threshold, direction = -1, refractory = None, filterFreqs = None, protocol=0):
        
        if filterFreqs is not None:
            data = direction * self.filterChannel(chan, filterFreqs)
        else:
            data = direction * self.data[str(protocol)]['data'][:, chan]
        
        threshold = direction * threshold
        spikeTimes = np.array(np.where(data > threshold)[0])
        
        if refractory is None:
            refractory = 1.0/self.sampleRate
        
        if spikeTimes.size > 0:
            ISI = np.diff(spikeTimes)
            goodISI = np.array(np.where(ISI > refractory*self.sampleRate)[0]) + 1
            goodISI = np.insert(goodISI, 0, 0)
            spikeTimes = spikeTimes[goodISI]
        return spikeTimes
        
        
    def computeFiringRate(self, spikeTimes, kernelLength = 0.05, protocol=0):
        fr = np.zeros(self.d[protocol]['numSamples'])
        fr[spikeTimes] = 1
        fr = np.convolve(fr, np.ones(kernelLength*self.sampleRate), 'same')/(kernelLength)
        return fr    
        
   
    def triggeredAverage(self, dataToAlign, alignmentPoints, win = [0.1, 0.1], sampleRate = None):
        if sampleRate is None:
            sampleRate = self.sampleRate
        aligned = np.full([sampleRate*(win[0] + win[1]), len(alignmentPoints)], np.nan)
        for index, point in enumerate(alignmentPoints):
            try:            
                aligned[:, index] = dataToAlign[point - win[0]*sampleRate : point + win[1]*sampleRate]
            except:
                continue
        return aligned
                
            
    def parseRunning(self, runThresh = 5.0, statThresh = 1.0, trialStarts = None, trialEnds = None):
        if not hasattr(self, 'wheelData'):
            self.decodeWheel()
            
        self.runningPoints = np.where(np.abs(self.wheelData) > runThresh)[0]
        self.stationaryPoints = np.where(np.abs(self.wheelData) < statThresh)[0]
        
        if trialStarts is not None:
            self.runningTrials = []
            self.stationaryTrials = []
            for trial in range(trialStarts.size):
                trialSpeed = np.mean(self.wheelData[trialStarts[trial]:trialEnds[trial]])
                if trialSpeed >= runThresh:
                    self.runningTrials.append(trial)
                elif trialSpeed <= statThresh:
                    self.stationaryTrials.append(trial)
     
               
    def findSpikesPerTrial(self, trialStarts, trialEnds, spikes): 
        spikesPerTrial = np.zeros(trialStarts.size)
        for trialNum in range(trialStarts.size):
            spikesPerTrial[trialNum] = np.count_nonzero(np.logical_and(spikes>=trialStarts[trialNum],spikes<=trialEnds[trialNum]))
        return spikesPerTrial
        
            
    def findRF(self, units=None, sigma = 2, plot = True, minLatency = 0.03, maxLatency = 0.13, trials = None, protocol=None, fit=True, useCache=True):
        if units is None:
            units = self.units.keys()
        if not isinstance(units,list):
            units = [units]
        for u in units[:]:
            if str(u) not in self.units.keys():
                units.remove(u)
                print(str(u)+' not in units.keys()')
        if len(units)<1:
            return
        
        units, unitsYPos = self.getOrderedUnits(units)
        
        if protocol is None:
            protocol = self.getProtocolIndex('sparseNoise')
        protocol = str(protocol)
        if plot:        
            plt.figure(figsize = (7.1, 3*len(units)))
            gs = gridspec.GridSpec(len(units), 2) 
        
        minLatency *= self.sampleRate
        maxLatency *= self.sampleRate
   
        xpos = np.sort(np.unique(self.visstimData[str(protocol)]['boxPositionHistory'][:, 0]))
        ypos = np.sort(np.unique(self.visstimData[str(protocol)]['boxPositionHistory'][:, 1]))
        
        posHistory = self.visstimData[str(protocol)]['boxPositionHistory'][:]
        colorHistory = self.visstimData[str(protocol)]['boxColorHistory'][:, 0:1]        
        gridExtent = self.visstimData[str(protocol)]['gridBoundaries']
        
        if fit:
            onCenters = np.full((len(units),2),np.nan)
            offCenters = np.copy(onCenters)
        
        for uindex, unit in enumerate(units):
            if ('rf' in self.units[str(unit)].keys()) and useCache:
                gridOnSpikes = self.units[str(unit)]['rf']['on']
                gridOffSpikes = self.units[str(unit)]['rf']['off']
                gridOnSpikes_filter = self.units[str(unit)]['rf']['on_filter']
                gridOffSpikes_filter = self.units[str(unit)]['rf']['off_filter']
                xpos = self.units[str(unit)]['rf']['xpos']
                ypos = self.units[str(unit)]['rf']['ypos']
                onFit = self.units[str(unit)]['rf']['onFit']
                offFit = self.units[str(unit)]['rf']['offFit']
            else:
                self.units[str(unit)]['rf'] = {}
                spikes = self.units[str(unit)]['times'][str(protocol)]
                grid = list(itertools.product(xpos,ypos))
                gridOnSpikes = np.zeros(len(grid))
                gridOffSpikes = np.zeros(len(grid))
                for index, pos in enumerate(grid):
                    po = np.intersect1d(np.where(posHistory[:, 0] == pos[0])[0], np.where(posHistory[:, 1] == pos[1])[0])
                    if trials is not None:
                        po = np.intersect1d(po, trials)
                    
                    posOnTrials = np.intersect1d(po, np.where(colorHistory == 1)[0])
                    posOffTrials = np.intersect1d(po, np.where(colorHistory == -1)[0])
                    
                    posOnFrames = self.visstimData[str(protocol)]['stimStartFrames'][posOnTrials]
                    posOffFrames = self.visstimData[str(protocol)]['stimStartFrames'][posOffTrials]
                    
                    posOnSamples = self.visstimData[str(protocol)]['frameSamples'][posOnFrames]
                    posOffSamples = self.visstimData[str(protocol)]['frameSamples'][posOffFrames]
                    
                    for p in posOnSamples:
                        # gridOnSpikes[index] += np.intersect1d(np.arange(p+minLatency, p+maxLatency), spikes).size
                        gridOnSpikes[index] += np.count_nonzero(np.logical_and(spikes>=p+minLatency,spikes<p+maxLatency))
                    
                    for p in posOffSamples:
                        # gridOffSpikes[index] += np.intersect1d(np.arange(p+minLatency, p+maxLatency), spikes).size
                        gridOffSpikes[index] += np.count_nonzero(np.logical_and(spikes>=p+minLatency,spikes<p+maxLatency))
                    
                    gridOnSpikes[index] = gridOnSpikes[index]/float(posOnTrials.size)
                    gridOffSpikes[index] = gridOffSpikes[index]/float(posOffTrials.size)
                    
                gridOnSpikes = gridOnSpikes.reshape(xpos.size,ypos.size).T    
                gridOffSpikes = gridOffSpikes.reshape(xpos.size,ypos.size).T  
                gridOnSpikes_filter = scipy.ndimage.filters.gaussian_filter(gridOnSpikes, sigma)
                gridOffSpikes_filter = scipy.ndimage.filters.gaussian_filter(gridOffSpikes, sigma)
               
                                
                if fit:
                    fitParams = []
                    pixPerDeg = self.visstimData[str(protocol)]['pixelsPerDeg']
                    for d in (gridOnSpikes_filter,gridOffSpikes_filter):
                        # params: x0 , y0, sigX, sigY, theta, amplitude
                        data = np.copy(d)-d.min()
                        elev, azim = ypos/pixPerDeg, xpos/pixPerDeg
                        i,j = np.unravel_index(np.argmax(data),data.shape)
                        initialParams = (azim[j], elev[i], azim[1]-azim[0], elev[1]-elev[0], 0, data.max())
                        fitParams.append(fitGauss2D(azim,elev,data,initialParams))
                    onFit,offFit = fitParams
                    if onFit is not None and gridExtent[0]<onFit[0]<gridExtent[2] and gridExtent[1]<onFit[1]<gridExtent[3]:
                        onCenters[uindex,:] = onFit[0:2]
                    if offFit is not None and gridExtent[0]<offFit[0]<gridExtent[2] and gridExtent[1]<offFit[1]<gridExtent[3]:
                        offCenters[uindex,:] = offFit[0:2]
                    self.units[str(unit)]['rf']['onFit'] = onFit
                    self.units[str(unit)]['rf']['offFit'] = offFit
                                        
                self.units[str(unit)]['rf']['on'] = gridOnSpikes
                self.units[str(unit)]['rf']['off'] = gridOffSpikes
                self.units[str(unit)]['rf']['on_filter'] = gridOnSpikes_filter
                self.units[str(unit)]['rf']['off_filter'] = gridOffSpikes_filter
                self.units[str(unit)]['rf']['xpos'] = xpos
                self.units[str(unit)]['rf']['ypos'] = ypos
                
            if plot:
                maxVal = max(np.max(gridOnSpikes_filter), np.max(gridOffSpikes_filter))
                minVal = min(np.min(gridOnSpikes_filter), np.min(gridOffSpikes_filter))
                
                a1 = plt.subplot(gs[uindex, 0])
                a1.imshow(gridOnSpikes_filter, clim=[minVal, maxVal], interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]] )
                if fit and onFit is not None:
                    a1.plot(onFit[0],onFit[1],'kx',markeredgewidth=2)
                    fitX,fitY = getEllipseXY(*onFit[:-1])
                    a1.plot(fitX,fitY,'k',linewidth=2)
                    a1.set_xlim(gridExtent[[0,2]]-0.5)
                    a1.set_ylim(gridExtent[[1,3]]-0.5)
                a1.set_ylabel(str(unit)+', ypos = '+str(round(unitsYPos[uindex])), fontsize='x-small')
           
                a2 = plt.subplot(gs[uindex, 1])
                im = a2.imshow(gridOffSpikes_filter, clim=[minVal, maxVal], interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]])
                if fit and offFit is not None:
                    a2.plot(offFit[0],offFit[1],'kx',markeredgewidth=2)
                    fitX,fitY = getEllipseXY(*offFit[:-1])
                    a2.plot(fitX,fitY,'k',linewidth=2)
                    a2.set_xlim(gridExtent[[0,2]]-0.5)
                    a2.set_ylim(gridExtent[[1,3]]-0.5)
                if uindex == 0:
                    a1.set_title('on response')
                    a2.set_title('off response')
                plt.colorbar(im, ax= [a1, a2], fraction=0.05, shrink=0.5, pad=0.05)
                a2.yaxis.set_visible(False)
                
        if plot and fit and len(units)>1:
            # comparison of RF and probe position
            onIncluded = np.logical_not(np.isnan(onCenters[:,0]))
            offIncluded = np.logical_not(np.isnan(offCenters[:,0]))
            unitsYPos = np.array(unitsYPos)
            # __LinFit__ = (slope, intercept, r-value, p-value, stderror)
            if np.count_nonzero(onIncluded)>1:
                onLinFitAzim = scipy.stats.linregress(unitsYPos[onIncluded],onCenters[onIncluded,0])
                onLinFitElev = scipy.stats.linregress(unitsYPos[onIncluded],onCenters[onIncluded,1])
            if np.count_nonzero(offIncluded)>1:
                offLinFitAzim = scipy.stats.linregress(unitsYPos[offIncluded],offCenters[offIncluded,0])
                offLinFitElev = scipy.stats.linregress(unitsYPos[offIncluded],offCenters[offIncluded,1]) 
            
            probePos = [self.units[n]['ypos'] for n in self.units.keys()]
            xlim = np.array([min(probePos)-10,max(probePos)+10])            
            
            # on azimuth
            plt.figure(facecolor='w')
            ax = plt.subplot(2,2,1)
            if np.count_nonzero(onIncluded)>1:
                ax.plot(xlim,onLinFitAzim[0]*xlim+onLinFitAzim[1],color='0.6')
                ax.text(0.5,0.95,'$\mathregular{r^2}$ = '+str(round(onLinFitAzim[2]**2,2))+', p = '+str(round(onLinFitAzim[3],2)),
                        transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',fontsize='xx-small',color='0.6')
            ax.plot(unitsYPos,onCenters[:,0],'ro')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
            ax.set_xlim(xlim)
            ax.set_ylim(gridExtent[[0,2]]+[-5,5])
            ax.set_ylabel('Azimuth',fontsize='medium')
            ax.set_title('On',fontsize='medium')
            
            # off azimuth
            ax = plt.subplot(2,2,2)
            if np.count_nonzero(offIncluded)>1:
                ax.plot(xlim,offLinFitAzim[0]*xlim+offLinFitAzim[1],color='0.6')
                ax.text(0.5,0.95,'$\mathregular{r^2}$ = '+str(round(offLinFitAzim[2]**2,2))+', p = '+str(round(offLinFitAzim[3],2)),
                        transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',fontsize='xx-small',color='0.6')
            ax.plot(unitsYPos,offCenters[:,0],'bo')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
            ax.set_xlim(xlim)
            ax.set_ylim(gridExtent[[0,2]]+[-5,5])
            ax.set_title('Off',fontsize='medium')
            
            # on elevation
            ax = plt.subplot(2,2,3)
            if np.count_nonzero(onIncluded)>1:
                ax.plot(xlim,onLinFitElev[0]*xlim+onLinFitElev[1],color='0.6')
                ax.text(0.5,0.95,'$\mathregular{r^2}$ = '+str(round(onLinFitElev[2]**2,2))+', p = '+str(round(onLinFitElev[3],2)),
                        transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',fontsize='xx-small',color='0.6')
            ax.plot(unitsYPos,onCenters[:,1],'ro')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
            ax.set_xlim(xlim)
            ax.set_ylim(gridExtent[[1,3]]+[-5,5])
            ax.set_xlabel('Probe Y Pos',fontsize='medium')
            ax.set_ylabel('Elevation',fontsize='medium')
            
            # off elevation
            ax = plt.subplot(2,2,4)
            if np.count_nonzero(offIncluded)>1:
                ax.plot(xlim,offLinFitElev[0]*xlim+offLinFitElev[1],color='0.6')
                ax.text(0.5,0.95,'$\mathregular{r^2}$ = '+str(round(offLinFitElev[2]**2,2))+', p = '+str(round(offLinFitElev[3],2)),
                        transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',fontsize='xx-small',color='0.6')
            ax.plot(unitsYPos,offCenters[:,1],'bo')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
            ax.set_xlim(xlim)
            ax.set_ylim(gridExtent[[1,3]]+[-5,5])
            ax.set_xlabel('Probe Y Pos',fontsize='medium')
    
    
    def analyzeGratings(self, units=None, trials = None, responseLatency = 0.05, plot=True, protocol=None, protocolType='stf', fit = True, useCache=True):
        
        if units is None:
            units = self.units.keys()
        if not isinstance(units,list):
            units = [units]
        for u in units[:]:
            if str(u) not in self.units.keys():
                units.remove(u)
                print(str(u)+' not in units.keys()')
        if len(units)<1:
            return
            
        units, unitsYPos = self.getOrderedUnits(units) 
            
        if protocol is None:
            if protocolType=='stf':
                label = 'gratings'
            elif protocolType=='ori':
                label = 'gratings_ori'
            protocol = self.getProtocolIndex(label)
        protocol = str(protocol)
               
        if plot:        
            plt.figure(figsize = (7.1, 3*len(units)))
            gs = gridspec.GridSpec(len(units), 3)                
        
        trialSF = self.visstimData[str(protocol)]['stimulusHistory_sf']
        trialTF = self.visstimData[str(protocol)]['stimulusHistory_tf']
        trialOri = self.visstimData[str(protocol)]['stimulusHistory_ori']
        trialContrast = self.visstimData[str(protocol)]['stimulusHistory_contrast']

        sf = np.unique(trialSF)
        tf = np.unique(trialTF)
        ori = np.unique(trialOri)
                
        responseLatency = int(responseLatency*self.sampleRate)
        
        if trials is None:
            trials = np.arange(self.visstimData[str(protocol)]['stimStartFrames'].size-1)
        
        trialStartFrame = self.visstimData[str(protocol)]['stimStartFrames'][trials]
        trialDuration = self.visstimData[str(protocol)]['stimTime']
        trialStartSamples = self.visstimData[str(protocol)]['frameSamples'][trialStartFrame]+responseLatency
        trialEndSamples = self.visstimData[str(protocol)]['frameSamples'][trialStartFrame + trialDuration]+responseLatency
        
        for uindex, unit in enumerate(units):
            if ('stf' in self.units[str(unit)].keys()) and protocolType=='stf' and useCache:
                stfMat = self.units[str(unit)]['stf']['stfMat']
                sf = self.units[str(unit)]['stf']['sf']
                tf = self.units[str(unit)]['stf']['tf']
                fitParams = self.units[str(unit)]['stf']['fitParams']
            elif ('ori' in self.units[str(unit)].keys()) and protocolType=='ori' and useCache:
                oriList = self.units[str(unit)]['ori']['tuningCurve']
            else:
                self.units[str(unit)][protocolType] = {}
                
                #spontaneous firing rate taken from interspersed gray trials
                spontRate = 0
                spontCount = 0
                
                stfMat = np.zeros([tf.size, sf.size])
                stfCountMat = np.zeros([tf.size, sf.size])
                oriList = [[] for i in range(ori.size)]
                    
                spikesPerTrial = self.findSpikesPerTrial(trialStartSamples,trialEndSamples,self.units[str(unit)]['times'][protocol])
                trialResponse = spikesPerTrial/((trialEndSamples-trialStartSamples)/self.sampleRate)
                
                
                #make STF mat for specified trials (default all trials)
                
                for trial in trials:
                    spikeRateThisTrial = trialResponse[trial]
                    
                    if trialContrast[trial] > 0:
                        sfIndex = int(np.where(sf == trialSF[trial])[0])
                        tfIndex = int(np.where(tf == trialTF[trial])[0])
                        oriIndex = int(np.where(ori == trialOri[trial])[0])
                            
                        stfMat[tfIndex, sfIndex] += spikeRateThisTrial
                        stfCountMat[tfIndex, sfIndex] += 1
                        
                        oriList[oriIndex].append(spikeRateThisTrial)
                    else:
                        spontRate += spikeRateThisTrial
                        spontCount += 1
                
                spontRate /= spontCount
                stfMat /= stfCountMat
                stfMat -= spontRate
                
                oriMean = np.zeros(len(oriList))                
                oriError = np.zeros(len(oriList))
                for oindex in range(len(oriList)):
                    oriMean[oindex] = np.mean(np.array(oriList[oindex]))
                    oriError[oindex] = np.std(np.array(oriList[oindex]))
                oriMean -= spontRate
                
                if protocolType=='ori':
                    dsi,prefDir = getDSI(oriMean+spontRate,ori)
                    osi,prefOri = getDSI(oriMean+spontRate,2*ori)
                    prefOri /= 2
                
                if fit and protocolType=='stf':
                    # params: sf0 , tf0, sigSF, sigTF, speedTuningIndex, amplitude
                    if stfMat.max()<0:
                        fitParams = None
                    else:
                        i,j = np.unravel_index(np.argmax(stfMat),stfMat.shape)
                        initialParams = (sf[j], tf[i], 1, 1, 0.5, stfMat.max())
                        fitParams = fitStfLogGauss2D(sf,tf,stfMat,initialParams)
                    self.units[str(unit)]['stf']['fitParams'] = fitParams
                    
                if protocolType=='stf':
                    self.units[str(unit)]['stf']['stfMat'] = stfMat
                    self.units[str(unit)]['stf']['sf'] = sf
                    self.units[str(unit)]['stf']['tf'] = tf
        
            if plot:
                if protocolType=='stf':                
                    xyNan = np.transpose(np.where(np.isnan(stfMat)))
                    stfMat[np.isnan(stfMat)] = 0
                   
                    a1 = plt.subplot(gs[uindex, 0])
                    plt.xlabel('sf')
                    plt.ylabel('tf')
                    plt.title(str(unit))
                    cLim = max(1,np.max(abs(stfMat)))
                    im = a1.imshow(stfMat, clim=(-cLim,cLim), cmap='bwr', origin = 'lower', interpolation='none')
                    for xypair in xyNan:    
                        a1.text(xypair[1], xypair[0], 'no trials', color='white', ha='center')
                    if fit and fitParams is not None:
                        a1.plot(np.log2(fitParams[0])-np.log2(sf[0]),np.log2(fitParams[1])-np.log2(tf[0]),'kx',markeredgewidth=2)
                        fitX,fitY = getStfContour(sf,tf,fitParams)
                        a1.plot(fitX,fitY,'k',linewidth=2)
                        a1.set_xlim([-0.5,sf.size-0.5])
                        a1.set_ylim([-0.5,tf.size-0.5])                
                    
                    a1.set_xticklabels(np.insert(sf, 0, 0))
                    a1.set_yticklabels(np.insert(tf, 0, 0))
                    plt.colorbar(im, ax=a1, fraction=0.05, shrink=0.5, pad=0.05)
                    
                    a2 = plt.subplot(gs[uindex,1])
                    values = np.mean(stfMat, axis=0)
                    error = np.std(stfMat, axis=0)
                    a2.plot(sf, values)
                    plt.fill_between(sf, values+error, values-error, alpha=0.3)
                    plt.xlabel('sf')
                    plt.ylabel('spikes')
                    plt.xticks(sf)
                    
                    a3 = plt.subplot(gs[uindex, 2])
                    values = np.mean(stfMat, axis=1)
                    error = np.std(stfMat, axis=1)
                    a3.plot(tf, values)
                    plt.fill_between(tf, values+error, values-error, alpha=0.3)
                    plt.xlabel('tf')
                    plt.ylabel('spikes')
                    plt.xticks(tf)
              
                elif protocolType=='ori':
                    a1 = plt.subplot(gs[uindex, :2])
                    plt.xlabel('ori')
                    plt.ylabel('spike rate: ' + str(unit))
                    a1.plot(ori, oriMean)
                    plt.fill_between(ori, oriMean+oriError, oriMean - oriError, alpha=0.3)
                    plt.xticks(ori)
                    
                    a2 = plt.subplot(gs[uindex, 2:], projection='polar')
                    theta = ori * (np.pi/180.0)
                    theta = np.append(theta, theta[0])
                    rho = np.append(oriMean, oriMean[0]) + spontRate
                    a2.plot(theta, rho)
                    a2.set_title('DSI = '+str(round(dsi,2))+', prefDir = '+str(round(prefDir))+'\n'+', OSI = '+str(round(osi,2))+', prefOri = '+str(round(prefOri)),fontsize='x-small')


    def analyzeSpots(self, units=None, protocol = None, plot=True, trials=None, useCache=True):
        if units is None:
            units = self.units.keys()
        if not isinstance(units,list):
            units = [units]
        for u in units[:]:
            if str(u) not in self.units.keys():
                units.remove(u)
                print(str(u)+' not in units.keys()')
        if len(units)<1:
            return
            
        units, unitsYPos = self.getOrderedUnits(units) 
        
        if protocol is None:
            protocol = self.getProtocolIndex('spots')
        protocol = str(protocol)
        
        if plot:        
            plt.figure(figsize = (11, 3*len(units)))
            gs = gridspec.GridSpec(len(units), 4)                        
        
        if trials is None:
            trials = np.arange((self.visstimData[str(protocol)]['trialStartFrame'][:-1]).size)
        
        trialStartFrames = self.visstimData[str(protocol)]['trialStartFrame'][trials]
        trialDuration = (self.visstimData[str(protocol)]['trialNumFrames'][trials]).astype(np.int)
        trialEndFrames = trialStartFrames + trialDuration
        frameSamples = self.visstimData[str(protocol)]['frameSamples']     
        trialStartSamples = frameSamples[trialStartFrames]
        trialEndSamples = frameSamples[trialEndFrames]
        
        trialPos = self.visstimData[str(protocol)]['trialSpotPos'][trials]
        trialColor = self.visstimData[str(protocol)]['trialSpotColor'][trials]
        trialSize = self.visstimData[str(protocol)]['trialSpotSize'][trials]
        trialDir = self.visstimData[str(protocol)]['trialSpotDir'][trials]
        trialSpeed = self.visstimData[str(protocol)]['trialSpotSpeed'][trials]
  
        horTrials = np.in1d(trialDir,[0,180])
        vertTrials = np.in1d(trialDir,[90,270])
#        horTrials = np.logical_or(trialDir==0, trialDir==180)
#        vertTrials = np.logical_or(trialDir==270, trialDir==90)
        
        azimuths = np.unique(trialPos[vertTrials])
        elevs = np.unique(trialPos[horTrials])
        
        
        for uindex, unit in enumerate(units):
            if ('spotResponse' in self.units[str(unit)].keys()) and useCache:
                responseDict = self.units[str(unit)]['spotResponse']['spot_responseDict']
                spotRF = responseDict['spotRF']
            else:
                self.units[str(unit)]['spotResponse'] = {}
                spikes = self.units[str(unit)]['times'][str(protocol)]
        
                # get RF         
                spikesPerTrial = self.findSpikesPerTrial(trialStartSamples, trialEndSamples, spikes)
                trialSpikeRate = spikesPerTrial/((1/self.visstimData[str(protocol)]['frameRate'])*trialDuration)

                azimuthSpikeRate = np.zeros(azimuths.size)        
                elevSpikeRate = np.zeros(elevs.size)
                azimuthTrialCount = np.zeros(azimuths.size)        
                elevTrialCount = np.zeros(elevs.size)
                for trial in range(trialPos.size):
                    if horTrials[trial]:
                        elevIndex = np.where(trialPos[trial]==elevs)[0]
                        elevSpikeRate[elevIndex] += trialSpikeRate[trial]
                        elevTrialCount[elevIndex] += 1
                    else:
                        azimuthIndex = np.where(trialPos[trial]==azimuths)[0]
                        azimuthSpikeRate[azimuthIndex] += trialSpikeRate[trial]
                        azimuthTrialCount[azimuthIndex] += 1
                
                elevSpikeRate /= elevTrialCount
                azimuthSpikeRate /= azimuthTrialCount
        
                #get spontaneous rate
                recoveryPeriod = 10
                interTrialIntervals = trialStartFrames[1:]- trialEndFrames[:-1]
                interTrialStarts = trialEndFrames[:-1] + recoveryPeriod
                interTrialEnds = trialEndFrames[:-1] + interTrialIntervals        
                itiSpikes = self.findSpikesPerTrial(frameSamples[interTrialStarts], frameSamples[interTrialEnds], spikes)
                itiRate = itiSpikes/((1/60.0)*(interTrialEnds - interTrialStarts))
                meanItiRate = itiRate.mean()
                
                #make tuning curves for various spot parameters        
                responseDict = {}        
                for param in ['trialSpotSize', 'trialSpotDir', 'trialSpotSpeed']:
                    trialValues = self.visstimData[str(protocol)][param][trials]            
                    possibleValues = np.unique(trialValues)
                    responseDict[param] = {}
                    meanResponse = np.zeros(possibleValues.size)
                    semResponse = np.zeros(possibleValues.size)
                    for ind, value in enumerate(possibleValues):
                        relevantTrials = np.where(trialValues==value)[0]
                        responseDict[param][value] = {}
                        responseDict[param][value]['trials'] = relevantTrials
                        responseDict[param][value]['response'] = np.zeros(relevantTrials.size)
                        for index, trial in enumerate(relevantTrials):
                            totalSpikes = spikesPerTrial[trial]
                            spikeRate = totalSpikes/((1/60.0)*trialDuration[trial])            
                            responseDict[param][value]['response'][index] = spikeRate
                        meanResponse[ind] = np.mean(responseDict[param][value]['response'])
                        semResponse[ind] = np.std(responseDict[param][value]['response'])/math.sqrt(relevantTrials.size)
                    spontSubtracted = meanResponse - np.mean(itiRate)
                    zscored = (spontSubtracted - np.mean(spontSubtracted))/np.std(spontSubtracted)
                    responseDict[param]['tuningCurve'] = {}
                    responseDict[param]['tuningCurve']['paramValues'] = possibleValues
                    responseDict[param]['tuningCurve']['meanResponse'] = meanResponse
                    responseDict[param]['tuningCurve']['sem'] = semResponse
                    responseDict[param]['tuningCurve']['mean_spontSubtracted'] = spontSubtracted
                    responseDict[param]['tuningCurve']['zscored'] = zscored
                    
                    
                x,y = np.meshgrid(azimuthSpikeRate-meanItiRate,elevSpikeRate-meanItiRate)
                spotRF = np.sqrt(abs(x*y))*np.sign(x+y)
                responseDict['spotRF'] = spotRF                
                self.units[str(unit)]['spotResponse']['spot_responseDict'] = responseDict
                
            if plot:   
                a1 = plt.subplot(gs[uindex, 0])            
                cLim = max(2,np.max(abs(spotRF)))
                im = a1.imshow(spotRF, clim = (-cLim,cLim), cmap='bwr', interpolation='none', origin='lower')
                plt.colorbar(im, ax=a1, fraction=0.05, pad=0.04)
                plt.title(str(unit), fontsize='x-small')
                
                for paramnum, param in enumerate(['trialSpotSize', 'trialSpotDir', 'trialSpotSpeed']):
                        a = plt.subplot(gs[uindex, paramnum+1])
                        values = responseDict[param]['tuningCurve']['mean_spontSubtracted'] 
                        error = responseDict[param]['tuningCurve']['sem'] 
                        a.plot(responseDict[param]['tuningCurve']['paramValues'], values)
                        plt.fill_between(responseDict[param]['tuningCurve']['paramValues'], values+error, values-error, alpha=0.3)
                        a.plot(responseDict[param]['tuningCurve']['paramValues'], np.zeros(values.size), 'r--')
                        plt.xlabel(param) 
                        plt.ylim(min(-0.1, np.min(values - error)), max(np.max(values + error), 0.1))
                        plt.locator_params(axis = 'y', nbins = 3)
                        a.set_xticks(responseDict[param]['tuningCurve']['paramValues'])
    
                                        
    def analyzeCheckerboard(self, units=None, protocol=None, trials=None, plot=True):
        if units is None:
            units = self.units.keys()
        if not isinstance(units,list):
            units = [units]
        for u in units[:]:
            if str(u) not in self.units.keys():
                units.remove(u)
                print(str(u)+' not in units.keys()')
        if len(units)<1:
            return
        units, unitsYPos = self.getOrderedUnits(units)
        
        if protocol is None:
            protocol = self.getProtocolIndex('checkerboard')
        protocol = str(protocol)          
        p = self.visstimData[protocol]
        assert(set(p['bckgndDir'])=={0,180} and set(p['patchDir'])=={0,180} and 0 in p['bckgndSpeed'] and 0 in p['patchSpeed'])
        
        if trials is None:
            trials = np.arange((p['trialStartFrame']).size)
        if p['frameSamples'][p['trialStartFrame'][-1]]+p['trialNumFrames'][-1]/p['frameRate']*self.sampleRate>p['frameSamples'][-1]:
            trials = trials[:-1]   
        trialStartFrame = p['trialStartFrame'][trials]
        trialDuration = (p['trialNumFrames'][trials]).astype(int)
        trialStartSamples = p['frameSamples'][trialStartFrame]
        trialEndSamples = p['frameSamples'][trialStartFrame+trialDuration]
        
        if plot:
            plt.figure(figsize=(10,3*len(units)),facecolor='w')
            gs = gridspec.GridSpec(2*len(units),4)
            row = 0
        bckgndSpeed = np.concatenate((-p['bckgndSpeed'][:0:-1],p['bckgndSpeed']))
        patchSpeed = np.concatenate((-p['patchSpeed'][:0:-1],p['patchSpeed']))
        resp = np.full((bckgndSpeed.size,patchSpeed.size,p['patchSize'].size,p['patchElevation'].size),np.nan)
        resp = np.tile(resp[:,:,:,:,None],math.ceil(trials.size/(resp.size-2*p['patchSpeed'].size*p['patchSize'].size))+3)
        for uindex,u in enumerate(units):
            spikesPerTrial = self.findSpikesPerTrial(trialStartSamples,trialEndSamples,self.units[str(u)]['times'][protocol])
            trialSpikeRate = spikesPerTrial/((1/p['frameRate'])*trialDuration)
            for n in trials:
                i = patchSpeed==p['trialPatchSpeed'][n] if p['trialPatchDir'][n]==0 else patchSpeed==-p['trialPatchSpeed'][n]
                j = bckgndSpeed==p['trialBckgndSpeed'][n] if p['trialBckgndDir'][n]==0 else bckgndSpeed==-p['trialBckgndSpeed'][n]
                k = p['patchSize']==p['trialPatchSize'][n]
                l = p['patchElevation']==p['trialPatchPos'][n]
                resp[i,j,k,l,np.count_nonzero(np.logical_not(np.isnan(resp[i,j,k,l,:])))] = trialSpikeRate[n]
            meanResp = np.nanmean(resp,axis=4)
            meanResp -= np.nanmean(resp[patchSpeed.size//2,bckgndSpeed.size//2,:,:,:])
            #meanResp /= np.nanstd(resp[patchSpeed.size//2,bckgndSpeed.size//2,:,:,:])
            for k in range(p['patchSize'].size):
                for l in range(p['patchElevation'].size):
                    meanResp[patchSpeed.size//2,:,k,l] = meanResp[patchSpeed.size//2,:,0,0]
            for i in range(patchSpeed.size):
                for j in range(bckgndSpeed.size):
                    if patchSpeed[i]==bckgndSpeed[j]:
                        meanResp[i,j,:,:] = meanResp[patchSpeed.size//2,j]            
            self.units[str(u)]['checkerboard'] = {'meanResp':meanResp}
            resp[:] = np.nan
            
            if plot:
                # plot response vs background and patch speed (averaging over patch size and elevation)
                ax = plt.subplot(gs[row:row+2,0:2])
                respMat = np.nanmean(np.nanmean(meanResp,axis=3),axis=2)
                cLim = max(1,np.max(abs(respMat)))
                plt.imshow(respMat,cmap='bwr',clim=(-cLim,cLim),interpolation='none',origin='lower')
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                ax.set_xticks(range(bckgndSpeed.size))
                ax.set_xticklabels(bckgndSpeed[:])
                ax.set_xlabel('Background Speed',fontsize='x-small')
                ax.set_yticks(range(patchSpeed.size))
                ax.set_yticklabels(patchSpeed[:])
                ax.set_ylabel('Patch Speed',fontsize='x-small')
                ax.set_title('Unit '+str(u),fontsize='medium')
                cb = plt.colorbar(fraction=0.05,pad=0.04,shrink=0.5)
                cb.set_ticks([-int(cLim),0,int(cLim)])
                cb.ax.tick_params(length=0,labelsize='xx-small')
                
                # plot mean response across background and patch speed axes
                ax = plt.subplot(gs[row,2])
                bck = np.nanmean(respMat,axis=0)
                pch = np.nanmean(respMat,axis=1)
                plt.plot(bckgndSpeed,bck,color='0.6',label='bckgnd mean')
                plt.plot(patchSpeed,pch,color='0',label='patch mean')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                xmin,xmax = min(bckgndSpeed.min(),patchSpeed.min()), max(bckgndSpeed.max(),patchSpeed.max())
                ax.set_xticks([xmin,0,xmax])
                ax.set_xlabel('Speed',fontsize='x-small')
                ymin,ymax = min(bck.min(),pch.min()), max(bck.max(),pch.max())
                ax.set_yticks([int(ymin),int(ymax)])
                ax.set_ylabel('Spikes/s',fontsize='x-small')
                plt.legend(loc='upper right',frameon=False,fontsize='x-small')
                
                # plot response to background or patch alone
                ax = plt.subplot(gs[row+1,2])
                bck = respMat[patchSpeed.size//2,:]
                pch = respMat[:,bckgndSpeed.size//2]
                plt.plot(bckgndSpeed,bck,color='0.6',label='bcknd only')
                plt.plot(patchSpeed,pch,color='0',label='patch only')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                ax.set_xticks([xmin,0,xmax])
                ax.set_xlabel('Speed',fontsize='x-small')
                ymin,ymax = min(bck.min(),pch.min()), max(bck.max(),pch.max())
                ax.set_yticks([int(ymin),int(ymax)])
                plt.legend(loc='upper right',frameon=False,fontsize='x-small')
                
                # plot response vs patch size (averaging across patch speed and elevation)
                ax = plt.subplot(gs[row,3])
                r = [np.nanmean(meanResp[patchSpeed!=0,bckgndSpeed.size//2,k,:]) for k in range(p['patchSize'].size)]
                plt.plot(p['patchSize'],r,color='0')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                ax.set_xticks(p['patchSize'])
                ax.set_xlabel('Patch Size',fontsize='x-small')
                ax.set_yticks([int(min(r)),int(max(r))])
                
                # plot response vs patch elevation (averaging across patch speed and size)
                ax = plt.subplot(gs[row+1,3])
                r = [np.nanmean(meanResp[patchSpeed!=0,bckgndSpeed.size//2,:,l]) for l in range(p['patchElevation'].size)]
                plt.plot(p['patchElevation'],r,color='0')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                ax.set_xticks(p['patchElevation'])
                ax.set_xlabel('Patch Elevation',fontsize='x-small')
                ax.set_yticks([int(min(r)),int(max(r))])
                
                row += 2
    
    def analyzeRunning(self, units, protocol):
        if units is None:
            units = self.units.keys()
        if not isinstance(units,list):
            units = [units]
        for u in units[:]:
            if str(u) not in self.units.keys():
                units.remove(u)
                print(str(u)+' not in units.keys()')
        if len(units)<1:
            return
        
        kernelWidth=500.0
        for uindex, u in enumerate(units):
            spikes = self.units[str(u)]['times'][str(protocol)]
            wd = -self.behaviorData[str(protocol)]['running']
            fh, _ = np.histogram(spikes, np.arange(0, (wd.size+1)*int(kernelWidth), int(kernelWidth)))
            frc = np.convolve(fh, np.ones(kernelWidth),'same')/kernelWidth
            
            speedBins = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
            binnedSpeed = np.digitize(wd, speedBins)
            fr_binned = []
            fr_std = []
            for sbin, _ in enumerate(speedBins):
                binIndices = binnedSpeed==sbin
                #print binIndices
                fr_thisbin = np.mean(frc[binIndices])
                fr_stdThisBin = np.std(frc[binIndices])
                fr_binned.append(fr_thisbin)
                fr_std.append(fr_stdThisBin)
            
            self.units[str(u)]['runModulation'] = {}
            self.units[str(u)]['runModulation'][str(protocol)] = [speedBins, np.array(fr_binned), np.array(fr_std)]
            
    
    def plotRaster(self,unit,protocol,startSamples=None,windowDur=None,paramNames=None):
        protocol = str(protocol)
        params = []
        if startSamples is None:
            p = self.visstimData[str(protocol)]
            try:
                trialStartFrame = p['trialStartFrame']
            except:
                trialStartFrame = p['stimStartFrames']
            startSamples = p['frameSamples'][trialStartFrame]
            if windowDur is None:
                windowDur = np.diff(startSamples)
            startSamples = startSamples[:-1]
            if paramNames is not None:
                for name in paramNames:
                    params.append(p[name][:-1])
        else:
            windowDur = [windowDur for _ in startSamples]
        spikes = self.units[str(unit)]['times'][protocol]
        
        plt.figure()
        ax = plt.subplot(1,1,1)
        if len(params)<1:
            self.appendToRaster(ax,spikes,startSamples,windowDur)
        else:
            self.parseRaster(ax,spikes,startSamples,windowDur,params)
        ax.set_xlim([0,max(windowDur)/self.sampleRate])
        ax.set_ylim([-0.5,startSamples.size+0.5])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trial')
        ax.set_title('Unit '+str(unit))
        return ax
        
        
    def parseRaster(self,ax,spikes,startSamples,windowDur,params,paramIndex=0,trialsIn=None,row=0):
        if trialsIn is None:
            trialsIn = range(len(startSamples))
        for val in np.unique(params[paramIndex]):
            trialIndex = np.intersect1d(trialsIn,np.where(params[paramIndex]==val)[0])
            if paramIndex<len(params)-1:
                row = self.parseRaster(ax,spikes,startSamples,windowDur,params,paramIndex+1,trialIndex,row)
            else:
                row = self.appendToRaster(ax,spikes,startSamples,windowDur,trialIndex,row)
        return row


    def appendToRaster(self,ax,spikes,startSamples,windowDur,trialIndex=None,row=0):
        if trialIndex is None:
            trialIndex = range(len(startSamples))
        for i in trialIndex:
            spikeTimes = (spikes[np.logical_and(spikes>startSamples[i],spikes<startSamples[i]+windowDur[i])]-startSamples[i])/self.sampleRate
            ax.vlines(spikeTimes,row-0.4,row+0.4,'k')
            row += 1
        return row
                        
        
    def getProtocolIndex(self, label):
        protocol = []
        protocol.extend([i for i,f in enumerate(self.kwdFileList) if ntpath.dirname(f).endswith(label)])
        if len(protocol)<1:
            raise ValueError('No protocols found matching: '+label)
        elif len(protocol)>1:
            raise ValueError('Multiple protocols found matching: '+label)
        return protocol[0]
    
    
    def runAllAnalyses(self, units=None, protocolsToRun = ['sparseNoise', 'gratings', 'gratings_ori', 'spots', 'checkerboard'], useCache=False):
        if units is None:
            units = self.units.keys()
        if type(units) is int:
            units = [units]
        
        for pro in protocolsToRun:
            protocol = self.getProtocolIndex(pro)
     
            if 'gratings'==pro:
                self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf')
            elif 'gratings_ori'==pro:
                self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori')
            elif 'sparseNoise' in pro:
                self.findRF(units, protocol=protocol, useCache=useCache)
            elif 'spots' in pro:
                self.analyzeSpots(units, protocol=protocol, useCache=useCache)
            elif 'checkerboard' in pro:
                self.analyzeCheckerboard(units, protocol=protocol)
            else:
                print("Couldn't find analysis script for protocol type:", pro)
                
                
    def getOrderedUnits(self,units=None):
        # orderedUnits, yPosition = self.getOrderedUnits(units)
        if units is None:
            units = self.units.keys()
        if not isinstance(units,list):
            units = [units]
        units = [str(u) for u in units]
        orderedUnits = [(u,self.units[u]['ypos']) for u in self.units.keys() if u in units]
        orderedUnits.sort(key=lambda i: i[1], reverse=True)
        return zip(*orderedUnits)
    
    
    def getSingleUnits(self, fileDir = None, protocolsToAnalyze = None):
        if fileDir is None:
            fileDir = getDir()
        fileList, nsamps = getKwdInfo(dirPath=fileDir)
        if protocolsToAnalyze is None:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir)
        else:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir, protocolsToAnalyze=protocolsToAnalyze)
    
    
    def loadClusteredData(self, kwdNsamplesList = None, protocolsToAnalyze = None, fileDir = None):
        from load_phy_template import load_phy_template
                 
        if fileDir is None:
            fileDir = getDir()
        
        if protocolsToAnalyze is None:
            protocolsToAnalyze = np.arange(len(self.d))
        
        self.units = load_phy_template(fileDir, sampling_rate = self.sampleRate)
        for unit in self.units.keys():
            spikeTimes = (self.units[unit]['times']).astype(int)
           
            if kwdNsamplesList is not None:
                self.units[unit]['times'] = {}
                protocolEnds = np.cumsum(kwdNsamplesList)
                protocolStarts = np.insert(protocolEnds, 0, 0)[:-1] - 1
                for pro in protocolsToAnalyze:                    
                    self.units[unit]['times'][str(pro)] = spikeTimes[np.logical_and(spikeTimes >= protocolStarts[pro], spikeTimes < protocolEnds[pro])]
                    self.units[unit]['times'][str(pro)] -= protocolStarts[pro]
            else:
              self.units[unit]['times'] = spikeTimes       


    def saveHDF5(self, fileSaveName = None, fileOut = None, saveDict = None, grp = None):
        if fileSaveName is None and fileOut is None:
            fileSaveName = saveFile()
            if fileSaveName=='':
                return
            fileOut = h5py.File(fileSaveName, 'w')
        elif fileSaveName is not None and fileOut is None:            
            fileOut = h5py.File(fileSaveName,'w')

        if saveDict is None:
            saveDict = self.__dict__
        if grp is None:    
            grp = fileOut['/']
        
        for key in saveDict:    
            if type(saveDict[key]) is dict:
                self.saveHDF5(fileOut=fileOut, saveDict=saveDict[key], grp=grp.create_group(key))
            else:
                try:
                    grp[key] = saveDict[key]
                except:
                    try:
                        grp.create_dataset(key,data=np.array(saveDict[key],dtype=object),dtype=h5py.special_dtype(vlen=str))
                    except:
                        print('Could not save: ', key)
                    
                    
    def loadHDF5(self, fileName=None, grp=None, loadDict=None):
        if fileName is None and grp is None:        
            fileName = getFile()
            if fileName=='':
                return
        if grp is None:
            grp = h5py.File(fileName)
        for key,val in grp.items():
            if isinstance(val,h5py._hl.dataset.Dataset):
                v = val.value
                if isinstance(v,np.ndarray) and v.dtype==np.object:
                    v = v.astype('U')
                if loadDict is None:
                    setattr(self,key,v)
                else:
                    loadDict[key] = v
            elif isinstance(val,h5py._hl.group.Group):
                if loadDict is None:
                    setattr(self,key,{})
                    self.loadHDF5(grp=val,loadDict=getattr(self,key))
                else:
                    loadDict[key] = {}
                    self.loadHDF5(grp=val,loadDict=loadDict[key])
                    
    
    def saveWorkspace(self, variables=None, saveGlobals = False, fileName=None, exceptVars = []):
        if fileName is None:
            fileName = saveFile()
            if fileName=='':
                return
        shelf = shelve.open(fileName, 'n')
        
        if variables is None:
            if not saveGlobals:
                variables = self.__dict__.keys()
            else:
                variables = self.__dict__.keys() + globals().keys()
        
        for key in variables:
            try:
                if key in self.__dict__.keys():
                    shelf[key] = self.__dict__[key]
                else:
                    shelf[key] = globals()[key]    
            except TypeError:
                # __builtins__, my_shelf, and imported modules can not be shelved.
                print('ERROR shelving: {0}'.format(key))
        shelf.close()


    def loadWorkspace(self, fileName = None):
        if fileName is None:        
            fileName = getFile()
            if fileName=='':
                return
        shelf = shelve.open(fileName)
        for key in shelf:
            setattr(self, key, shelf[key])
        shelf.close()


# utility functions

def getFile():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return QtGui.QFileDialog.getOpenFileName(None,'Choose File')
    
    
def getDir(rootDir=''):
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return QtGui.QFileDialog.getExistingDirectory(None,'Choose Directory',rootDir) 
    

def saveFile():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return QtGui.QFileDialog.getSaveFileName(None,'Save As')


def getKwdInfo(dirPath=None):
    # kwdFiles, nSamples = getKwdInfo()
    # returns kwd file paths and number of samples in each file ordered by file start time
    if dirPath is None:    
        dirPath = getDir(dataDir)
        if dirPath == '':
            return
    kwdFiles = []
    startTime = []
    nSamples = []
    for item in os.listdir(dirPath):
        itemPath = os.path.join(dirPath,item)
        if os.path.isdir(itemPath):
            for f in os.listdir(itemPath):
                if f[-4:]=='.kwd':
                    startTime.append(datetime.datetime.strptime(os.path.basename(itemPath)[0:19],'%Y-%m-%d_%H-%M-%S'))
                    kwdFiles.append(os.path.join(itemPath,f))
                    kwd = h5py.File(kwdFiles[-1],'r')
                    nSamples.append(kwd['recordings']['0']['data'].shape[0])
    return zip(*[n[1:] for n in sorted(zip(startTime,kwdFiles,nSamples),key=lambda z: z[0])])


def makeDat(kwdFiles):
    dirPath = os.path.dirname(os.path.dirname(kwdFiles[0]))
    datFilePath = os.path.join(dirPath,os.path.basename(dirPath)+'.dat')
    datFile = open(datFilePath,'wb')
    for filenum, filePath in enumerate(kwdFiles):
        kwd = h5py.File(filePath,'r')
        dset = kwd['recordings']['0']['data']
        i = 0
        while i<dset.shape[0]:
            (dset[i:i+dset.chunks[0],:128]).tofile(datFile)                        
            i += dset.chunks[0]
        print('Completed file ' + str(filenum) + ' of ' + str(len(kwdFiles)))
    datFile.close()
    copyPath = r'\\10.128.38.3\data_local_1\corbett'
    print('copying dat file to ' + copyPath)
    shutil.copy(datFilePath,copyPath)
    
    
def gauss2D(xyTuple,x0,y0,sigX,sigY,theta,amplitude):
    x,y = xyTuple # (x,y)
    y = y[:,None]                                                                                                             
    a = (math.cos(theta)**2)/(2*sigX**2)+(math.sin(theta)**2)/(2*sigY**2)   
    b = (math.sin(2*theta))/(4*sigX**2)-(math.sin(2*theta))/(4*sigY**2)    
    c = (math.sin(theta)**2)/(2*sigX**2)+(math.cos(theta)**2)/(2*sigY**2)   
    z = amplitude * np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))                                   
    return z.ravel()


def fitGauss2D(x,y,data,initialParams):
    '''
    # test:
    import probeData
    import numpy as np
    from matplotlib import pyplot as plt
    params = (5,5,1,1,0,1)
    x = np.arange(11)
    y = np.arange(11)
    data = probeData.gauss2D((x,y),*params).reshape(y.size,x.size)
    fitParams = probeData.fitGauss2D(x,y,data,params)
    xreal,yreal = probeData.getEllipseXY(*params[:-1])
    xfit,yfit = probeData.getEllipseXY(*fitParams[:-1])
    plt.figure()
    ax = plt.subplot(1,1,1)
    ax.imshow(data,cmap='gray',interpolation='none')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xreal,yreal,'m',linewidth=2)
    ax.plot(xfit,yfit,'y:',linewidth=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    '''
    try:
        lowerBounds = np.array([-np.inf,-np.inf,0,0,0,0])
        upperBounds = np.array([np.inf,np.inf,np.inf,np.inf,2*math.pi,1.5*initialParams[-1]])
        fitParams,fitCov = scipy.optimize.curve_fit(gauss2D,(x,y),data.flatten(),p0=initialParams,bounds=(lowerBounds,upperBounds))
    except RuntimeError:
        print('fit failed')
        return
    # fitData = gauss2D((x,y),*fitParams).reshape(y.size,x.size)
    return fitParams


def getEllipseXY(x,y,a,b,angle):
    sinx = np.sin(np.arange(0,361)*math.pi/180)
    cosx = np.cos(np.arange(0,361)*math.pi/180)
    X = x+a*cosx*math.cos(angle)-b*sinx*math.sin(angle)
    Y = y+a*cosx*math.sin(angle)+b*sinx*math.cos(angle)
    return X,Y
    

def stfLogGauss2D(stfTuple,sf0,tf0,sigSF,sigTF,speedTuningIndex,amplitude):
    sf,tf = stfTuple
    tf = tf[:,None]
    z = amplitude * np.exp(-((np.log2(sf)-np.log2(sf0))**2)/(2*sigSF**2)) * np.exp(-((np.log2(tf)-(speedTuningIndex*(np.log2(sf)-np.log2(sf0))+np.log2(tf0)))**2)/(2*sigTF**2))
    return z.ravel()


def fitStfLogGauss2D(sf,tf,data,initialParams):
    '''
    # test:
    import probeData
    import numpy as np
    from matplotlib import pyplot as plt
    params = (0.08,2,2,2,1,1)
    sf = np.array([0.01,0.02,0.04,0.08,0.16,0.32])
    tf = np.array([0.25,0.5,1,2,4,8])
    data = probeData.stfLogGauss2D((sf,tf),*params).reshape(sf.size,tf.size)
    fitParams = probeData.fitStfLogGauss2D(sf,tf,data,params)
    xreal,yreal = probeData.getStfContour(sf,tf,params)
    xfit,yfit = probeData.getStfContour(sf,tf,fitParams)
    plt.figure()
    ax = plt.subplot(1,1,1)
    ax.imshow(data,cmap='gray',origin='lower',interpolation='none')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xreal,yreal,'m',linewidth=2)
    ax.plot(xfit,yfit,'y:',linewidth=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    '''
    try:
        lowerBounds = np.array([0,0,0,0,-0.5,0])
        upperBounds = np.array([1,16,np.inf,np.inf,1.5,1.5*initialParams[-1]])
        fitParams,fitCov = scipy.optimize.curve_fit(stfLogGauss2D,(sf,tf),data.flatten(),p0=initialParams,bounds=(lowerBounds,upperBounds))
    except RuntimeError:
        print('fit failed')
        return
    # fitData = stfLogGauss2D((sf,tf),*fitParams).reshape(tf.size,sf.size)
    return fitParams


def getStfContour(sf,tf,fitParams):
    intpPts = 100
    sfIntp = np.logspace(np.log2(sf[0]*0.5),np.log2(sf[-1]*2),intpPts,base=2)
    tfIntp = np.logspace(np.log2(tf[0]*0.5),np.log2(tf[-1]*2),intpPts,base=2)
    intpFit = stfLogGauss2D((sfIntp,tfIntp),*fitParams).reshape(intpPts,intpPts)
    thresh = 0.6065*intpFit.max() # one stdev
    contourLine = np.full((2*intpPts,2),np.nan)
    for i in range(len(sfIntp)):
        c = np.where(intpFit[:,i]>thresh)[0]
        if len(c)>0:
            contourLine[i,0] = sfIntp[i]
            contourLine[i,1] = tfIntp[c[0]]
            contourLine[-1-i,0] = sfIntp[i]
            contourLine[-1-i,1] = tfIntp[c[-1]]
    contourLine = contourLine[np.logical_not(np.isnan(contourLine[:,0])),:]
    if contourLine.shape[0]>0:
        contourLine = np.concatenate((contourLine,contourLine[0,:][None,:]),axis=0)
    x,y = (np.log2(contourLine)-np.log2([sfIntp[0],tfIntp[0]])).T-1
    return x,y
    
    
def getDSI(resp,theta):
    theta = np.copy(theta)*math.pi/180
    sumX = np.sum(resp*np.cos(theta))
    sumY = np.sum(resp*np.sin(theta))
    dsi = np.sqrt(sumX**2+sumY**2)/resp.sum()
    # corbett's beautiful method
    prefTheta = ((math.atan2(sumY,sumX)*180/math.pi) + 360)%360
# sam's eqivalent ugly method
#    prefTheta = math.atan(sumY/sumX)*180/math.pi
#    if sumX<0:
#        if sumY>0:
#            prefTheta += 180
#        else:
#            prefTheta += 180
#    elif sumY<0:
#        prefTheta += 360
    return dsi, prefTheta


   
if __name__=="__main__":
    pass       