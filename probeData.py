# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:19:20 2016

@author: SVC_CCG
"""

from __future__ import division
import fileIO
import datetime, h5py, json, math, ntpath, os, re, shelve, shutil
import numpy as np
import scipy.ndimage.filters
import scipy.optimize
import scipy.signal
import scipy.stats
from matplotlib import pyplot as plt 
from matplotlib import gridspec
from matplotlib import cm
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel, convolve
import pandas
import extractWaveforms
import itertools

dataDir = r'C:\Users\SVC_CCG\Desktop\Data'

class probeData():
    
    def __init__(self):
        self.recording = 0
        self.TTLChannelLabels = ['VisstimOn', 'CamExposing', 'CamSaving', 'OrangeLaserShutter']
        self.channelMapFile = r'C:\Users\SVC_CCG\Documents\Python Scripts\imec_channel_map_D.prb'
        self.sampleRate = 30000.0
        self.digitalGain = 0.195
        self.analogGain = 0.00015258789
        self.wheelChannel = 134
        self.diodeChannel = 135
        self.visStimOnChannel = 136
        self.blueLaserChannel = 137
        self.orangeLaserChannel = 138
        self.camExposingChannel = 139
        self.camSavingChannel = 140

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
        
        
    def loadExperiment(self, dirPath=None, loadRunningData=False, loadUnits=True, loadWaveforms=False):
        self.kwdFileList, self.nsamps = getKwdInfo(dirPath)
        filelist = self.kwdFileList
        filePaths = [os.path.dirname(f) for f in filelist]            
        
        self._d = []
        for index, f in enumerate(filelist):
            ff = os.path.basename(os.path.dirname(f))
            ff = ff.split('_')[-1]  
            datDict = self.loadKwd(f)
            datDict['protocolName'] = ff
            datDict['numSamples'] = self.nsamps[index]
            self._d.append(datDict)
                    
        if loadUnits:
            self.getSingleUnits(fileDir=os.path.dirname(filePaths[0]))
        if loadRunningData:
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
                    wd = self._d[pro]['data'][:, self.wheelChannel]*self._d[pro]['gains'][self.wheelChannel]
                    wd = wd[::500]
                    self.behaviorData[str(pro)]['running'] = self.decodeWheel(wd)
                if not visStimFound:
                    print('No vis stim data found for ' + os.path.basename(proPath))
                if not eyeDataFound:
                    print('No eye tracking data found for ' + os.path.basename(proPath))
            
            for i, pro in enumerate(self.kwdFileList):
                if 'laser' in pro:
                    self.findAnalogPulses(self.blueLaserChannel, i)
            
            if loadWaveforms:
                self.getWaveforms()
    
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
            self.TTL[str(protocol)][self.TTLChannelLabels[chan]]['rising'] = timeSamples[np.intersect1d(eventsForChan, np.where(edges == 1))] - self._d[protocol]['firstAnalogSample']
            self.TTL[str(protocol)][self.TTLChannelLabels[chan]]['falling'] = timeSamples[np.intersect1d(eventsForChan, np.where(edges ==0))] - self._d[protocol]['firstAnalogSample']
        
        if str(protocol) in self.visstimData:
            if not hasattr(self, 'frameSamples'):
                self.alignFramesToDiode(protocol=protocol)
    
    
    def getVisStimData(self, filePath=None, protocol=0):
        if filePath is None:        
            filePath = fileIO.getFile()
        
        dataFile = h5py.File(filePath)        
        self.visstimData[str(protocol)] = {}
        for params in dataFile:
            if dataFile[params].size > 1:
                self.visstimData[str(protocol)][params] = dataFile[params][:]
            else:
                self.visstimData[str(protocol)][params] = dataFile[params][()]   
    
    
    def getEyeTrackData(self):
        expDate,anmID = self.getExperimentInfo()
        dirPath = os.path.join('\\\\aibsdata2\\nc-ophys\\corbettb\\Probe',expDate+'_'+anmID,'EyeTrackAnalysis')
        if not os.path.isdir(dirPath):
            print('could not find '+dirPath)
            return
                    
        for fileName in os.listdir(dirPath):
            protocolName = re.findall('MouseEyeTracker_'+'(.+)'+'_\d{8,8}_\d{6,6}_analysis',fileName)[0]
            protocolIndex = self.getProtocolIndex(protocolName)
            protocol = str(protocolIndex)
            eyeDataFile = h5py.File(os.path.join(dirPath,fileName))
            frameTimes = eyeDataFile['frameTimes'][:]
            
            if len(self.TTL[protocol])>0:
                camExposingSamples = self.TTL[protocol]['CamExposing']['rising']
                camSavingSamples = self.TTL[protocol]['CamSaving']['rising']
                firstFrameSample = camExposingSamples[np.where(camExposingSamples<camSavingSamples[0])[0][-1]]
                frameSamples = (frameTimes*self.sampleRate+firstFrameSample).astype(int)
            else:
                kwdFile = h5py.File(self.kwdFileList[protocolIndex])
                thresh = 10000
                camExposing,camSaving = kwdFile['recordings']['0']['data'][:,[self.camExposingChannel,self.camSavingChannel]].T
                camExposingSamples = np.where(np.logical_and(camExposing[:-1]<=thresh,camExposing[1:]>thresh))[0]+1
                camSavingSamples = np.where(np.logical_and(camSaving[:-1]<=thresh,camSaving[1:]>thresh))[0]+1
#                frameSamples = camExposingSamples[np.searchsorted(camExposingSamples,camSavingSamples)-1]
                firstFrameIndex = np.where(camExposingSamples<camSavingSamples[0])[0][-1]
                frameSampleIndex = firstFrameIndex+np.concatenate(([0],np.cumsum(np.round(np.diff(frameTimes)*60)).astype(int)))
                frameSamples = camExposingSamples[frameSampleIndex[frameSampleIndex<camExposingSamples.size]]        
        
            self.behaviorData[protocol]['eyeTracking'] = {'samples':frameSamples,'frameTimes':frameTimes}    
            for param in ('pupilArea','pupilX','pupilY','negSaccades','posSaccades'):      
                self.behaviorData[protocol]['eyeTracking'][param] = eyeDataFile[param][:]
    
    
    def alignFramesToDiode(self, frameSampleAdjustment = None, plot = False, protocol=0):
        if frameSampleAdjustment is None:
            self._frameSampleAdjustment = np.round((4.5/60.0) * self.sampleRate)
        
        thresh = 10000
        visStimOn = self._d[protocol]['data'][:,self.visStimOnChannel]
        self.visstimData[str(protocol)]['frameSamples'] = np.where(np.logical_and(visStimOn[:-1]<thresh,visStimOn[1:]>thresh))[0]+1+self._frameSampleAdjustment
#        self.visstimData[protocol]['frameSamples'] = (self.TTL[protocol]['VisstimOn']['falling'] + self._frameSampleAdjustment).astype(int)
        
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
#        self.channelMapping = self.channelMapping[np.where(self.channelMapping > 0)] - 1
        self.channelMapping = self.channelMapping - 1
    
    
    def decodeWheel(self, wheelData, kernelLength = 0.5, wheelSampleRate = 60.0):

        sampleRate = wheelSampleRate
        wheelData = wheelData - np.min(wheelData)
        wheelData = 2*np.pi*wheelData/np.max(wheelData)

        smoothFactor = sampleRate/60.0       
        angularWheelData = np.arctan2(np.sin(wheelData), np.cos(wheelData))
        angularWheelData = np.convolve(angularWheelData, np.ones(int(smoothFactor)), 'same')/smoothFactor

        artifactThreshold = (100.0/sampleRate)/7.6      #reasonable bound for how far (in radians) a mouse could move in one sample point (assumes top speed of 100 cm/s)
        angularDisplacement = (np.diff(angularWheelData) + np.pi)%(2*np.pi) - np.pi
        angularDisplacement[np.abs(angularDisplacement) > artifactThreshold ] = 0
        wheelData = np.convolve(angularDisplacement, np.ones(int(kernelLength*sampleRate)), 'same')/(kernelLength*sampleRate)
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
        fr = np.zeros(self._d[protocol]['numSamples'])
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
    
    def triggeredSDF(self, units, protocol, triggerPoints, win=[-0.5, 1.0], sdfSampInt = 0.001, appendToUnitDict=False):
        units, unitsYPos = self.getOrderedUnits(units)        
        winSamples = np.array(win)*self.sampleRate
        sdf=[]
        for u in units:
            spikes = self.units[u]['times'][str(protocol)] 
            usdf, time = self.getSDF(spikes, triggerPoints + winSamples[0], np.diff(winSamples), sampInt=sdfSampInt, avg=True)                  
            sdf.append(usdf)
            
            if appendToUnitDict:
                self.units[u][self.getProtocolLabel(protocol)] = {'_sdf': usdf, '_sdfTime': time}
                
        return np.array(sdf), time
    
    def findStatToRunPoints(self, protocol, runThresh=1, statThresh=1, window=2.0, wheelSampleRate=60, refractoryPeriod=5):
        if 'running' not in self.behaviorData[str(protocol)]:
            print('Could not find running data for this protocol')
            return
        
        w = self.behaviorData[str(protocol)]['running']
        if np.mean(w) < 0:
            w = -w
        
        window = int(window*wheelSampleRate)
        
        srt =[np.logical_and(np.mean(w[i:i+window]) > runThresh, all(w[i-window:i] < statThresh)) for i in np.arange(window, w.size-window)]
        srt = np.array(srt)
        srt = np.concatenate((np.array([False]*window), srt, np.array([False]*window)))
        srt = np.where([np.logical_and(srt[i], not any(srt[i-refractoryPeriod:i])) for i in np.arange(refractoryPeriod, srt.size)])[0] + refractoryPeriod
        
        for i,_ in enumerate(srt):
            if w[srt[i]] > runThresh:    
                while w[srt[i]] > runThresh:
                    srt[i] -= 1
            else:
                while w[srt[i]] < runThresh:
                    srt[i] += 1        
                
        srt *= int(self.sampleRate/wheelSampleRate)
        self.behaviorData[str(protocol)]['statToRunPoints'] = srt
    
    def runTriggeredAverage(self, units=None, protocol=None, win=[-0.5, 1.0], runThresh=1, statThresh=1, refractoryPeriod=5, plot=True):

        units, unitsYPos = self.getOrderedUnits(units)
        if protocol is None:
            protocol = range(len(self.kwdFileList))
        elif not isinstance(protocol,list):
            protocol = [protocol]
            
        winSamples = np.array(win)*self.sampleRate
        sdf = []
        for u in units:
            unitSDF = []
            for pro in protocol:
                if 'statToRunPoints' not in self.behaviorData[str(pro)]:
                    self.findStatToRunPoints(pro)
                spikes = self.units[u]['times'][str(pro)]
                rta, rtaTime = self.getSDF(spikes, self.behaviorData[str(pro)]['statToRunPoints'] + winSamples[0], np.diff(winSamples), avg=False)                  
                unitSDF.append(rta)
            unitSDF = np.array(unitSDF)
            sdf.append(np.nanmean(np.concatenate(unitSDF), axis=0))
        sdf = np.array(sdf)
        
        numEvents = 0
        wdTotal = []
        for pro in protocol:        
            ps = self.behaviorData[str(pro)]['statToRunPoints']            
            ps = (ps/500).astype(np.int)
            numEvents += len(ps)
            rWin = [-win[0], win[1]]
            wd = self.behaviorData[str(pro)]['running']
            tr = self.triggeredAverage(-wd, ps, win=rWin, sampleRate=60.)
            wdTotal.append(tr)
        wdTotal = np.nanmean(np.concatenate(wdTotal, axis=1), axis=1)

        if plot:
            if len(units) > 1:
                self.plotSDF1Axis(sdf, rtaTime)
                a = plt.gca()
                a.set_title(str(numEvents) + ' stat to run transitions')
                y = a.get_ylim()
                a.plot([rWin[0]]*2, [y[0], y[1]], 'k--')
                plt.figure(facecolor='w')
                plt.plot(rtaTime, np.nanmean(sdf, axis=0))
                plt.plot(np.linspace(0, rtaTime[-1], wdTotal.shape[0]), wdTotal)                
                a = plt.gca()
                y = a.get_ylim()
                a.plot([rWin[0]]*2, [y[0], y[1]], 'k--')
            else:
                fig = plt.figure(facecolor='w')                
                a = fig.add_subplot(2,1,1)
                a.plot(rtaTime, sdf[0, :], 'k')
                a.set_title(str(numEvents) + ' stat to run transitions')
                y = a.get_ylim()
                a.plot([rWin[0]]*2, [y[0], y[1]], 'k--')
                a2 = fig.add_subplot(2,1,2)
                a2.plot(np.linspace(0, rtaTime[-1], wdTotal.shape[0]), wdTotal, 'r')
                y = a2.get_ylim()
                a2.plot([rWin[0]]*2, [y[0], y[1]], 'k--')
            

    def findSpikesPerTrial(self, trialStarts, trialEnds, spikes): 
        spikesPerTrial = np.zeros(trialStarts.size)
        for trialNum in range(trialStarts.size):
            spikesPerTrial[trialNum] = np.count_nonzero(np.logical_and(spikes>=trialStarts[trialNum],spikes<=trialEnds[trialNum]))
        return spikesPerTrial
        
            
    def findRF(self, units=None, adjustForPupil=False, usePeakResp=True, sigma=1, plot=True, minLatency=0.05, maxLatency=0.15, trials=None, protocol=None, fit=True, saveTag='', useCache=False, cmap='Blues'):

        units, unitsYPos = self.getOrderedUnits(units)
        
        if protocol is None:
            protocol = self.getProtocolIndex('sparseNoise')
        protocol = str(protocol)

        trialStartFrame = self.visstimData[protocol]['stimStartFrames']
        trialEndFrame = trialStartFrame + self.visstimData[protocol]['trialDuration']
        lastFullTrial = np.where(trialEndFrame<self.visstimData[protocol]['frameSamples'].size)[0][-1]
        if trials is None:
            trials = np.arange(lastFullTrial+1)
        elif len(trials)<1:
            return
        else:
            trials = np.array(trials)
            trials = trials[trials<=lastFullTrial]
        
        stimStartFrames = self.visstimData[protocol]['stimStartFrames'][trials]
        stimStartSamples = self.visstimData[protocol]['frameSamples'][stimStartFrames]

#        if trials is None:
#            trials = np.arange(self.visstimData[protocol]['stimStartFrames'].size-1)
#        else:
#            trials = np.array(trials)
#        
#        if len(trials) == 0:            
#            return
        
        minLatencySamples = minLatency*self.sampleRate
        maxLatencySamples = maxLatency*self.sampleRate
        
        posHistory = np.copy(self.visstimData[protocol]['boxPositionHistory'][trials])
        xpos = np.unique(posHistory[:,0])
        ypos = np.unique(posHistory[:,1])
        pixPerDeg = self.visstimData[str(protocol)]['pixelsPerDeg']
        elev, azim = ypos/pixPerDeg, xpos/pixPerDeg
        gridExtent = self.visstimData[protocol]['gridBoundaries']
        
        rfArea = np.full((len(units),2),np.nan)
        
        adjustX = np.zeros_like(stimStartSamples).astype(float)
        adjustY = np.zeros_like(stimStartSamples).astype(float)
        eyeWindow = int(self.sampleRate*self.visstimData[protocol]['trialDuration']/60.0)
        gridSpacing = self.visstimData[protocol]['gridSpacing']
        if adjustForPupil:
            if protocol not in self.behaviorData or 'eyeTracking' not in self.behaviorData[protocol]:
                print('no eye tracking data')
                if not plot:
                    return rfArea
            px = self.behaviorData[protocol]['eyeTracking']['pupilX']
            py = self.behaviorData[protocol]['eyeTracking']['pupilY']
            eyeSamples = self.behaviorData[protocol]['eyeTracking']['samples']
            for it, t in enumerate(trials):
                trialEyeFrames = np.logical_and(eyeSamples >= stimStartSamples[t], eyeSamples < stimStartSamples[t] + eyeWindow)
#                if np.nanmedian(px[trialEyeFrames]) < 25 or np.nanmedian(px[trialEyeFrames])>3:
#                    posHistory[t,0] = np.nan
#                    posHistory[t,1] = np.nan
                if np.isnan(np.nanmedian(px[trialEyeFrames])):
#                    posHistory[t,0] = np.nan
#                    posHistory[t,1] = np.nan
                    pass
                
                else:    
                    adjustX[it] = np.round((np.nanmedian(px) - np.nanmedian(px[trialEyeFrames]))/gridSpacing)
                    adjustY[it] = np.round((np.nanmedian(py) - np.nanmedian(py[trialEyeFrames]))/gridSpacing)
                
                    currentXPosIndex = np.where(xpos==posHistory[t, 0])[0][0]
                    currentYPosIndex = np.where(ypos==posHistory[t, 1])[0][0]
                    
                    newXindex = currentXPosIndex + adjustX[it]                   
                    newYindex = currentYPosIndex + adjustY[it]
                    
                    if min(newXindex, newYindex)>=0 and np.logical_and(newXindex <= xpos.size-1, newYindex <= ypos.size-1):
                        posHistory[t,0] = xpos[newXindex]
                        posHistory[t,1] = ypos[newYindex]
                    else:
                        posHistory[t,0] = np.nan
                        posHistory[t,1] = np.nan
                
        colorHistory = self.visstimData[protocol]['boxColorHistory'][trials, 0]
        boxSizeHistory = self.visstimData[protocol]['boxSizeHistory'][trials]/pixPerDeg
        boxSize = np.unique(boxSizeHistory)

        sizeTuningOn = np.full((len(units),boxSize.size),np.nan)
        sizeTuningOff = np.copy(sizeTuningOn)
        sizeTuningSize = boxSize.copy()
        sizeTuningLabel = boxSize.copy()
        if any(boxSize>100):
            sizeTuningSize[boxSize>100] = 50
            sizeTuningLabel = list(sizeTuningLabel)
            sizeTuningLabel[-1] = 'full'
        boxSize = boxSize[boxSize<100]
        
        onVsOff = np.full(len(units),np.nan)
        respLatency = np.full((len(units),2),np.nan)
        respNormArea = np.copy(respLatency)
        respHalfWidth = np.copy(respLatency)
            
        sdfSampInt = 0.001
        sdfSigma = 0.01
        sdfSamples = minLatencySamples+2*maxLatencySamples
        
        gaussianKernel = Gaussian2DKernel(stddev=sigma)
        
        if fit:
            onFit = np.full((len(units),len(boxSize),7),np.nan)
            offFit = np.copy(onFit)
            onFitError = np.full((len(units),len(boxSize)),np.nan)
            offFitError = np.copy(onFitError)
        
        if plot:
            fig = plt.figure(figsize=(10,10*len(units)),facecolor='w')
            gs = gridspec.GridSpec(len(units)*(len(boxSize)+1),4)
        
        for uindex, unit in enumerate(units):
            spikes = self.units[unit]['times'][protocol]
            if spikes.size<1:
                continue
            onResp = np.full((len(boxSize),ypos.size,xpos.size),np.nan)
            offResp = np.copy(onResp)
            sdfOn = np.zeros((len(boxSize),ypos.size,xpos.size,int(round(sdfSamples/self.sampleRate/sdfSampInt))))
            sdfOff = np.zeros_like(sdfOn)
            for sizeInd,size in enumerate(boxSize):
                boxSizeTrials = boxSizeHistory==size
                for i,y in enumerate(ypos):
                    for j,x in enumerate(xpos):
                        posTrials = np.logical_and(posHistory[:, 1] == y,posHistory[:, 0] == x)
                        posOnTrials = np.logical_and(posTrials, colorHistory == 1)
                        posOffTrials = np.logical_and(posTrials, colorHistory == -1)
                        
                        posOnSamples = stimStartSamples[np.logical_and(posOnTrials,boxSizeTrials)]
                        if any(posOnSamples):
                            onResp[sizeInd,i,j] = np.mean(self.findSpikesPerTrial(posOnSamples+minLatencySamples,posOnSamples+maxLatencySamples,spikes))
                            sdfOn[sizeInd,i,j,:],_ = self.getSDF(spikes,posOnSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
                        
                        posOffSamples = stimStartSamples[np.logical_and(posOffTrials,boxSizeTrials)]
                        if any(posOffSamples):                            
                            offResp[sizeInd,i,j] = np.mean(self.findSpikesPerTrial(posOffSamples+minLatencySamples,posOffSamples+maxLatencySamples,spikes))
                            sdfOff[sizeInd,i,j,:],sdfTime = self.getSDF(spikes,posOffSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
            
            # convert spike count to spike rate
            onResp /= maxLatency-minLatency
            offResp /= maxLatency-minLatency
            
            # get full field flash resp
            fullFieldOnResp = fullFieldOffResp = np.nan
            fullFieldTrials = boxSizeHistory>100
            if any(fullFieldTrials):
                ffOnSamples = stimStartSamples[np.logical_and(fullFieldTrials,colorHistory==1)]
                if any(ffOnSamples):
                    fullFieldOnResp = np.mean(self.findSpikesPerTrial(ffOnSamples+minLatencySamples,ffOnSamples+maxLatencySamples,spikes))
                    fullFieldOnResp /= maxLatency-minLatency
                    fullFieldOnSDF,_ = self.getSDF(spikes,ffOnSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
                ffOffSamples = stimStartSamples[np.logical_and(fullFieldTrials,colorHistory==-1)]
                if any(ffOffSamples):
                    fullFieldOffResp = np.mean(self.findSpikesPerTrial(ffOffSamples+minLatencySamples,ffOffSamples+maxLatencySamples,spikes))
                    fullFieldOffResp /= maxLatency-minLatency
                    fullFieldOffSDF,_ = self.getSDF(spikes,ffOffSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
            
            # optionally use peak resp instead of mean rate
            inAnalysisWindow = np.logical_and(sdfTime>minLatency*2,sdfTime<minLatency+maxLatency)
            if usePeakResp:
                onResp = np.nanmax(sdfOn[:,:,:,inAnalysisWindow],axis=3)
                offResp = np.nanmax(sdfOff[:,:,:,inAnalysisWindow],axis=3)
                if not np.isnan(fullFieldOnResp):
                    fullFieldOnResp = np.nanmax(fullFieldOnSDF[inAnalysisWindow])
                if not np.isnan(fullFieldOffResp):
                    fullFieldOffResp = np.nanmax(fullFieldOffSDF[inAnalysisWindow])
                        
            # calculate size tuning
            sizeTuningOn[uindex,:boxSize.size] = np.nanmax(np.nanmax(onResp,axis=2),axis=1)
            sizeTuningOff[uindex,:boxSize.size] = np.nanmax(np.nanmax(offResp,axis=2),axis=1)
            if any(fullFieldTrials):
                sizeTuningOn[uindex,-1] = fullFieldOnResp
                sizeTuningOff[uindex,-1] = fullFieldOffResp
            
            # estimate spontRate using random trials and interval 0:minLatency
            nTrialTypes = np.unique(posHistory[~np.isnan(posHistory)]).size*boxSize.size*2
            nTrials = int(np.count_nonzero(boxSizeHistory<100)/nTrialTypes)
            nreps = 200
            spontPeakDist = np.zeros(nreps)
            spontCountDist = np.zeros(nreps)
            for ind in range(nreps):
                randTrials = np.random.choice(np.arange(trials.size),nTrials)
                spontPeakDist[ind] = np.max(self.getSDF(spikes,stimStartSamples[randTrials],minLatencySamples,sdfSigma,sdfSampInt))
                spontCountDist[ind] = np.mean(self.findSpikesPerTrial(stimStartSamples[randTrials],stimStartSamples[randTrials]+minLatencySamples,spikes))
            spontRateDist = spontPeakDist if usePeakResp else spontCountDist/minLatency
            spontRateMean = spontRateDist.mean()
            spontRateStd = spontRateDist.std()
            
            # determine which box sizes elicited significant responses                
            respThresh = spontRateMean+5*spontRateStd
            hasOnResp = np.zeros(len(boxSize),dtype=bool)
            hasOffResp = np.copy(hasOnResp)
            for sizeInd,_ in enumerate(boxSize):
                hasOnResp[sizeInd] = np.nanmax(onResp[sizeInd])>respThresh
                hasOffResp[sizeInd] = np.nanmax(offResp[sizeInd])>respThresh
            
            # filter responses for each box size
            onRespRaw = onResp.copy()
            offRespRaw = offResp.copy()
            for resp in (onResp,offResp):
                for sizeInd,_ in enumerate(boxSize):
                    resp[sizeInd] = convolve(resp[sizeInd], gaussianKernel, boundary='extend')
            
            # fit significant responses
            if fit:
                maxOffGrid = 10
                for sizeInd,_ in enumerate(boxSize):
                    for hasResp,resp,fitParams,fitError in zip((hasOnResp[sizeInd],hasOffResp[sizeInd]),(onResp[sizeInd],offResp[sizeInd]),(onFit[uindex,sizeInd],offFit[uindex,sizeInd]),(onFitError[uindex],offFitError[uindex])):
                        if hasResp and not np.any(np.isnan(resp)):
                            # params: x0 , y0, sigX, sigY, theta, amplitude, offset
                            i,j = np.unravel_index(np.argmax(resp),resp.shape)
                            sigmaGuess = (azim[1]-azim[0])*0.5*np.sqrt(np.count_nonzero(resp>resp.min()+0.5*(resp.max()-resp.min())))
                            initialParams = (azim[j],elev[i],sigmaGuess,sigmaGuess,0,resp.max(),np.percentile(resp,10))
                            fitResult,rmse = fitRF(azim,elev,resp,initialParams,maxOffGrid)
                            if fitResult is not None:
                                fitParams[:] = fitResult
                                fitError[sizeInd] = rmse
                
            # compare on and off resp magnitude (max across all box sizes)
            onMax = np.nanmax(onResp)
            offMax = np.nanmax(offResp)
            onVsOff[uindex] = (onMax-offMax)/(onMax+offMax)
            
            # calculate response latency and duration
            # SDF time is minLatency before stim onset through 2*maxLatency
            # Hence stim starts at minLatency and analysisWindow starts at 2*minLatency
            # Search analysisWindow for peak but allow searching outside analaysisWindow for halfMax
            sdfMaxInd = np.zeros((2,4),dtype=int)
            halfMaxInd = np.zeros((2,2),dtype=int)
            respLatencyInd = np.zeros(2,dtype=int)
            latencyThresh = spontPeakDist.mean()+5*spontPeakDist.std()
            for i,sdf in enumerate((sdfOn,sdfOff)):
                if not np.any(sdf[:,:,:,inAnalysisWindow]>latencyThresh):
                    continue
                sdfMaxInd[i,:] = np.unravel_index(np.nanargmax(sdf[:,:,:,inAnalysisWindow]),sdf[:,:,:,inAnalysisWindow].shape)
                sdfMaxInd[i,3] += np.where(inAnalysisWindow)[0][0]
                bestSDF = np.copy(sdf[sdfMaxInd[i,0],sdfMaxInd[i,1],sdfMaxInd[i,2],:])
                maxInd = sdfMaxInd[i,3]
                # find last thresh cross before peak for latency
                lastCrossing = np.where(bestSDF[:maxInd]<latencyThresh)[0]
                respLatencyInd[i] = lastCrossing[-1]+1 if any(lastCrossing) else np.where(inAnalysisWindow)[0][0]
                respLatency[uindex,i] = respLatencyInd[i]*sdfSampInt-minLatency
                # subtract min for calculating resp duration
                bestSDF -= np.min(bestSDF[inAnalysisWindow])
                # respNormArea = (area under SDF in analysisWindow) / (peak * analysisWindow duration)
                respNormArea[uindex,i] = np.trapz(bestSDF[inAnalysisWindow])*sdfSampInt/(bestSDF[maxInd]*(maxLatency-minLatency))                 
                # find last half-max cross before peak
                halfMax = 0.5*bestSDF[maxInd]
                preHalfMax = np.where(bestSDF[:maxInd]<halfMax)[0]
                halfMaxInd[i,0] = preHalfMax[-1]+1 if any(preHalfMax) else np.where(inAnalysisWindow)[0][0]
                # find first half-max cross after peak
                postHalfMax = np.where(bestSDF[maxInd:]<halfMax)[0]
                halfMaxInd[i,1] = maxInd+postHalfMax[0]-1 if any(postHalfMax) else bestSDF.size-1
                respHalfWidth[uindex,i] = (halfMaxInd[i,1]-halfMaxInd[i,0])*sdfSampInt
            
            # cache results
            self.units[unit]['sparseNoise' + saveTag] = {'gridExtent': gridExtent,
                                                         'elev': elev,
                                                         'azim': azim,
                                                         'boxSize': boxSize,
                                                         'onRespRaw': onRespRaw,
                                                         'offRespRaw': offRespRaw,
                                                         'onResp': onResp,
                                                         'offResp': offResp,
                                                         'spontRateMean': spontRateMean,
                                                         'spontRateStd': spontRateStd,
                                                         'onFit': onFit[uindex],
                                                         'offFit': offFit[uindex],
                                                         'onFitError': onFitError[uindex],
                                                         'offFitError': offFitError[uindex],
                                                         'sizeTuningOn': sizeTuningOn[uindex],
                                                         'sizeTuningOff': sizeTuningOff[uindex],
                                                         'onVsOff': onVsOff[uindex],
                                                         'respLatency': respLatency[uindex],
                                                         'respNormArea': respNormArea[uindex],
                                                         'respHalfWidth': respHalfWidth[uindex],
                                                         'trials': trials,
                                                         '_sdfOn': sdfOn,
                                                         '_sdfOff': sdfOff,
                                                         '_sdfTime': sdfTime}
            
            if plot:
                # sdfs and rf map
                maxVal = max(np.nanmax(onResp), np.nanmax(offResp))
                minVal = min(np.nanmin(onResp), np.nanmin(offResp))
                sdfMax = max(np.nanmax(sdfOn),np.nanmax(sdfOff))
                spacing = 0.2
                sdfXMax = sdfTime[-1]
                sdfYMax = sdfMax
                for sizeInd,size in enumerate(boxSize):
                    for onOffInd,(sdf,hasResp,resp,fitParams) in enumerate(zip((sdfOn[sizeInd],sdfOff[sizeInd]),(hasOnResp[sizeInd],hasOffResp[sizeInd]),(onResp[sizeInd],offResp[sizeInd]),(onFit[uindex,sizeInd],offFit[uindex,sizeInd]))):
                        onOffTitle = 'Off' if onOffInd else 'On'                        
                        row = uindex*(len(boxSize)+1)+sizeInd
                        col = onOffInd*2
                        ax = fig.add_subplot(gs[row,col])
                        x = 0
                        y = 0
                        for i,_ in enumerate(ypos):
                            for j,_ in enumerate(xpos):
                                ax.plot(x+sdfTime,y+sdf[i,j,:],color='k')
                                if not np.isnan(respLatency[uindex,onOffInd]) and all((sizeInd,i,j)==sdfMaxInd[onOffInd,:3]):
                                    ax.plot(x+sdfTime[halfMaxInd[onOffInd]],y+sdf[i,j,halfMaxInd[onOffInd]],color='r',linewidth=2)
                                    ax.plot(x+sdfTime[respLatencyInd[onOffInd]],y+sdf[i,j,respLatencyInd[onOffInd]],'bo')
                                x += sdfXMax*(1+spacing)
                            x = 0
                            y += sdfYMax*(1+spacing)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                        if onOffInd==0 and sizeInd==len(boxSize)-1:
                            ax.set_xticks([minLatency,minLatency+0.1])
                            ax.set_xticklabels(['','100 ms'])
                            ax.set_yticks([0,int(sdfMax)])
                        else:
                            ax.set_xticks([])
                            ax.set_yticks([])
                        ax.set_xlim([-sdfXMax*spacing,sdfXMax*(1+spacing)*xpos.size])
                        ax.set_ylim([-sdfYMax*spacing,sdfYMax*(1+spacing)*ypos.size])
                        if onOffInd==0:
                            ax.set_ylabel(str(int(size))+' deg',fontsize='medium')
                            if sizeInd==0:
                                ax.set_title('Unit '+str(unit),fontsize='medium')
                        
                        ax = fig.add_subplot(gs[row,col+1])
                        im = ax.imshow(resp, cmap=cmap, clim=(minVal,maxVal), interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]])
                        if not np.all(np.isnan(fitParams)):
                            ax.plot(fitParams[0],fitParams[1],'kx',markeredgewidth=2)
                            fitX,fitY = getEllipseXY(*fitParams[:-2])
                            ax.plot(fitX,fitY,'k',linewidth=2)
                            ax.set_xlim(gridExtent[[0,2]]-0.5)
                            ax.set_ylim(gridExtent[[1,3]]-0.5)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        cb = plt.colorbar(im, ax=ax, fraction=0.05, shrink=0.5, pad=0.04)
                        cb.ax.tick_params(length=0,labelsize='x-small')
                        cb.set_ticks([math.ceil(minVal),int(maxVal)])
                        if sizeInd==0:
                            ax.set_title(onOffTitle,fontsize='medium')
                
                if len(boxSize)>1:
                    ax = fig.add_subplot(gs[row+1,2])
                    ax.plot(sizeTuningSize,sizeTuningOn[uindex],'r')
                    ax.plot(sizeTuningSize,sizeTuningOff[uindex],'b')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                    ax.set_xlim([0,boxSize[-1]+boxSize[0]])
                    ax.set_ylim([0,1.05*max(np.nanmax(sizeTuningOn[uindex]),np.nanmax(sizeTuningOff[uindex]))])
                    ax.set_xticks(sizeTuningSize)
                    ax.set_xticklabels(sizeTuningLabel)
                    ax.set_xlabel('Size',fontsize='small')
                    ax.set_ylabel('Spikes/s',fontsize='small')
        
        sizeInd = np.argmin(np.absolute(boxSize-10))
        rfArea[:,0] = np.pi*np.prod(onFit[:,sizeInd,2:4],axis=1)
        rfArea[:,1] = np.pi*np.prod(offFit[:,sizeInd,2:4],axis=1)
        if adjustForPupil:                                             
            return rfArea
                    
        if plot and len(units)>1:
            # population plots
            # size tuning
            if len(boxSize)>1:
                plt.figure(facecolor='w')
                gspec = gridspec.GridSpec(2,2)
                for ind,(sizeResp,onOrOff) in enumerate(zip((sizeTuningOn,sizeTuningOff),('On','Off'))):
                    ax = plt.subplot(gspec[0,ind])
                    sizeRespNorm = sizeResp/np.nanmax(sizeResp,axis=1)[:,None]
                    sizeRespMean = np.nanmean(sizeRespNorm,axis=0)
                    sizeRespStd = np.nanstd(sizeRespNorm,axis=0)
                    ax.plot(sizeTuningSize,sizeRespMean,'k')
                    plt.fill_between(sizeTuningSize,sizeRespMean+sizeRespStd,sizeRespMean-sizeRespStd,color='0.6',alpha=0.3)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                    ax.set_xlim([0,boxSize[-1]+boxSize[0]])
                    ax.set_ylim([0,1.1])
                    ax.set_xticks(sizeTuningSize)
                    ax.set_yticks([0,0.5,1])
                    ax.set_xticklabels([])
                    if ind==0:
                        ax.set_ylabel('Norm Spikes/s',fontsize='medium')
                    else:
                        ax.set_yticklabels([])
                    ax.set_title(onOrOff,fontsize='large')
                    
                    ax = plt.subplot(gspec[1,ind])
                    sizeRespNorm[sizeRespNorm<1] = 0
                    bestSizeCount = np.nansum(sizeRespNorm,axis=0)
                    ax.bar(sizeTuningSize,bestSizeCount)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                    ax.set_xlim([0,boxSize[-1]+boxSize[0]])
                    ax.set_xticks(sizeTuningSize)
                    ax.set_xticklabels(sizeTuningLabel)
                    ax.set_xlabel('Size',fontsize='medium')
                    if ind==0:
                        ax.set_ylabel('Best Size Count',fontsize='medium')
            
            # onVsOff, respLatency, respNormArea, respHalfWidth, and rfArea
            for i,(data,bins,label) in enumerate(zip((respLatency,respNormArea,rfArea),
                                                (np.arange(0,0.275,0.025),np.arange(0,1.1,0.1),np.arange(0,4400,400)),
                                                ('Resp Latency','Resp Norm Area','RF Area'))):
                plt.figure(facecolor='w')
                for j,title in enumerate(('On','Off')):                
                    ax = plt.subplot(1,2,j+1)
                    ax.hist(data[:,j][~np.isnan(data[:,j])],bins)
                    ax.set_xlim(bins[[0,-1]])
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                    ax.set_xlabel(label,fontsize='medium')
                    ax.set_title(title,fontsize='large')
                    if j==0:
                        ax.set_ylabel('# Units',fontsize='medium')
            
            if fit:
                # RF centers
                plt.figure(facecolor='w')
                ax = plt.subplot(1,1,1)
                ax.plot(gridExtent[[0,2,2,0,0]],gridExtent[[1,1,3,3,1]],color='0.6')
                ax.plot(onFit[:,sizeInd,0],onFit[:,sizeInd,1],'o',markeredgecolor='r',markerfacecolor='none')
                ax.plot(offFit[:,sizeInd,0],offFit[:,sizeInd,1],'o',markeredgecolor='b',markerfacecolor='none')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                ax.set_xlim(gridExtent[[0,2]]+[-maxOffGrid,maxOffGrid])
                ax.set_ylim(gridExtent[[1,3]]+[-maxOffGrid,maxOffGrid])
                ax.set_xlabel('Azimuth',fontsize='medium')
                ax.set_ylabel('Elevation',fontsize='medium')
                ax.set_title('RF center (red = on, blue = off)',fontsize='large')
                
                # comparison of RF and probe position
                plt.figure(facecolor='w')
                gspec = gridspec.GridSpec(2,2)
                unitsYPos = np.array(unitsYPos)
                xlim = np.array([min(unitsYPos)-10,max(unitsYPos)+10])
                for j,(rfCenters,onOrOff) in enumerate(zip((onFit[:,sizeInd,:2],offFit[:,sizeInd,:2]),('On','Off'))):
                    for i,azimOrElev in enumerate(('Azimuth','Elevation')):
                        ax = plt.subplot(gspec[i,j])
                        hasRF = np.logical_not(np.isnan(rfCenters[:,i]))
                        if np.count_nonzero(hasRF)>1:
                            # linFit = (slope, intercept, r-value, p-value, stderror)
                            linFit = scipy.stats.linregress(unitsYPos[hasRF],rfCenters[hasRF,i])
                            ax.plot(xlim,linFit[0]*xlim+linFit[1],color='0.6')
                            ax.text(0.5,0.95,'$\mathregular{r^2}$ = '+str(round(linFit[2]**2,2))+', p = '+str(round(linFit[3],2)),
                                    transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',color='0.6')
                        ax.plot(unitsYPos,rfCenters[:,i],'ko',markerfacecolor='none')
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                        ax.set_xlim(xlim)
                        if i==0:
                            ax.set_title(onOrOff,fontsize='large')
                            ax.set_ylim(gridExtent[[0,2]]+[-maxOffGrid,maxOffGrid])
                            ax.set_xticklabels([])
                        else:
                            ax.set_xlabel('Probe Y Pos',fontsize='medium')
                            ax.set_ylim(gridExtent[[1,3]]+[-maxOffGrid,maxOffGrid])
                        if j==0:
                            ax.set_ylabel(azimOrElev,fontsize='medium')
                        else:
                            ax.set_yticklabels([])
                
    
    def analyzeFlash(self, units=None, trials=None, protocol=None, responseLatency=0.25, plot=True, sdfSigma=0.005, useCache=False, saveTag=''):
        units, unitsYPos = self.getOrderedUnits(units) 
            
        if protocol is None:
            label = 'flash'
            protocol = self.getProtocolIndex(label)
            
        protocol = str(protocol)
        
        if trials is None:
            trials = np.arange(self.visstimData[str(protocol)]['stimStartFrames'].size-1)
        else:
            trials = np.array(trials)

        if len(trials) == 0:            
            return
        
        if plot:
            plt.figure(figsize =(10, 4*len(units)),facecolor='w')
            gs = gridspec.GridSpec(len(units)+1, 2)

        trialStartFrames = self.visstimData[protocol]['stimStartFrames'][trials]
        trialDuration = self.visstimData[protocol]['stimDur']
        trialStartSamples = self.visstimData[protocol]['frameSamples'][trialStartFrames]
        trialEndSamples = self.visstimData[protocol]['frameSamples'][trialStartFrames + trialDuration]
        
        lumValues = np.unique(self.visstimData[protocol]['stimHistory'])
        trialLumValues = self.visstimData[protocol]['stimHistory'][trials]
        
        preTime = self.visstimData[protocol]['grayDur']/self.visstimData[protocol]['frameRate']
        stimTime = self.visstimData[protocol]['stimDur']/self.visstimData[protocol]['frameRate']
        postTime = preTime
        sdfSamples = round((preTime+stimTime+postTime)*self.sampleRate)
        sdfSampInt = 0.001
        
        onLatencies = []
        offLatencies = []
        for uindex, unit in enumerate(units):
            sdf = np.full((lumValues.size,round(sdfSamples/self.sampleRate/sdfSampInt)),np.nan)
            sdfOn = []
            sdfOff = []
            spikes = self.units[str(unit)]['times'][protocol]
 
            for lumindex, lum in enumerate(lumValues):
                lumTrials = np.where(trialLumValues==lum)[0]
                if len(lumTrials)>0:
                    sdf[lumindex], sdfTime = self.getSDF(spikes, trialStartSamples[lumTrials] - preTime*self.sampleRate, sdfSamples, sigma=sdfSigma)
                    if lum > 0:
                        sdfOn.append(sdf[lumindex])
                        soff, _ = self.getSDF(spikes, trialEndSamples[lumTrials] - preTime*self.sampleRate, sdfSamples, sigma=sdfSigma)
                        sdfOff.append(soff)
                    elif lum < 0:
                        sdfOff.append(sdf[lumindex])
                        son, _ = self.getSDF(spikes, trialEndSamples[lumTrials] - preTime*self.sampleRate, sdfSamples, sigma=sdfSigma)
                        sdfOn.append(son)
            
            sdfOn = np.array(sdfOn)
            sdfOff = np.array(sdfOff)
            sdfMeans = np.array([np.mean(sdfOn, axis=0), np.mean(sdfOff, axis=0)])
            
            baselineStart = 300
            baselineEnd = 500
            baselines = np.mean(sdf[:, baselineStart:baselineEnd], axis = 0)
            
            onLatency = np.where(sdfMeans[0, 500:1000] > np.mean(sdfMeans[0, 300:500], axis=0) + 5*np.std(sdfMeans[0, 300:500], axis=0))[0]
            onLatency = onLatency[0] if any(onLatency) else None
            offLatency = np.where(sdfMeans[1, 500:1000] > np.mean(sdfMeans[1, 300:500], axis=0) + 5*np.std(sdfMeans[1, 300:500], axis=0))[0]
            offLatency = offLatency[0] if any(offLatency) else None

           
            if onLatency is not None:
                onLatencies.append(onLatency)
            if offLatency is not None:
                offLatencies.append(offLatency)
            
            self.units[unit]['flash' + saveTag] = {'meanResp': sdf,
                                                   'lumValues': lumValues,
                                                   'trials': trials}    
            if plot:
                ax = plt.subplot(gs[uindex,0])
                ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                for lumi, lum in enumerate(sdf):
                    rval = 1 if lumValues[lumi]>0 else 0
                    bval = 1 if lumValues[lumi]<0 else 0
                    color = (rval, 0, bval) if np.max([rval, bval])>0 else (1,1,1)
                    alpha = abs(lumValues[lumi]) if abs(lumValues[lumi])>0 else 0.5                    
                    ax.plot(lum, color=color, alpha=alpha)

                ax.set_ylabel(str(unit), fontsize='small')
                
                ax2 = plt.subplot(gs[uindex, 1])
                ax2.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                ax2.plot(np.mean(sdfOn, axis=0), color='r')
                ax2.plot(np.mean(sdfOff, axis=0), color='b')
                if onLatency is not None:                
                    ax2.plot(onLatency+500, sdfMeans[0, onLatency+500], 'ro')
                if offLatency is not None:
                    ax2.plot(offLatency+500, sdfMeans[1, offLatency+500], 'bo')
                                
                if uindex==len(units)-1:
                    ax.set_xlabel('Time, ms', fontsize='medium')
                    ax2.set_xlabel('Time, ms', fontsize = 'medium')
                else:
                    ax.tick_params(bottom='off', labelbottom='off')
                    ax2.tick_params(bottom='off', labelbottom='off')
        if plot and len(units)>1:
            plt.figure(facecolor='w')
            if any(onLatencies):
                plt.hist(onLatencies, bins=np.arange(0, 210, 10), color='r')
            if any(offLatencies):
                plt.hist(offLatencies, bins=np.arange(0, 210, 10), color='b', alpha=0.5)
            a = plt.gca()
            a.set_xlabel('Latency, ms')
            a.set_ylabel('Count')
            
        return sdfMeans
     
              
    def analyzeGratings(self, units=None, trials=None, responseLatency=0.25, usePeakResp=False, sdfSigma = 0.02, plot=True, protocol=None, protocolType='stf', fit=True, saveTag='', useCache=False):
    
        units, unitsYPos = self.getOrderedUnits(units) 
            
        if protocol is None:
            if protocolType=='stf':
                label = 'gratings'
            elif protocolType=='ori':
                label = 'gratings_ori'
            protocol = self.getProtocolIndex(label)
        protocol = str(protocol)

        if trials is None:
            trials = np.arange(self.visstimData[str(protocol)]['stimStartFrames'].size-1)
        else:
            trials = np.array(trials)
            numFullTrials = self.visstimData[str(protocol)]['stimStartFrames'].size-1
            trials = trials[trials<numFullTrials]
        
        if len(trials) == 0:            
            return
        
        # ignore trials with bogus sf or tf
        trialContrast = self.visstimData[str(protocol)]['stimulusHistory_contrast'][trials]
        trialSF = self.visstimData[str(protocol)]['stimulusHistory_sf'][trials]
        trialTF = self.visstimData[str(protocol)]['stimulusHistory_tf'][trials]
        tol = 1e-9
        sf = np.copy(self.visstimData[protocol]['sf'])
        sf = sf[np.logical_and(sf>0.01-tol,sf<0.32+tol)]
        tf = np.copy(self.visstimData[protocol]['tf'])
        tf = tf[np.logical_and(tf>0.5-tol,tf<8+tol)]
        stfTrialsToUse = np.logical_and(np.logical_and(trialSF>0.01-tol,trialSF<0.32+tol),
                                        np.logical_and(trialTF>0.5-tol,trialTF<8+tol))
        trialsToUse = np.logical_or(np.isclose(trialContrast,0),stfTrialsToUse)
        trials = trials[trialsToUse]         
        
        trialContrast = trialContrast[trialsToUse]
        trialSF = trialSF[trialsToUse]
        trialTF = trialTF[trialsToUse]
        
        # as presented ori 0 is rightward moving vertical bars and ori 90 is downward moving horizontal bars
        # change to more typical convention (90 up)
        trialOri = np.copy(self.visstimData[str(protocol)]['stimulusHistory_ori'][trials])
        trialOri[trialOri>0] = -(trialOri[trialOri>0]-360)
        ori = np.copy(self.visstimData[protocol]['ori'])
        ori[ori>0] = -(ori[ori>0]-360)
        ori = np.sort(ori)
        
        latencySamples = int(responseLatency*self.sampleRate)
        trialStartFrame = self.visstimData[protocol]['stimStartFrames'][trials]
        trialDuration = self.visstimData[protocol]['stimTime']
        trialStartSamples = self.visstimData[protocol]['frameSamples'][trialStartFrame]
        trialEndSamples = self.visstimData[protocol]['frameSamples'][trialStartFrame+trialDuration]
        
        preTime = self.visstimData[protocol]['preTime']/self.visstimData[protocol]['frameRate']
        stimTime = self.visstimData[protocol]['stimTime']/self.visstimData[protocol]['frameRate']
        postTime = self.visstimData[protocol]['postTime']/self.visstimData[protocol]['frameRate']
        if 'laserPreFrames' in self.visstimData[protocol]:
            laserPreTime = self.visstimData[protocol]['laserPreFrames']/self.visstimData[protocol]['frameRate']
            laserPostTime = self.visstimData[protocol]['laserPostFrames']/self.visstimData[protocol]['frameRate']
        else:
            laserPreTime = 0
            laserPostTime = 0
        if 'laserPreFrames' in self.visstimData[protocol]:        
            sdfSamples = round((laserPreTime+preTime+stimTime+postTime+laserPostTime)*self.sampleRate)
        else:
            sdfSamples = round((preTime+stimTime+postTime)*self.sampleRate)
        sdfSampInt = 0.001
        inAnalysisWindow = None
        
        if protocolType=='stf':
            stfFitParams = np.full((len(units),7),np.nan)
        else:
            dsi = np.full(len(units),np.nan)
            prefDir = np.copy(dsi)
            osi = np.copy(dsi)
            prefOri = np.copy(dsi)
        
        if plot:
            fig = plt.figure(figsize =(15,3*len(units)),facecolor='w')
            gridWidth = 3*len(ori) if protocolType=='stf' else len(tf)*len(sf)
            gs = gridspec.GridSpec(len(units),gridWidth)   
        
        for uindex, unit in enumerate(units):    
            spikes = self.units[str(unit)]['times'][protocol]
            trialResp = self.findSpikesPerTrial(trialStartSamples+latencySamples,trialEndSamples,spikes)
            trialResp /= (trialEndSamples-trialStartSamples+latencySamples)/self.sampleRate
            preTrialRate = self.findSpikesPerTrial(trialStartSamples-latencySamples,trialStartSamples,spikes)
            preTrialRate /= responseLatency
            
            # make resp matrix with shape tf x sf x ori
            # make similar matrices for pre-trial spike rate, spike density functons, and f1/f0
            respMat = np.full((len(tf),len(sf),len(ori)),np.nan)
            preTrialMat = respMat.copy()
            sdf = np.full(respMat.shape+(int(round(sdfSamples/self.sampleRate/sdfSampInt)),),np.nan)
            f1f0Mat = respMat.copy()
            contrastTrials = trialContrast>0+tol
            for tfInd,thisTF in enumerate(tf):
                tfTrials = np.isclose(trialTF,thisTF)
                for sfInd,thisSF in enumerate(sf):
                    sfTrials = np.isclose(trialSF,thisSF)
                    for oriInd,thisOri in enumerate(ori):
                        trialIndex = np.isclose(trialOri,thisOri)
                        for i in (contrastTrials,tfTrials,sfTrials):
                            trialIndex = np.logical_and(trialIndex,i)
                        if any(trialIndex):
                            respMat[tfInd,sfInd,oriInd] = np.mean(trialResp[trialIndex])
                            preTrialMat[tfInd,sfInd,oriInd] = np.mean(preTrialRate[trialIndex])
                            sdf[tfInd,sfInd,oriInd,:],sdfTime = self.getSDF(spikes,trialStartSamples[trialIndex]-int((preTime+laserPreTime)*self.sampleRate),sdfSamples,sigma=sdfSigma,sampInt=sdfSampInt)
                            if inAnalysisWindow is None:
                                inAnalysisWindow = np.logical_and(sdfTime>preTime+responseLatency,sdfTime<preTime+stimTime)
                            s = sdf[tfInd,sfInd,oriInd,inAnalysisWindow]
                            f,pwr = scipy.signal.welch(s,1/sdfTime[1],nperseg=s.size,detrend='constant',scaling='spectrum')
                            pwr **= 0.5
                            f1Ind = np.argmin(np.absolute(f-thisTF))
                            f1f0Mat[tfInd,sfInd,oriInd] = pwr[f1Ind-1:f1Ind+2].max()/s.mean()
            
            peakRespMat = np.nanmax(sdf[:,:,:,inAnalysisWindow],axis=3)               
            if usePeakResp:
                respMat = peakRespMat
            
            # calculate spontRate from gray screen trials
            grayTrials = trialContrast<0+tol
            if any(grayTrials):
                peakSpontRateDist = self.getSDFNoise(spikes,trialStartSamples[grayTrials],max(trialEndSamples[grayTrials]-trialStartSamples[grayTrials]),sigma=sdfSigma,sampInt=sdfSampInt)
                peakSpontRateMean = peakSpontRateDist.mean()
                peakSpontRateStd = peakSpontRateDist.std()
                if usePeakResp:
                    spontRateMean = peakSpontRateMean
                    spontRateStd = peakSpontRateStd
                else:
                    nreps = 100
                    spontRateDist = np.zeros(nreps)
                    grayTrialInd = np.where(grayTrials)[0]
                    for ind in range(nreps):
                        spontRateDist[ind] = trialResp[np.random.choice(grayTrialInd,grayTrialInd.size)].mean()
                    spontRateMean = spontRateDist.mean()
                    spontRateStd = spontRateDist.std()
                hasResp = respMat>spontRateMean+5*spontRateStd
            else:
                spontRateMean = spontRateStd = peakSpontRateMean = peakSpontRateStd = np.nan
                hasResp = np.zeros_like(respMat,dtype=bool)
            
            # find significant responses
            tfHasResp,sfHasResp,oriHasResp = [np.unique(i) for i in np.where(hasResp)]
            maxRespInd = np.unravel_index(np.nanargmax(respMat),respMat.shape)
            
            if protocolType=='stf':
                # fit stf matrix for ori that elicited max resp
                stfFitError = np.nan
                stfFitOri = np.nan
                if fit and oriHasResp.size>0:
                    # params: sf0 , tf0, sigSF, sigTF, speedTuningIndex, amplitude, offset
                    for oriInd in np.argsort([np.nanmax(respMat[:,:,i]) for i in range(ori.size)])[::-1]:
                        resp = respMat[:,:,oriInd].copy()
                        if not np.any(np.isnan(resp)):
                            i,j = np.unravel_index(np.argmax(resp),resp.shape)
                            initialParams = (sf[j], tf[i], 1, 1, 0.25, resp.max(), resp.min())
                            fitParams,rmse = fitStf(sf,tf,resp,initialParams)
                            if fitParams is not None:
                                # get confidence intervals for sf0, tf0, and speedTuningIndex
                                nreps = 100
                                trialResampledFitParams = np.full((nreps,)+fitParams.shape,np.nan)
                                for ind in range(nreps):
                                    resp[:] = 0
                                    for tfInd,thisTF in enumerate(tf):
                                        for sfInd,thisSF in enumerate(sf):
                                            trialIndex = np.where(np.isclose(trialOri,ori[oriInd]) & np.isclose(trialSF,thisSF) & np.isclose(trialTF,thisTF))[0]
                                            trialIndexResampled = np.random.choice(trialIndex,trialIndex.size)
                                            if usePeakResp:
                                                s,_ = self.getSDF(spikes,trialStartSamples[trialIndexResampled]-int(preTime*self.sampleRate),sdfSamples,sigma=sdfSigma,sampInt=sdfSampInt)
                                                resp[tfInd,sfInd] = s[inAnalysisWindow].max()
                                            else:
                                                resp[tfInd,sfInd] = trialResp[trialIndexResampled].mean()
                                    i,j = np.unravel_index(np.argmax(resp),resp.shape)
                                    initialParams = (sf[j], tf[i], 1, 1, 0.25, resp.max(), resp.min())
                                    f,_ = fitStf(sf,tf,resp,initialParams)
                                    if f is not None:
                                        trialResampledFitParams[ind] = f
                                trialResampledFitParams = trialResampledFitParams[~np.isnan(trialResampledFitParams[:,0])]
                                if trialResampledFitParams.shape[0]>0:
                                    ci = np.percentile(trialResampledFitParams,[2.5,97.5],axis=0)
                                    ci[:,:2] = np.log2(ci[:,:2])
                                    ci = np.diff(ci,axis=0).squeeze()
                                    if np.all(ci[:2]<3) and ci[4]<1:
                                        stfFitParams[uindex] = fitParams
                                        stfFitError = rmse
                                        stfFitOri = ori[oriInd]
                                        break
            elif protocolType=='ori':
                # calculate DSI and OSI for sf/tf that elicited max resp
                if tfHasResp.size>0 and sfHasResp.size>0:
                    resp = respMat[maxRespInd[0],maxRespInd[1],:]
                    dsi[uindex],prefDir[uindex] = getDSI(resp,ori)
                    osi[uindex],prefOri[uindex] = getDSI(resp,2*ori)
                    prefOri[uindex] /= 2
            
            # cache results
            tag = 'gratings_' + protocolType + saveTag
            self.units[str(unit)][tag] = {'sf': sf,
                                          'tf': tf,
                                          'ori': ori,
                                          'spontRateMean': spontRateMean,
                                          'spontRateStd': spontRateStd,
                                          'respMat': respMat,
                                          'f1f0Mat': f1f0Mat,
                                          'trials': trials,
                                          'peakSpontRateMean': peakSpontRateMean,
                                          'peakSpontRateStd': peakSpontRateStd,
                                          'peakRespMat': peakRespMat,
                                          '_sdf': sdf,
                                          '_sdfTime': sdfTime}
            if protocolType=='stf':
                self.units[str(unit)][tag]['stfFitParams'] = stfFitParams[uindex]
                self.units[str(unit)][tag]['stfFitError'] = stfFitError
                self.units[str(unit)][tag]['stfFitOri'] = stfFitOri
            elif protocolType=='ori':
                self.units[str(unit)][tag]['dsi'] = dsi[uindex]
                self.units[str(unit)][tag]['prefDir'] = prefDir[uindex]
                self.units[str(unit)][tag]['osi'] = osi[uindex]
                self.units[str(unit)][tag]['prefOri'] = prefOri[uindex]
        
            if plot:
                if protocolType=='stf':
                    spacing = 0.2
                    sdfXMax = sdfTime[-1]
                    sdfYMax = np.nanmax(sdf)
                    centerPoint = spontRateMean if not np.isnan(spontRateMean) else np.nanmedian(respMat)
                    cLim = np.nanmax(np.absolute(respMat-centerPoint))
                    for oriInd,thisOri in enumerate(ori):
                        ax = fig.add_subplot(gs[uindex,oriInd*3:oriInd*3+2])
                        x = 0
                        y = 0
                        for i,_ in enumerate(tf):
                            for j,_ in enumerate(sf):
                                ax.plot(x+sdfTime,y+sdf[i,j,oriInd,:],color='k')
                                x += sdfXMax*(1+spacing)
                            x = 0
                            y += sdfYMax*(1+spacing)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                        ax.set_xticks([preTime,preTime+stimTime])
                        ax.set_xticklabels(['0',str(stimTime)+' s'])
                        ax.set_yticks([0,int(sdfYMax)])
                        ax.set_xlim([-sdfXMax*spacing,sdfXMax*(1+spacing)*sf.size])
                        ax.set_ylim([-sdfYMax*spacing,sdfYMax*(1+spacing)*tf.size])
                        if oriInd==0:
                            ax.set_ylabel('Unit '+str(unit), fontsize='medium')
                        if uindex==0:
                            ax.set_title('ori = '+str(thisOri),fontsize='medium') 
                       
                        ax = fig.add_subplot(gs[uindex,oriInd*3+2])
                        respMatOri = np.copy(respMat[:,:,oriInd])
                        xyNan = np.transpose(np.where(np.isnan(respMatOri)))
                        nanTrials = np.isnan(respMatOri)
                        respMatOri[nanTrials] = 0
                        im = ax.imshow(respMatOri, clim=(centerPoint-cLim, centerPoint+cLim), cmap='bwr', origin = 'lower', interpolation='none')
                        for xypair in xyNan:    
                            ax.text(xypair[1], xypair[0], 'nan', color='white', ha='center')
                        if fit and not all(np.isnan(stfFitParams[uindex])) and ori[oriInd]==stfFitOri:
                            ax.plot(np.log2(stfFitParams[uindex][0])-np.log2(sf[0]),np.log2(stfFitParams[uindex][1])-np.log2(tf[0]),'kx',markeredgewidth=2)
                            fitX,fitY = getStfContour(sf,tf,stfFitParams[uindex])
                            ax.plot(fitX,fitY,'k',linewidth=2)
                            ax.set_xlim([-0.5,sf.size-0.5])
                            ax.set_ylim([-0.5,tf.size-0.5])
                        ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                        ax.set_xticks([0,sf.size-1])
                        ax.set_yticks([0,tf.size-1])
                        ax.set_xticklabels([sf[0],sf[-1]])
                        ax.set_yticklabels([tf[0],tf[-1]])
                        ax.set_xlabel('Cycles/deg',fontsize='small')
                        ax.set_ylabel('Cycles/s',fontsize='small')
                        if oriInd not in oriHasResp:
                            ax.set_title('no resp',fontsize='x-small')
                        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
                        cb.set_ticks([math.ceil(centerPoint-cLim),round(centerPoint),math.floor(centerPoint+cLim)])
                        cb.ax.tick_params(length=0,labelsize='xx-small')
                else:
                    for i,_ in enumerate(tf):
                        for j,_ in enumerate(sf):
                            ax = fig.add_subplot(gs[uindex,i*len(tf)+j], projection='polar')
                            theta = ori * (np.pi/180.0)
                            theta = np.append(theta, theta[0])
                            rho = np.append(respMat[i,j,:], respMat[i,j,0])
                            ax.plot(theta, rho)
                            ax.set_rmax(np.nanmax(respMat)*1.05)
                            if i==maxRespInd[0] and j==maxRespInd[1] and not np.isnan(dsi[uindex]):
                                ax.set_title('DSI = '+str(round(dsi[uindex],2))+', prefDir = '+str(round(prefDir[uindex]))+'\n'+
                                             'OSI = '+str(round(osi[uindex],2))+', prefOri = '+str(round(prefOri[uindex])),fontsize='x-small')
                            if i==0 and j==0:
                                ax.set_ylabel('Unit '+str(unit))
                        
        if plot and len(units)>1:
            if protocolType=='stf':
                plt.figure(facecolor='w')
                ax = plt.subplot(1,2,1)
                ax.plot(np.log2(stfFitParams[:,0]),np.log2(stfFitParams[:,1]),'ko',markerfacecolor='none')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                ax.set_xlim(np.log2([sf[0]*0.5,sf[-1]*1.5]))
                ax.set_ylim(np.log2([tf[0]*0.5,tf[-1]*1.5]))
                ax.set_xticks(np.log2([sf[0],sf[-1]]))
                ax.set_yticks(np.log2([tf[0],tf[-1]]))
                ax.set_xticklabels([sf[0],sf[-1]])
                ax.set_yticklabels([tf[0],tf[-1]])
                ax.set_xlabel('Cycles/deg')
                ax.set_ylabel('Cycles/s')
                
                ax = plt.subplot(1,2,2)
                hasFit = np.logical_not(np.isnan(stfFitParams[:,4]))
                sortedSpeedTuning = np.sort(stfFitParams[hasFit,4])
                cumProbSpeedTuning = [np.count_nonzero(sortedSpeedTuning<=sti)/sortedSpeedTuning.size for sti in sortedSpeedTuning]
                ax.plot(sortedSpeedTuning,cumProbSpeedTuning,'k')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                ax.set_xlim([-0.5,1.5])
                ax.set_ylim([0,1])
                ax.set_yticks([0,0.5,1])
                ax.set_xlabel('Speed Tuning')
                ax.set_ylabel('Cum Prob')
                
            else:
                plt.figure(facecolor='w')
                ax = plt.subplot(1,3,1)
                ax.plot([0,0],[1,1],color='0.6')
                ax.plot(osi,dsi,'ko',markerfacecolor='none')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                amax = 1.1*max(np.nanmax(dsi),np.nanmax(osi))
                ax.set_xlim([0,amax])
                ax.set_ylim([0,amax])
                ax.set_xlabel('OSI')
                ax.set_ylabel('DSI')
                ax.set_aspect('equal')
                
                for ind,(pref,oriOrDir) in enumerate(zip((prefOri,prefDir),('Ori','Dir'))):
                    ax = plt.subplot(1,3,ind+2)
                    hasData = np.logical_not(np.isnan(pref))
                    p = np.copy(pref[hasData])
                    maxBin = 180 if ind==0 else 360
                    bins = np.arange(0,maxBin+1,45,dtype=float)
                    bins[1:] -= 22.5
                    p[p>bins[-1]] = 0
                    c,_ = np.histogram(p,bins)
                    bins[1:] += 22.5
                    ax.bar(bins[:-1],c)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                    ax.set_xlim([bins[0]-22.5,bins[-2]+22.5])
                    ax.set_xticks(bins[:-1])
                    ax.set_xlabel(oriOrDir)
                    ax.set_ylabel('Pref '+oriOrDir+' Count')

                                    
    def analyzeCheckerboard(self, units=None, protocol=None, trials=None, laser=False, latency=0.25, sdfSigma = 0.02, usePeakResp=True, plot=True, saveTag='', useCache=False):
        
        units, unitsYPos = self.getOrderedUnits(units)
        
        if protocol is None:
            protocol = self.getProtocolIndex('checkerboard')
        protocol = str(protocol)          
        p = self.visstimData[protocol]
        assert(set(p['bckgndDir'])=={0,180} and set(p['patchDir'])=={0,180} and 0 in p['bckgndSpeed'] and 0 in p['patchSpeed'])
        
        trialStartFrame = p['trialStartFrame']
        trialNumFrames = p['trialNumFrames'].astype(int)
        trialEndFrame = trialStartFrame+trialNumFrames[:trialStartFrame.size]
        lastFullTrial = np.where(trialEndFrame<p['frameSamples'].size)[0][-1]
        if trials is None:
            trials = np.arange(lastFullTrial+1)
        elif len(trials)<1:
            return
        else:
            trials = np.array(trials)
            trials = trials[trials<=lastFullTrial]
        
        if 'trialLaserPower' in p:
            if laser:
                trials = trials[p['trialLaserPower'][trials]>0]
            else:
                trials = trials[p['trialLaserPower'][trials]==0]
            if len(trials)<1:
                m = 'laser' if laser else 'non-laser'
                print('no '+m+' trials')
                return

        # find longest trial in entire presentation
        trialStartSamples = p['frameSamples'][trialStartFrame[np.arange(lastFullTrial+1)]]
        trialEndSamples = p['frameSamples'][trialEndFrame[np.arange(lastFullTrial+1)]]
        maxTrialDuration = max(trialEndSamples-trialStartSamples)
        
        # now pull out specified trial subset
        trialStartSamples = p['frameSamples'][trialStartFrame[trials]]
        trialEndSamples = p['frameSamples'][trialEndFrame[trials]]
        minInterTrialTime = p['interTrialInterval'][0]
        if 'laserPreFrames' in p:
            minInterTrialTime += (p['laserPreFrames']+p['laserPostFrames'])/p['frameRate']
        minInterTrialSamples = int(minInterTrialTime*self.sampleRate)
        latencySamples = int(latency*self.sampleRate)
        
        bckgndSpeed = np.concatenate((-p['bckgndSpeed'][:0:-1],p['bckgndSpeed']))
        patchSpeed = np.concatenate((-p['patchSpeed'][:0:-1],p['patchSpeed']))
        
        sdfSampInt = 0.001
        sdfTime = np.arange(0,2*minInterTrialTime+maxTrialDuration/self.sampleRate,sdfSampInt)
        
        if plot:
            fig = plt.figure(figsize=(15,3*len(units)),facecolor='w')
            gs = gridspec.GridSpec(len(units),3)
            
        for uInd,u in enumerate(units):
            spikes = self.units[str(u)]['times'][protocol]
            spikesPerTrial = self.findSpikesPerTrial(trialStartSamples+latencySamples,trialEndSamples,spikes)
            trialSpikeRate = spikesPerTrial/((trialEndSamples-trialStartSamples-latencySamples)/self.sampleRate)
            meanResp = np.full((patchSpeed.size,bckgndSpeed.size,p['patchSize'].size,p['patchElevation'].size),np.nan)
            peakResp = np.copy(meanResp)
            sdf = np.full((meanResp.shape+(sdfTime.size,)),np.nan)
            maxTrialTypeDur = np.zeros_like(meanResp)
            for pchSpeedInd,pchSpeed in enumerate(patchSpeed):
                pchDir = 180 if pchSpeed<0 else 0
                a = p['trialPatchDir'][trials]==pchDir
                b = p['trialPatchSpeed'][trials]==abs(pchSpeed)
                for bckSpeedInd,bckSpeed in enumerate(bckgndSpeed):
                    bckDir = 180 if bckSpeed<0 else 0
                    c = p['trialBckgndDir'][trials]==bckDir
                    d = p['trialBckgndSpeed'][trials]==abs(bckSpeed)
                    for pchSizeInd,pchSize in enumerate(p['patchSize']):
                        e = p['trialPatchSize'][trials]==pchSize
                        for pchElevInd,pchElev in enumerate(p['patchElevation']):
                            trialInd = p['trialPatchPos'][trials]==pchElev
                            for i in (a,b,c,d,e):
                                trialInd = np.logical_and(trialInd,i)
                            if any(trialInd):
                                meanResp[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd] = trialSpikeRate[trialInd].mean()
                                s,t = self.getSDF(spikes,trialStartSamples[trialInd]-minInterTrialSamples,2*minInterTrialSamples+max(trialEndSamples[trialInd]-trialStartSamples[trialInd]),sigma=sdfSigma,sampInt=sdfSampInt)
                                maxTrialTypeDur[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd] = t[-1]-2*minInterTrialTime
                                peakResp[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd] = s[np.logical_and(t>minInterTrialTime+latency,t<t[-1]-minInterTrialTime)].max()
                                sdf[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd,:s.size] = s
            
            # fill in resp for patch and bckgnd speeds not tested for every patch size and elevation
            for r in (meanResp,peakResp,sdf,maxTrialTypeDur):
                for pchSizeInd,_ in enumerate(p['patchSize']):
                    for pchElevInd,_ in enumerate(p['patchElevation']):
                        r[patchSpeed==0,:,pchSizeInd,pchElevInd] = r[patchSpeed==0,:,0,0]
                for pchSpeedInd,pchSpeed in enumerate(patchSpeed):
                    for bckSpeedInd,bckSpeed in enumerate(bckgndSpeed):
                        if pchSpeed==bckSpeed:
                            r[pchSpeedInd,bckSpeedInd] = r[patchSpeed==0,bckSpeedInd]
            
            # get spont rate and find best resp over all patch sizes and elevations
            spontTrials = np.logical_and(p['trialBckgndSpeed'][trials]==0,p['trialPatchSpeed'][trials]==0)
            if any(spontTrials):            
                if usePeakResp:
                    spontRateDist = self.getSDFNoise(spikes,trialStartSamples[spontTrials],max(trialEndSamples[spontTrials]-trialStartSamples[spontTrials]),sigma=sdfSigma,sampInt=sdfSampInt)
                else:
                    nreps = 100
                    spontRateDist = np.zeros(nreps)
                    spontTrialInd = np.where(spontTrials)[0]
                    for ind,_ in range(nreps):
                        spontRateDist[ind] = np.nanmean(trialSpikeRate[np.random.choice(spontTrialInd,spontTrialInd.size)])
                spontRateMean = spontRateDist.mean()
                spontRateStd = spontRateDist.std()
            else:
                spontRateMean = spontRateStd = np.nan
            
            respMat = peakResp.copy() if usePeakResp else meanResp.copy()
            patchResp = respMat[patchSpeed!=0,bckgndSpeed==0,:,:]
            bestPatchRespInd = np.unravel_index(np.argmax(patchResp),patchResp.shape)
            respMat = respMat[:,:,bestPatchRespInd[1],bestPatchRespInd[2]]
            
            # cache results
            self.units[str(u)]['checkerboard' + saveTag] = {'bckgndSpeed': bckgndSpeed,
                                                            'patchSpeed': patchSpeed,
                                                            'patchSize': p['patchSize'],
                                                            'patchElevation': p['patchElevation'],
                                                            'spontRateMean': spontRateMean,
                                                            'spontRateStd': spontRateStd,
                                                            'meanResp': meanResp,
                                                            'peakResp': peakResp,
                                                            'bestPatchRespInd': bestPatchRespInd,
                                                            'respMat': respMat,
                                                            'trials': trials,
                                                            '_sdf': sdf,
                                                            '_sdfTime': sdfTime}
            
            if plot:
                ax = fig.add_subplot(gs[uInd,0:2])
                spacing = 0.2
                sdfXMax = sdfTime[-1]
                sdfYMax = np.nanmax(sdf[:,:,bestPatchRespInd[1],bestPatchRespInd[2],:])
                x = 0
                y = 0
                for i,_ in enumerate(patchSpeed):
                    for j,_ in enumerate(bckgndSpeed):
                        ax.plot(x+sdfTime,y+sdf[i,j,bestPatchRespInd[1],bestPatchRespInd[2],:],color='k')
                        # show response window start and end
#                        ax.plot([x+minInterTrialTime+latency]*2,[y+0,y+sdfYMax],color='r')
#                        ax.plot([x+minInterTrialTime+maxTrialTypeDur[i,j,bestPatchRespInd[1],bestPatchRespInd[2]]]*2,[y+0,y+sdfYMax],color='r')
                        x += sdfXMax*(1+spacing)
                    x = 0
                    y += sdfYMax*(1+spacing)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='medium')
                ax.set_xticks([minInterTrialTime,sdfTime[-1]-2*minInterTrialTime])
                ax.set_xticklabels(['0',str(int(sdfTime[-1]-2*minInterTrialTime))+' s'])
                ax.set_yticks([0,int(sdfYMax)])
                ax.set_xlim([-sdfXMax*spacing,sdfXMax*(1+spacing)*bckgndSpeed.size])
                ax.set_ylim([-sdfYMax*spacing,sdfYMax*(1+spacing)*patchSpeed.size])
                ax.set_ylabel('Unit '+str(u), fontsize='large')
                
                ax = fig.add_subplot(gs[uInd,2])
                centerPoint = respMat[patchSpeed==0,bckgndSpeed==0][0] if not np.isnan(respMat[patchSpeed==0,bckgndSpeed==0][0]) else np.nanmedian(respMat)
                cLim = np.nanmax(abs(respMat-centerPoint))
                im = ax.imshow(respMat,cmap='bwr',clim=(centerPoint-cLim,centerPoint+cLim),interpolation='none',origin='lower')
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='medium')
                ax.set_xticks(range(bckgndSpeed.size))
                ax.set_xticklabels(bckgndSpeed)
                ax.set_yticks(range(patchSpeed.size))
                ax.set_yticklabels(patchSpeed)
                ax.set_xlabel('Background Speed')
                ax.set_ylabel('Patch Speed')
                ax.set_title('Spikes/s',fontsize='large')
                cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
                cb.set_ticks([math.ceil(centerPoint-cLim),round(centerPoint),math.floor(centerPoint+cLim)])
                cb.ax.tick_params(length=0,labelsize='large')
    
    def analyzeLoom(self, units=None, trials=None, protocol=None, saveTag='', sdfSigma=0.1, plot=True):
        units, unitsYPos = self.getOrderedUnits(units)
        if protocol is None:
            protocol = self.getProtocolIndex('loom')        
#            if protocol is None:
#                print 'Could not find loom protocol'
#                return
        
        v = self.visstimData[str(protocol)]
        if trials is None:
            trials = np.arange(v['stimStartFrames'].size-1)
        else:
            trials = np.array(trials)
            numFullTrials = v['stimStartFrames'].size - 1
            trials = trials[trials<numFullTrials]
        
        if len(trials) == 0:            
            return
        
        trialStarts = v['frameSamples'][v['stimStartFrames'][trials]]
        trialFrameLengths = (np.diff(v['stimStartFrames']) - v['interTrialInterval'])[trials]
        trialEnds = v['frameSamples'][v['stimStartFrames'][trials] + trialFrameLengths]
        trialSampleLengths = trialEnds - trialStarts
        
        trialPos = np.copy(v['posHistory'][trials])/v['pixelsPerDeg']
        trialLV = v['lvHistory'][trials]
        trialColor = v['colorHistory'][trials]        
        trialParams = np.stack((trialLV, trialPos[:, 0], trialPos[:, 1], trialColor)).T        
#        trialParams = np.round(trialParams).astype(int)
        trialConditions = np.array(list(itertools.product(np.unique(trialLV), np.unique(trialPos[:, 0]), np.unique(trialPos[:, 1]), np.unique(trialColor))))
        lvRatio = np.unique(trialLV)
        
        longTrials = np.where(trialLV==lvRatio.max())[0]
            
        sdfSamples = trialSampleLengths.max() + 10
        sdfSampInt = 0.001

        #figure out time to collision for LV values
        time = np.arange(0, 360000, sdfSampInt*1000)
        timeToCollision = []
        for lv in v['lvratio']:        
            halfTheta = np.arctan(lv/time)[::-1] * (180./np.pi)
            start = np.where(halfTheta>=v['startRadius'])[0][0]
            timeToCollision.append(360000 - start)
#            plt.plot(halfTheta[360000-9168:])
            
        sdfPadding = int(round(0.4/sdfSampInt))
        
        for uindex, unit in enumerate(units):
            spikes = self.units[str(unit)]['times'][str(protocol)]
            sdf = np.full((trialConditions.shape[0],int(round(sdfSamples/self.sampleRate/sdfSampInt))),np.nan)
            peakTimeFromCollision = np.full(trialConditions.shape[0], np.nan)
            peakResp = np.full(trialConditions.shape[0], np.nan)

            for ic, c in enumerate(trialConditions):
                condTrials = np.array([i for i,t in enumerate(trialParams) if all(t == c)])
                if condTrials.size>0:
                    thissdf, sdfTime = self.getSDF(spikes, trialEnds[condTrials] - np.median(trialSampleLengths[condTrials]), np.median(trialSampleLengths[condTrials]), sigma=sdfSigma)                    
                    collision = timeToCollision[int(np.where(np.unique(trialLV)==c[0])[0])]
                    thissdf = thissdf[:collision+sdfPadding]
                    sdf[ic, -thissdf.size:] = thissdf
                    peakTimeFromCollision[ic] = sdf.shape[1] - np.nanargmax(sdf[ic]) - sdfPadding
                    peakResp[ic] = np.nanmax(sdf[ic])
                
            # get spont rate (defined as the first second of activity during the longest trial condition)
            nreps = 100
            spontRateDist = np.zeros(nreps)
            spontRateMean = None
            spontRateStd = None
            if len(longTrials)>0:
                trialReps = trials.size//trialConditions.shape[0]
                nsamps = np.max(trialSampleLengths[longTrials])
                for ind in range(nreps):
                    shuffledLongTrials = np.random.choice(longTrials,trialReps)
                    longsdf, sdfTime = self.getSDF(spikes, trialEnds[shuffledLongTrials]-nsamps, nsamps, sigma=sdfSigma)
                    spontRateDist[ind] = np.nanmean(np.nanmax(longsdf[:1000]))
                spontRateMean = spontRateDist.mean()
                spontRateStd = spontRateDist.std()
                
            bestCondition = np.argmax(peakResp)
            bestLVs = [i for i,t in enumerate(trialConditions) if all(t[1:] == trialConditions[bestCondition][1:])]
            bestCondPeakResps = peakResp[bestLVs]
            bestCondPeakTimes = peakTimeFromCollision[bestLVs]
                
#            #get mean responses for each LV value at best parameters
            
#            peakTimeFromCollision = np.full(numLVs, np.nan)
#            peakResp = np.full(numLVs, np.nan)
#            bestCond = np.copy(trialConditions[np.unravel_index(np.nanargmax(sdf), sdf.shape)[0]])
#            lvCurves = []
#            for ilv, lv in enumerate(np.unique(trialLV)):
#                bestCond[0] = lv
#                condition = np.array([i for i,t in enumerate(trialConditions) if all(t == bestCond)])
#                lvCurves.append(sdf[condition].T)
#                peakTimeFromCollision[ilv] = sdf.shape[1] - np.nanargmax(sdf[condition]) - sdfPadding
#                peakResp = np.nanmax(sdf[condition])
            
            if plot:
                alphas = np.linspace(0.2, 1, lvRatio.size)
                fig = plt.figure()
                gs = gridspec.GridSpec(v['ypos'].size, v['xpos'].size)
                colors = ['b', 'r']
                for ic, c in enumerate(trialConditions):
                    xaxis = int(np.where(np.unique(trialPos[:, 0]) == c[1])[0])
                    yaxis = abs(int(np.where(np.unique(trialPos[:, 1]) == c[2])[0])-1)
                    linecolor = colors[int(np.where(np.unique(trialColor) == c[3])[0])]
                    alpha = alphas[int(np.where(np.unique(trialLV) == c[0])[0])]
                    ax = fig.add_subplot(gs[yaxis, xaxis])
                    ax.plot(sdf[ic], color=linecolor, alpha=alpha)
                    ax.set_ylim([0, np.nanmax(sdf)])
                    ylims = ax.get_ylim()
                    ax.plot([sdf.shape[1]-sdfPadding, sdf.shape[1]-sdfPadding], [ylims[0], ylims[1]], 'k--')
                    formatFigure(fig, ax)
            
            # cache results
            self.units[str(unit)]['loom' + saveTag] = {'trialConditions': trialConditions,
                                                        '_sdf' : sdf,
                                                        '_sdfTime': sdfTime,
                                                        'trials': trials,
                                                        'spontRateMean': spontRateMean,
                                                        'spontRateStd': spontRateStd,
                                                        'peakTimeFromCollision': peakTimeFromCollision,
                                                        'peakResp': peakResp,
                                                        'bestConditionPeaks': bestCondPeakResps,
                                                        'bestConditionPeakTimes': bestCondPeakTimes}
                                                            
            
    def analyzeSpots(self, units=None, protocol = None, plot=True, trials=None, useCache=False, saveTag=''):
         
        units, unitsYPos = self.getOrderedUnits(units) 
         
        if protocol is None:
            protocol = self.getProtocolIndex('spots')
        protocol = str(protocol)
         
        if plot:        
            plt.figure(figsize = (10, 4*len(units)))
            gs = gridspec.GridSpec(2*len(units), 4)                        
         
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
        spotPos = np.unique(trialPos)
        spotColor = np.unique(trialColor)
        spotSize = np.unique(trialSize)
        spotDir = np.unique(trialDir)
        spotSpeed = np.unique(trialSpeed)
   
        horzTrials = np.logical_or(trialDir==0, trialDir==180)
        vertTrials = np.logical_or(trialDir==270, trialDir==90)
        azimuths = np.unique(trialPos[vertTrials])
        elevs = np.unique(trialPos[horzTrials])
        
        numTrialTypes = spotSpeed.size*spotSize.size*(2*azimuths.size+elevs.size)*spotColor.size
        maxTrialsPerType = math.ceil(trials.size/numTrialTypes)
        resp = np.full((spotSpeed.size,spotSize.size,spotDir.size,spotPos.size,spotColor.size,maxTrialsPerType),np.nan)
        
        if plot:
            plt.figure(figsize=(10,4*len(units)),facecolor='w')
            gs = gridspec.GridSpec(len(units),3)        
        
        for uindex, unit in enumerate(units):
            if ('spotResponse' + saveTag) in self.units[str(unit)] and useCache:
                responseDict = self.units[str(unit)]['spotResponse' + saveTag]['spot_responseDict']
                spotRF = responseDict['spotRF']
                spontRate = responseDict['spontRate']
            else:
                self.units[str(unit)]['spotResponse' + saveTag] = {}
                spikes = self.units[str(unit)]['times'][str(protocol)]
         
                # get RF         
                spikesPerTrial = self.findSpikesPerTrial(trialStartSamples, trialEndSamples, spikes)
                trialSpikeRate = spikesPerTrial/((1/self.visstimData[str(protocol)]['frameRate'])*trialDuration)
 
                azimuthSpikeRate = np.zeros(azimuths.size)        
                elevSpikeRate = np.zeros(elevs.size)
                azimuthTrialCount = np.zeros(azimuths.size)        
                elevTrialCount = np.zeros(elevs.size)
                for trial in range(trialPos.size):
                    if horzTrials[trial]:
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
                spontRate = itiRate.mean()
                sdfSigma = 0.1
                sdf,_ = self.getSDF(spikes,frameSamples[interTrialStarts],max(frameSamples[interTrialEnds]-frameSamples[interTrialStarts]),sigma=sdfSigma)
                peakSpontRate = sdf.max()
                 
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
                    responseDict[param]['tuningCurve'] = {}
                    responseDict[param]['tuningCurve']['paramValues'] = possibleValues
                    responseDict[param]['tuningCurve']['meanResponse'] = meanResponse
                    responseDict[param]['tuningCurve']['sem'] = semResponse                     
                     
                x,y = np.meshgrid(azimuthSpikeRate,elevSpikeRate)
                spotRF = np.sqrt(abs(x*y))*np.sign(x+y)
                responseDict['spontRate'] = spontRate
                responseDict['spotRF'] = spotRF                
                self.units[str(unit)]['spotResponse' + saveTag]['spot_responseDict'] = responseDict
                
                # speed x size x dir x pos x color matrix for mean and peak resp
                peakResp = np.full(resp.shape[:-1],np.nan)
                for speedInd,speed in enumerate(spotSpeed):
                    speedTrials = trialSpeed==speed
                    for sizeInd,size in enumerate(spotSize):
                        sizeTrials = trialSize==size
                        for dirInd,direction in enumerate(spotDir):
                            dirTrials = trialDir==direction
                            for posInd,pos in enumerate(spotPos):
                                posTrials = trialPos==pos
                                for colorInd,color in enumerate(spotColor):
                                    trialInd = trialColor==color
                                    for i in (speedTrials,sizeTrials,dirTrials,posTrials):
                                        trialInd = np.logical_and(trialInd,i)
                                    if any(trialInd):
                                        resp[speedInd,sizeInd,dirInd,posInd,colorInd,:np.count_nonzero(trialInd)] = trialSpikeRate[trialInd]
                                        sdf,_ = self.getSDF(spikes,trialStartSamples[trialInd],max(trialEndSamples[trialInd]-trialStartSamples[trialInd]),sigma=sdfSigma)
                                        peakResp[speedInd,sizeInd,dirInd,posInd,colorInd] = sdf.max()
                meanResp = np.nanmean(resp,axis=5)
                resp[:] = np.nan
                 
            if plot:
                axInd = 0
                for r,spRate in zip((meanResp,peakResp),(spontRate,peakSpontRate)):
                    for m in ('mean','max'):
                        # speed vs size
                        ax = plt.subplot(gs[uindex*2,axInd])
                        if m=='mean':
                            speedSizeResp = np.nanmean(np.nanmean(np.nanmean(r,axis=4),axis=3),axis=2)
                        else:
                            speedSizeResp = np.nanmax(np.nanmax(np.nanmax(r,axis=4),axis=3),axis=2)
                        centerPoint = spRate if not np.isnan(spRate) else np.nanmedian(speedSizeResp)
                        cLim = np.nanmax(abs(speedSizeResp-centerPoint))
                        plt.imshow(speedSizeResp,cmap='bwr',clim=(centerPoint-cLim,centerPoint+cLim),interpolation='none',origin='lower')
                        ax.spines['left'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                        ax.set_xticks(range(spotSize.size))
                        ax.set_xticklabels([])
                        ax.set_yticks(range(spotSpeed.size))
                        ax.set_yticklabels([])
                        if axInd==0:
                            ax.set_yticklabels(spotSpeed)
                            ylab = 'Unit '+str(unit)+'\nSpot Speed' if uindex==0 else 'Unit '+str(unit)
                            ax.set_ylabel(ylab,fontsize='x-small')
                        if uindex==len(units)-1 and axInd==0:
                            ax.set_xticklabels(spotSize)
                            ax.set_xlabel('Spot Size',fontsize='x-small')
                        if uindex==0:
                            if axInd==0:
                                title = 'meanResp\nmean'
                            elif axInd==1:
                                title = 'meanResp\nmax'
                            elif axInd==2:
                                title = 'peakResp\nmean'
                            else:
                                title = 'peakResp\nmax'
                            ax.set_title(title,fontsize='x-small')
                        cb = plt.colorbar(fraction=0.05,pad=0.04,shrink=0.5)
                        cb.set_ticks([math.ceil(centerPoint-cLim),round(centerPoint),math.floor(centerPoint+cLim)])
                        cb.ax.tick_params(length=0,labelsize='xx-small')
                        
                        # direction
                        ax = plt.subplot(gs[uindex*2+1,axInd])
                        if m=='mean':
                            dirResp = np.nanmean(np.nanmean(np.nanmean(np.nanmean(r,axis=4),axis=3),axis=1),axis=0)
                        else:
                            dirResp = np.nanmax(np.nanmax(np.nanmax(np.nanmax(r,axis=4),axis=3),axis=1),axis=0)
                        plt.plot(spotDir,dirResp)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                        ax.set_xticks(spotDir)
                        ax.set_xticklabels([])
                        ax.set_xlim([spotDir[0],spotDir[-1]])
                        ax.set_ylim([0,dirResp.max()+0.5])
                        if axInd==0:
                            ylab = 'Unit '+str(unit)+'\nSpikes/s' if uindex==0 else 'Unit '+str(unit)
                            ax.set_ylabel(ylab,fontsize='x-small')
                        if uindex==len(units)-1 and axInd==0:
                            ax.set_xlabel('Direction',fontsize='x-small')
                            ax.set_xticklabels(spotDir)
                        
                        axInd += 1
    
    
    def parseRunning(self, protocol, runThresh = 5.0, statThresh = 1.0, trialStarts = None, trialEnds = None, wheelDownsampleFactor = 500.0):
        if trialStarts is None:
            trialStarts, trialEnds = self.getTrialStartsEnds(protocol)
        runningTrials = []
        stationaryTrials = []
        speeds = []
        if trialStarts is not None:
            if 'running' in self.behaviorData[str(protocol)]:
                wheelData = -self.behaviorData[str(protocol)]['running']
                for trial in range(trialStarts.size):
                    trialSpeed = np.mean(wheelData[int(round(trialStarts[trial]/wheelDownsampleFactor)):int(round(trialEnds[trial]/wheelDownsampleFactor))])
                    speeds.append(trialSpeed)
                    if trialSpeed >= runThresh:
                        runningTrials.append(trial)
                    elif trialSpeed <= statThresh:
                        stationaryTrials.append(trial)
            else:
                runningTrials = list(range(trialStarts.size))
        return stationaryTrials, runningTrials, np.array(speeds)
    
    
    def analyzeRunning(self, protocol=None, units=None, wheelSampleRate=60.0, smoothKernel=0.5, plot=True):
        
        units, unitsYPos = self.getOrderedUnits(units)
        if protocol is None:
            protocol = range(len(self.kwdFileList))
        elif not isinstance(protocol,list):
            protocol = [protocol]
        
        if plot:
            plt.figure(figsize=(10,3*len(units)),facecolor='w')
            gs = gridspec.GridSpec(len(units),1)
        
        speedBins = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        kernelWidth=smoothKernel*wheelSampleRate
        wheelBinSize = self.sampleRate/wheelSampleRate
        prefSpeeds = []
        tuningCurves = []
        for uindex, u in enumerate(units):
            fr_binned = [[] for i in xrange(len(speedBins))]
            for pro in protocol:            
                spikes = self.units[str(u)]['times'][str(pro)]
                wd = -self.behaviorData[str(pro)]['running']
                fh, _ = np.histogram(spikes, np.arange(0, (wd.size+1)*int(wheelBinSize), int(wheelBinSize)))
                fh *= int(wheelSampleRate)
                frc = np.convolve(fh, np.ones(int(round(kernelWidth))),'same')/kernelWidth
                
                binnedSpeed = np.digitize(wd, speedBins)
                for sbin, _ in enumerate(speedBins):
                    binIndices = binnedSpeed==sbin
                    fr_binned[sbin].append(frc[binIndices])

            frMean = np.array([np.mean(np.concatenate(b)) for b in fr_binned])
            frSTD = np.array([np.std(np.concatenate(b)) for b in fr_binned])
            if plot:
                ax = plt.subplot(gs[uindex, 0])
                ax.plot(speedBins, frMean)
                plt.fill_between(speedBins, frMean+frSTD, frMean-frSTD, alpha=0.3)
            
            prefSpeeds.append(speedBins[np.nanargmax(frMean)])
            tuningCurves.append(frMean)

        if plot:
            fig = plt.figure(facecolor='w')
            ax  = fig.add_subplot(111)
            ax.hist(np.array(prefSpeeds), speedBins)
            ax.set_xscale('log', basex=2)
            self.plotSDF1Axis(np.array(tuningCurves), np.array(speedBins)) 
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(111)
            ax.plot(unitsYPos, prefSpeeds, 'ko', alpha=0.5)

        return tuningCurves

        
    def analyzeSaccades(self,units=None,protocol=0,preTime=1,postTime=1,analysisWindow=[0,0.2],sdfSigma=0.01,sdfSampInt=0.001,plot=True):
        units,_ = self.getOrderedUnits(units)        
        protocol = str(protocol)
        if not hasattr(self,'behaviorData') or 'eyeTracking' not in self.behaviorData[protocol]:
            print('no eye tracking data for protocol '+protocol)
            return
        
        eyeTrackSamples = self.behaviorData[protocol]['eyeTracking']['samples']
        pupilX = self.behaviorData[protocol]['eyeTracking']['pupilX']
        negSaccades = self.behaviorData[protocol]['eyeTracking']['negSaccades']
        negSaccades = negSaccades[negSaccades<eyeTrackSamples.size]
        negSaccades = negSaccades[~np.isnan(pupilX[negSaccades])]
        posSaccades = self.behaviorData[protocol]['eyeTracking']['posSaccades']
        posSaccades = posSaccades[posSaccades<eyeTrackSamples.size]
        posSaccades = posSaccades[~np.isnan(pupilX[posSaccades])]
        allSaccades = np.sort(np.concatenate((negSaccades,posSaccades)))
        saccadeRate = allSaccades.size/(eyeTrackSamples[-1]-eyeTrackSamples[0])*self.sampleRate
        
        # get average saccade and saccade amplitudes
        preFrames = int(preTime*60)
        postFrames = int(postTime*60)
        saccadeTime = np.arange(-preTime,postTime,1/60.0)
        avgSaccade = np.full((2,preFrames+postFrames),np.nan)
        for j,saccades in enumerate((negSaccades,posSaccades)):
            if saccades.size>0:
                x = np.full((saccades.size,preFrames+postFrames),np.nan)
                for i,s in enumerate(saccades):
                    if s-preFrames>=0 and s+postFrames<pupilX.size:
                        x[i] = pupilX[s-preFrames:s+postFrames]
                avgSaccade[j] = np.nanmean(x,axis=0)
        negAmp = np.array([np.nanmin(pupilX[s:s+6])-np.nanmedian(pupilX[s-6:s-2]) for s in negSaccades])
        posAmp = np.array([np.nanmax(pupilX[s:s+6])-np.nanmedian(pupilX[s-6:s-2]) for s in posSaccades])
                
        # get spike count, sdf, and latency
        winDur = analysisWindow[1]-analysisWindow[0]
        baseWindow = [analysisWindow[0]-2*winDur,analysisWindow[0]-winDur]
        preSaccadeSpikeCount = [[[] for j in range(3)] for i in units]
        postSaccadeSpikeCount = [[[] for j in range(3)] for i in units]
        hasResp = np.zeros((len(units),3),dtype=bool)
        respPolarity = np.zeros((len(units),3),dtype=int)
        preSamples = int(preTime*self.sampleRate)
        postSamples = int(postTime*self.sampleRate)
        sdf = np.full((len(units),3,int(round((preSamples+postSamples)/self.sampleRate/sdfSampInt))),np.nan)
        peakRate = np.full((len(units),3),np.nan)
        latency = peakRate.copy()
        for i,u in enumerate(units):
            spikes = self.units[str(u)]['times'][protocol]
            for j,saccades in enumerate((negSaccades,posSaccades,allSaccades)):
                if saccades.size>0:
                    saccadeSamples = eyeTrackSamples[saccades]
                    # get pre- and post-saccade spike counts and test for significant difference
                    for s in saccadeSamples:
                        preSaccadeSpikeCount[i][j].append(np.count_nonzero(np.logical_and(spikes>s+int(baseWindow[0]*self.sampleRate),spikes<s+int(baseWindow[1]*self.sampleRate))))
                        postSaccadeSpikeCount[i][j].append(np.count_nonzero(np.logical_and(spikes>s+int(analysisWindow[0]*self.sampleRate),spikes<s+int(analysisWindow[1]*self.sampleRate))))
                    _,pval = scipy.stats.wilcoxon(preSaccadeSpikeCount[i][j],postSaccadeSpikeCount[i][j])
                    respPolarity[i,j] = 1 if sum(postSaccadeSpikeCount[i][j])>sum(preSaccadeSpikeCount[i][j]) else -1
                    # get sdf, peak resp, and latency
                    sdf[i,j],sdfTime = self.getSDF(spikes,saccadeSamples-preSamples,preSamples+postSamples,sigma=sdfSigma,sampInt=sdfSampInt)
                    inAnalysisWindow = np.logical_and(sdfTime>preTime+analysisWindow[0],sdfTime<preTime+analysisWindow[1])
                    peakInd = np.argmax(sdf[i,j,inAnalysisWindow]) if respPolarity[i,j]>0 else np.argmin(sdf[i,j,inAnalysisWindow])
                    peakInd += np.where(inAnalysisWindow)[0][0]
                    peakRate[i,j] = sdf[i,j,peakInd]
                    spontRateDist = self.getSDFNoise(spikes,saccadeSamples+baseWindow[0],int(winDur*self.sampleRate),sigma=sdfSigma,sampInt=sdfSampInt)
                    spontRateMean = spontRateDist.mean()
                    spontRateStd = spontRateDist.std()
                    latencyThresh = spontRateMean+5*spontRateStd if respPolarity[i,j]>0 else spontRateMean-5*spontRateStd
                    if pval<0.05:
                        if (respPolarity[i,j]>0 and peakRate[i,j]>latencyThresh) or (respPolarity[i,j]<0 and peakRate[i,j]<latencyThresh):
                            latencyInd = np.where(sdf[i,j,:peakInd]<latencyThresh)[0][-1]+1 if respPolarity[i,j] else np.where(sdf[i,j,:peakInd]>latencyThresh)[0][-1]+1
                            latency[i,j] = latencyInd*sdfSampInt-preTime
                            hasResp[i,j] = True
        
        if not plot:
            return {'pupilX':pupilX, 'saccadeRate':saccadeRate, 'negAmp':negAmp, 'posAmp':posAmp, 
                    'preSaccadeSpikeCount':preSaccadeSpikeCount, 'postSaccadeSpikeCount':postSaccadeSpikeCount,
                    'hasResp':hasResp, 'respPolarity':respPolarity, 'peakRate':peakRate, 'latency':latency}
        
        # pupil position histogram
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        pupilX -= np.nanmedian(pupilX)
        ax.hist(pupilX[~np.isnan(pupilX)],bins=np.arange(np.nanmin(pupilX)-1,np.nanmax(pupilX)+1),color='k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlabel('Horizontal Pupil Position (degrees)')
        ax.set_ylabel('# Frames')
        
        # saccdade amplitude histogram
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        bins = np.arange(30)
        negCount,_ = np.histogram(-negAmp[~np.isnan(negAmp)],bins)
        plt.bar(bins[:-1],-negCount,color='b')
        posCount,_ = np.histogram(posAmp[~np.isnan(posAmp)],bins)
        plt.bar(bins[:-1],posCount,color='r')
        ax.tick_params(direction='out',top=False,right=False)
        maxCount = max(negCount.max(),posCount.max())
        ax.set_ylim((-maxCount,maxCount))
        ax.set_yticks((-maxCount,maxCount))
        ax.set_yticklabels((maxCount,maxCount))
        ax.set_xlabel('Saccade Amplitude (degrees)')
        ax.set_ylabel('Count')
        
        # sdfs and mean saccade
        fig = plt.figure(facecolor='w')
        saccadeColor = ('b','r','k')
        ax = fig.add_subplot(1,1,1)
        sdfMax = np.nanmax(sdf)
        ymax = 1.2*sdfMax
        yoffset = 0
        ax.plot([0,0],[0,ymax*(len(units)+2.5)],color='0.5')
        ax.plot([analysisWindow[0]]*2,[0,ymax*(len(units)+2.5)],'k:')
        ax.plot([analysisWindow[1]]*2,[0,ymax*(len(units)+2.5)],'k:')
        for i,u in zip(reversed(range(len(units))),reversed(units)):
            for j,clr in enumerate(saccadeColor):
                ax.plot(sdfTime-preTime,sdf[i,j]+yoffset,color=clr)
            ax.text(-preTime-0.02*(preTime+postTime),ymax/2+yoffset,str(u),fontsize='xx-small',horizontalalignment='right',verticalalignment='center')
            yoffset += ymax
        for j,clr in enumerate(saccadeColor[:2]):
            s = avgSaccade[j].copy()
            s -= s.min()
            s *= 1.5*ymax/s.max()
            ax.plot(saccadeTime,s+yoffset+0.5*ymax,color= clr)
        for side in ('left','bottom','right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,left=False,labelleft=False,labelright=True,labelsize='x-small')
        ax.set_xlim((-preTime,postTime))
        ax.set_ylim((0,ymax*(len(units)+2.5)))
        ax.set_yticks((0,round(sdfMax)))
        
        # mean sdf
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        meanSDF = np.nanmean(sdf[:,2],axis=0)
        ax.plot(sdfTime-preTime,meanSDF,'k')
        for side in ('left','bottom','right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,left=False,labelleft=False,labelright=True)
        ax.set_xlim((-preTime,postTime))
        ax.set_ylim((0,1.1*meanSDF.max()))
        ax.set_yticks((0,round(meanSDF.max())))
        
        # spike rate scatter plot
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        maxRate = 1.1*max(np.mean(count)/winDur for unit in postSaccadeSpikeCount for count in unit)
        ax.plot([0,maxRate],[0,maxRate],'k:')
        for i,(pre,post) in enumerate(zip(preSaccadeSpikeCount,postSaccadeSpikeCount)):
            for j,clr in enumerate(saccadeColor):
                faceColor = clr if hasResp[i,j] else 'none'
                ax.plot(np.mean(pre[j])/winDur,np.mean(post[j])/winDur,'o',mec=clr,mfc=faceColor)
        ax.set_xlim((0,maxRate))
        ax.set_ylim((0,maxRate))
        ax.set_aspect('equal')
        ax.tick_params(direction='out',top=False,right=False)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.set_xlabel('Pre-Saccade Spikes/s')
        ax.set_ylabel('Post-Saccade Spikes/s')
        
        # saccade direction comparison
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        maxDiff = 1.1*max(abs(np.mean(post[j])-np.mean(pre[j]))/winDur for pre,post in zip(preSaccadeSpikeCount,postSaccadeSpikeCount) for j in (0,1))
        ax.plot([-maxDiff,maxDiff],[-maxDiff,maxDiff],'k:')
        for i,(pre,post) in enumerate(zip(preSaccadeSpikeCount,postSaccadeSpikeCount)):
            if hasResp[i,:2].any():
                ax.plot((np.mean(post[0])-np.mean(pre[0]))/winDur,(np.mean(post[1])-np.mean(pre[1]))/winDur,'ko')
        ax.set_xlim((-maxDiff,maxDiff))
        ax.set_ylim((-maxDiff,maxDiff))
        ax.set_aspect('equal')
        ax.tick_params(direction='out',top=False,right=False)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.set_xlabel('Neg Saccades Spikes/s')
        ax.set_ylabel('Pos Saccades Spikes/s')
        
        # latency
        if len(units)>1:
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            lat = latency[hasResp[:,-1],-1]*1000
            ax.hist(lat[~np.isnan(lat)],bins=np.arange(np.nanmin(lat)-25,np.nanmax(lat)+50,25),color='k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlabel('Latency')
            ax.set_ylabel('Count')
            
            
    def analyzeOKR(self,protocolName='gratings',smoothPts=3,plot=True):
        if protocolName not in ('gratings','checkerboard'):
            print('protocolName must be gratings or checkerboard')
            return
        protocol = self.getProtocolIndex(protocolName)
        if protocol is None:
            expDate,anmID = self.getExperimentInfo()
            print('no '+protocolName+' for '+expDate+'_'+anmID)
            return
        else:
            protocol = str(protocol)
        if not hasattr(self,'behaviorData') or 'eyeTracking' not in self.behaviorData[protocol]:
            print('no eye tracking data for '+protocolName)
            return
        
        p = self.visstimData[protocol]        
        if protocolName=='gratings':
            trialStartFrames = p['stimStartFrames']
            trialEndFrames = trialStartFrames+p['stimTime']
            trials = np.logical_and(trialEndFrames<p['frameSamples'].size,np.logical_and(p['stimulusHistory_contrast'][:trialStartFrames.size]>0,p['stimulusHistory_ori'][:trialStartFrames.size]==0))
            trialStartSamples = p['frameSamples'][trialStartFrames[trials]]
            trialEndSamples = p['frameSamples'][trialEndFrames[trials]]   
            trialX = p['stimulusHistory_sf'][:trialStartFrames.size][trials]
            trialY = p['stimulusHistory_tf'][:trialStartFrames.size][trials]
            xparamName = 'sf'
            yparamName = 'tf'
        elif protocolName=='checkerboard':
            trialStartFrames = p['trialStartFrame']
            trialEndFrames = trialStartFrames+p['trialNumFrames'].astype(int)
            trials = trialEndFrames<p['frameSamples'].size
            trialStartSamples = p['frameSamples'][trialStartFrames[trials]]
            trialEndSamples = p['frameSamples'][trialEndFrames[trials]]
            trialX = p['trialBckgndSpeed'][trials].copy()
            trialX[p['trialBckgndDir'][trials]==180] *= -1
            trialY = p['trialPatchSpeed'][trials].copy()
            trialY[p['trialPatchDir'][trials]==180] *= -1
            xparamName = 'background speed'
            yparamName = 'patch speed'
        
        eyeTrackSamples = self.behaviorData[protocol]['eyeTracking']['samples']
        pupilX = self.behaviorData[protocol]['eyeTracking']['pupilX'][0:eyeTrackSamples.size]
        pupilVel = -np.diff(pupilX)/np.diff(eyeTrackSamples)*self.sampleRate
        negSaccades = self.behaviorData[protocol]['eyeTracking']['negSaccades']
        negSaccades = negSaccades[negSaccades<eyeTrackSamples.size]
        negSaccades = negSaccades[~np.isnan(pupilX[negSaccades])]
        posSaccades = self.behaviorData[protocol]['eyeTracking']['posSaccades']
        posSaccades = posSaccades[posSaccades<eyeTrackSamples.size]
        posSaccades = posSaccades[~np.isnan(pupilX[posSaccades])]
        allSaccades = np.sort(np.concatenate((negSaccades,posSaccades)))
        for saccade in allSaccades:
            pupilVel[int(saccade-6):int(saccade+12)] = np.nan
        
        n = smoothPts//2
        pupilVelSmoothed = np.convolve(pupilVel,np.ones(smoothPts)/smoothPts,mode='same')
        pupilVelSmoothed[:n] = pupilVel[:n].mean()
        pupilVelSmoothed[-n:] = pupilVel[-n:].mean()
        
        xparam = np.unique(trialX)
        yparam = np.unique(trialY)
        meanPupilVel = np.zeros((yparam.size,xparam.size))
        for i,y in enumerate(yparam):
            for j,x in enumerate(xparam):
                trialInd = np.where(np.logical_and(trialY==y,trialX==x))[0]
                v = np.zeros(trialInd.size)
                for n,trial in enumerate(trialInd):
                    v[n] = np.nanmean(pupilVelSmoothed[np.argmin(abs(eyeTrackSamples-trialStartSamples[trial])):np.argmin(abs(eyeTrackSamples-trialEndSamples[trial]))])
                meanPupilVel[i,j] = np.nanmean(v)
        meanPupilVel[np.isnan(meanPupilVel)] = 0
        if protocolName=='checkerboard':
            for i in range(yparam.size):
                for j in range(xparam.size):
                    if i==j:
                        meanPupilVel[i,j] = meanPupilVel[yparam==0,j]
        
        if protocolName=='gratings':
            stimSpeed = yparam[:,None]/xparam
            stimSpeedLabel = 'grating'
        else:
            stimSpeed = np.tile(xparam,(xparam.size,1))
            stimSpeedLabel = 'background'
        okrGain = meanPupilVel/stimSpeed
        okrGain[okrGain<0] = 0
        okrGain[np.isinf(okrGain)] = 0
        okrGain[np.isnan(okrGain)] = 0
        
        if not plot:
            return {'xparam':xparam, 'yparam':yparam, 'meanPupilVel':meanPupilVel, 'stimSpeed':stimSpeed, 'okrGain':okrGain}
                
        fig = plt.figure(facecolor='w')
        gs = gridspec.GridSpec(2,2)
        ax = fig.add_subplot(gs[0,0])
        maxVel = np.absolute(meanPupilVel).max()
        if protocolName=='gratings':
            clim = (0,maxVel)
            cmap = 'gray'
        else:
            clim = (-maxVel,maxVel)
            cmap = 'bwr'
        im = ax.imshow(meanPupilVel,clim=clim,cmap=cmap,origin='lower',interpolation='none')
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks([0,xparam.size-1])
        ax.set_yticks([0,yparam.size-1])
        if protocolName=='gratings':
            ax.set_xticks([0,xparam.size-1])
            ax.set_yticks([0,yparam.size-1])
            ax.set_xticklabels(xparam[[0,-1]])
            ax.set_yticklabels(yparam[[0,-1]])
        else:
            ax.set_xticks([0,xparam.size//2,xparam.size-1])
            ax.set_yticks([0,yparam.size//2,yparam.size-1])
            ax.set_xticklabels([xparam[0],0,xparam[-1]])
            ax.set_yticklabels([yparam[0],0,yparam[-1]])
        ax.set_xlabel(xparamName)
        ax.set_ylabel(yparamName)
        ax.set_title('pupil speed (deg/s)')
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.set_ticks(clim)
        
        ax = fig.add_subplot(gs[1,0])
        im = ax.imshow(okrGain,clim=[0,okrGain.max()],cmap='gray',origin='lower',interpolation='none')
        ax.tick_params(direction='out',top=False,right=False)
        if protocolName=='gratings':
            ax.set_xticks([0,xparam.size-1])
            ax.set_yticks([0,yparam.size-1])
            ax.set_xticklabels(xparam[[0,-1]])
            ax.set_yticklabels(yparam[[0,-1]])
        else:
            ax.set_xticks([0,xparam.size//2,xparam.size-1])
            ax.set_yticks([0,yparam.size//2,yparam.size-1])
            ax.set_xticklabels([xparam[0],0,xparam[-1]])
            ax.set_yticklabels([yparam[0],0,yparam[-1]])
        ax.set_xlabel(xparamName)
        ax.set_ylabel(yparamName)
        ax.set_title('pupil speed / '+stimSpeedLabel+' speed')
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.set_ticks([0,okrGain.max()])
        
        ax = fig.add_subplot(gs[1,1])
        if protocolName=='gratings':
            ax.semilogx(stimSpeed.ravel(),okrGain.ravel(),'ko')
            ax.set_xlim([0.9,1000])
        else:
            ax.plot(stimSpeed.ravel(),okrGain.ravel(),'ko')
            ax.set_xlim([-100,100])
        for side in ('top','right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(which='both',direction='out',top=False,right=False)
        ax.set_ylim([0,1.15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlabel(stimSpeedLabel+' speed')
        ax.set_ylabel('OKR gain')
        
        plt.tight_layout()

        
    def plotISIHist(self,units=None,protocol=None,binWidth=0.001,maxInterval=0.02):
        units,unitsYPos = self.getOrderedUnits(units)
        if protocol is None:
            protocol = range(len(self.kwdFileList))
        elif not isinstance(protocol,list):
            protocol = [protocol]
        bins = np.arange(0,maxInterval+binWidth,binWidth)
        plt.figure(facecolor='w')
        gs = gridspec.GridSpec(len(units),len(protocol))
        for i,u in enumerate(units):
            ax = []
            ymax = 0
            for j,p in enumerate(protocol):
                spikeTimes = self.units[u]['times'][str(p)]/self.sampleRate
                isiHist,_ = np.histogram(np.diff(spikeTimes),bins)
                isiProb = isiHist/spikeTimes.size
                ymax = max(ymax,isiProb.max())
                ax.append(plt.subplot(gs[i,j]))
                ax[-1].bar(bins[:-1],isiProb,binWidth,color='b',edgecolor='b')
                ax[-1].set_xlim([0,maxInterval])
                ax[-1].spines['right'].set_visible(False)
                ax[-1].spines['top'].set_visible(False)
                ax[-1].tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                if i==0:
                    protocolName = os.path.dirname(self.kwdFileList[j])
                    ax[-1].set_title(protocolName[protocolName.rfind('_')+1:],fontsize='x-small')
                if i==len(units)-1 and j==len(protocol)-1:
                    ax[-1].set_xticks([0,maxInterval])
                    ax[-1].set_xlabel('ISI (s)',fontsize='x-small')
                else:
                    ax[-1].set_xticks([])
            for j,a in enumerate(ax):
                a.set_ylim([0,ymax])
                if j==0:
                    a.set_yticks([int(ymax*100)/100])
                    a.set_ylabel(u,fontsize='x-small')
                else:
                    a.set_yticks([])
                    
    def plotAutoCorr(self,units=None,protocols=None, bin_width=0.0005, width=0.1, rfViolWin=0.0015):
        units,unitsYPos = self.getOrderedUnits(units)
        
        if protocols is None:
            protocols = []
            for pro in self.kwdFileList:
                name = os.path.dirname(pro)
                pname = name[name.rfind('_')+1:]
                protocols.append(pname)
        
        if not isinstance(protocols,list):
            protocols = [protocols]
            
        plt.figure(facecolor='w', figsize=[14, 5])
        gs = gridspec.GridSpec(len(units),len(protocols))
        for ii,u in enumerate(units):
            ax = []
            for jj,p in enumerate(protocols):
                proIndex = self.getProtocolIndex(p)
                if proIndex is None:
                    print('No protocol matching ' + str(p))
                    continue
                spike_times = self.units[u]['times'][str(proIndex)]/float(self.sampleRate)
                d = []                   # Distance between any two spike times
                n_sp = len(spike_times)  # Number of spikes in the input spike train
                
                i, j = 0, 0
                for t in spike_times:
                    # For each spike we only consider those spikes times that are at most
                    # at a 'width' time lag. This requires finding the indices
                    # associated with the limiting spikes.
                    while i < n_sp and spike_times[i] < t - width:
                        i += 1
                    while j < n_sp and spike_times[j] < t + width:
                        j += 1
                    # Once the relevant spikes are found, add the time differences
                    # to the list
                    d.extend(spike_times[i:j] - t)
                    
                rfViolations = np.sum(np.logical_and(np.array(d)>0, np.array(d)<rfViolWin))
                rfViolPer = np.round(100*rfViolations/len(spike_times), 3)
                n_b = int( np.ceil(width / bin_width) )  # Num. edges per side
                # Define the edges of the bins (including rightmost bin)
                b = np.linspace(-width, width, 2 * n_b, endpoint=True)
                [h, hb] = np.histogram(d, bins=b)
                h[np.ceil(len(h)/2).astype(int) - 1] = 0
                                
                
                ax.append(plt.subplot(gs[ii,jj]))
                ax[-1].bar(hb[:-1], h, bin_width)
                ax[-1].set_xlim([-width,width])
                ax[-1].spines['right'].set_visible(False)
                ax[-1].spines['top'].set_visible(False)
                ax[-1].tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                if ii==0:
                    ax[-1].set_title(p,fontsize='x-small')
                ax[-1].text(-width*0.9, 0.95*ax[-1].get_ylim()[1], 'rf viol: ' + str(rfViolPer), color='r')
             
    def plotSDF(self,unit,protocol,startSamples=None,offset=0,windowDur=None,sigma=0.02,sampInt=0.001,paramNames=None):
        # offset in seconds
        # windowDur input in seconds then converted to samples
        if paramNames is not None and len(paramNames)>2:
            raise ValueError('plotSDF does not accept more than 2 parameters')
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
        if windowDur is not None and not isinstance(windowDur,np.ndarray):
            windowDur = int(windowDur*self.sampleRate)
            windowDur = np.array([windowDur for _ in startSamples])
        paramSet = [np.unique(param) for param in params]
        startSamples += int(offset*self.sampleRate)
        spikes = self.units[str(unit)]['times'][protocol]
        
        if len(params)==0:
            rows = cols = [0]
        elif len(params)==1:
            rows = range(len(set(params[0])))
            cols = [0]
        else:
            cols,rows = [range(len(set(params[i]))) for i in (0,1)]
        plt.figure(facecolor='w')
        gs = gridspec.GridSpec(len(rows),len(cols))
        ax = []
        xmax = max(windowDur)/self.sampleRate+offset
        ymax = 0
        for i in rows:
            if len(params)>0:
                trials = np.where(params[len(params)-1]==paramSet[len(params)-1][::-1][i])[0]
            else:
                trials = np.arange(len(startSamples))
            for j in cols:
                if len(params)>1:
                    trialIndex = np.intersect1d(trials,np.where(params[0]==paramSet[0][j])[0])
                else:
                    trialIndex = trials
                sdf,t = self.getSDF(spikes,startSamples[trialIndex],max(windowDur[trialIndex]),sigma=sigma,sampInt=sampInt)
                ymax = max(ymax,sdf.max())
                ax.append(plt.subplot(gs[i,j]))
                ax[-1].plot(t+offset,sdf)
        ymax *= 1.05
        for ind,a in enumerate(ax):
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
            a.set_xlim([offset,xmax])
            a.set_ylim([0,ymax])
            if ind==len(ax)-1:
                a.set_xticks([0,xmax])
                a.set_xlabel('Time (s)',fontsize='x-small')
            else:
                a.set_xticks([])
            if ind==0:
                a.set_yticks([0,ymax])
                a.set_ylabel('Spikes/s',fontsize='x-small')
                a.set_title('Unit '+str(unit),fontsize='small')
            else:
                a.set_yticks([])
    
    def plotSDF1Axis(self, sdf, sdfTime, sdfMax=None, ax=None, lineColor=None, figureTitle = None):
        if len(sdf.shape)==2:
            sdf = sdf[:,None, :]
        if sdfMax is None:
            sdfMax = np.nanmax(sdf)
        spacing = 0.2
        sdfXMax = sdfTime[-1]
        sdfYMax = sdfMax
        if ax is None:
            if figureTitle is not None:
                fig = plt.figure(figureTitle, facecolor='w')        
            else:
                fig = plt.figure(facecolor='w')        
            ax = fig.add_subplot(111)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
            ax.set_xticks([0, round(sdfXMax/5)])
            ax.set_xticklabels(['0', str(int(round(sdfXMax/5))) + ' s'])
            ax.set_yticks([0,int(sdfMax)])
            ax.set_xlim([-sdfXMax*spacing,sdfXMax*(1+spacing)*sdf.shape[1]])
            ax.set_ylim([-sdfYMax*spacing,sdfYMax*(1+spacing)*sdf.shape[0]])
        if lineColor is None:
            lineColor = 'k'
        
        x = 0
        y = 0
        for i in xrange(sdf.shape[0]):
            for j in xrange(sdf.shape[1]):
                ax.plot(x+sdfTime,y+sdf[i,j,:],color=lineColor)
                x += sdfXMax*(1+spacing)
            x = 0
            y += sdfYMax*(1+spacing)

            
               
    def getSDF(self,spikes,startSamples,windowSamples,sigma=0.02,sampInt=0.001,avg=True):
        binSamples = int(sampInt*self.sampleRate)
        bins = np.arange(0,windowSamples+binSamples,binSamples)
        binnedSpikeCount = np.zeros((len(startSamples),len(bins)-1))
        for i,start in enumerate(startSamples):
            binnedSpikeCount[i],_ = np.histogram(spikes[np.logical_and(spikes>=start,spikes<=start+windowSamples)]-start,bins)
        sdf = scipy.ndimage.filters.gaussian_filter1d(binnedSpikeCount,sigma/sampInt,axis=1)
        if avg:
            sdf = sdf.mean(axis=0)
        sdf /= sampInt
        t = bins[:-1]/self.sampleRate
        return sdf,t
        
        
    def getSDFNoise(self,spikes,startSamples,windowSamples,sigma=0.02,sampInt=0.001,nReps=None):
        if nReps is None:
            nReps = int(windowSamples/self.sampleRate/sampInt/2)
        bufferTime = sigma*5
        bufferSamples = int(bufferTime*self.sampleRate)
        sdf,t = self.getSDF(spikes,startSamples-bufferSamples,windowSamples+2*bufferSamples,sigma=sigma,sampInt=sampInt,avg=False)
        sdf = sdf[:,np.logical_and(t>bufferTime,t<t[-1]-bufferTime)]
        peaks = np.zeros(nReps)
        for n in range(nReps):
            for i,_ in enumerate(sdf):
                sdf[i] = np.roll(sdf[i],np.random.randint(0,sdf.shape[1]))
            peaks[n] = sdf.mean(axis=0).max()
        return peaks
    
    
    def plotRaster(self,unit,protocol,startSamples=None,offset=0,windowDur=None,paramNames=None,paramColors=None,grid=False,axes=None):
        # offset and windowDur input in seconds then converted to samples
        offset = int(offset*self.sampleRate)
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
        if windowDur is not None and not isinstance(windowDur,np.ndarray):
            windowDur = int(windowDur*self.sampleRate)
            windowDur = [windowDur for _ in startSamples]
        startSamples = startSamples+offset
        spikes = self.units[str(unit)]['times'][str(protocol)]
        
        if paramColors is None:
            paramColors = [None]*len(params)
        else:
            for i,c in enumerate(paramColors):
                if c=='auto' and i<len(params):
                    paramColors[i] = cm.Dark2(range(0,256,int(256/len(set(params[i])))))
                    break
        grid = True if grid and len(paramNames)==2 else False
        
        if axes is None:
            plt.figure(facecolor='w')
            if grid:
                axes = []
                rows = []
                gs = gridspec.GridSpec(len(set(params[1])),len(set(params[0])))
            else:
                axes = [plt.subplot(1,1,1)]
                rows = [0]
                gs = None
        else:
            axes = [axes]
            rows = [0]
            gs = None
            
        if len(params)<1:
            self.appendToRaster(axes,spikes,startSamples,offset,windowDur,rows=rows)
        else:
            self.parseRaster(axes,spikes,startSamples,offset,windowDur,params,paramColors,rows=rows,grid=grid,gs=gs)
        for ax,r in zip(axes,rows):
            ax.set_xlim([offset/self.sampleRate,(max(windowDur)+offset)/self.sampleRate])
            ax.set_ylim([-0.5,r-0.5])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            if grid:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Trial')
        axes[-1].set_title('Unit '+str(unit)+', '+self.getProtocolLabel(protocol))
         
         
    def parseRaster(self,axes,spikes,startSamples,offset,windowDur,params,paramColors,paramIndex=0,trialsIn=None,rows=[0],grid=False,gs=None,grow=None,gcol=0):
        paramVals = np.unique(params[paramIndex])
        if grid and grow is None:
            grow = gs.get_geometry()[0]-1
        for i,val in enumerate(paramVals):
            if grid:
                axes.append(plt.subplot(gs[grow,gcol]))
                rows.append(0)
                grow -= 1
            trialIndex = np.where(params[paramIndex]==val)[0]
            if trialsIn is not None:
                trialIndex = np.intersect1d(trialsIn,trialIndex)
            if paramIndex<len(params)-1:
                if paramColors[paramIndex] is not None:
                    paramColors[paramIndex+1] = [paramColors[paramIndex][i]]*len(set(params[paramIndex+1]))
                self.parseRaster(axes,spikes,startSamples,offset,windowDur,params,paramColors,paramIndex+1,trialIndex,rows,grid,gs,None,gcol)
                if grid:
                    gcol += 1
            else:
                color = 'k' if paramColors[paramIndex] is None else paramColors[paramIndex][i]
                self.appendToRaster(axes,spikes,startSamples,offset,windowDur,trialIndex,rows,color)
 
 
    def appendToRaster(self,axes,spikes,startSamples,offset,windowDur,trialIndex=None,rows=[0],color='k'):
        if trialIndex is None:
            trialIndex = range(len(startSamples))
        for i in trialIndex:
            spikeTimes = (spikes[np.logical_and(spikes>startSamples[i],spikes<startSamples[i]+windowDur[i])]-startSamples[i]+offset)/self.sampleRate
            axes[-1].vlines(spikeTimes,rows[-1]-0.4,rows[-1]+0.4,color=color)
            rows[-1] += 1
            
            
    def plotLaserRaster(self,units=None,figNum=None,nonMU=False):
        if units is None:
            if nonMU:
                units,_ = self.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            else:
                units,_ = self.getOrderedUnits()
        elif not isinstance(units,(list,tuple)):
            units = [units]
        laserProtocols = [protocol for protocol,label in enumerate(self.kwdFileList) if 'laser' in label]
        if len(laserProtocols)>0:
            for u in units:
                if figNum is not None:
                    figNum += 1
                f = plt.figure(num=figNum,figsize=(10,10))
                for ind,protocol in enumerate(laserProtocols):
                    ax = f.add_subplot(len(laserProtocols),1,ind+1)
                    laserStartSamples = self.behaviorData[str(protocol)]['137_pulses'][0]
                    self.plotRaster(u,str(protocol),laserStartSamples,offset=-2,windowDur=6,axes=ax)
    
    
    def runAllAnalyses(self, units=None, protocolsToRun=None, splitRunning=False, useCache=False, plot=True, ttx=False):
        if protocolsToRun is None:
            protocolsToRun = ['sparseNoise', 'flash', 'gratings', 'gratings_ori', 'checkerboard', 'loom']
        for pro in protocolsToRun:
            protocol = self.getProtocolIndex(pro)
            if protocol is None:
                print(pro+ ' not found')
            elif 'loom' in pro and 'lvHistory' not in self.visstimData[str(protocol)]:
                continue
            else:
                trialStarts, trialEnds = self.getTrialStartsEnds(protocol)
                laserTag = '_laserOff'
                if 'gratings'==pro:
                    if splitRunning:
                        statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)  
                        laserTrials, controlTrials = self.findLaserTrials(protocol)
                        laserCondition = [controlTrials, laserTrials]
                        hasLaser = True if any([c in self.visstimData[str(protocol)] for c in ['laserPowerHistory', 'stimulusHistory_laserPower', 'trialLaserPower']]) else False
                        analyzeLaserTrials = (False,True) if hasLaser and np.any(self.visstimData[str(protocol)]['laserPower']>0) else (False,)
                        for il, laser in enumerate(analyzeLaserTrials):
                            laserTag = '_laserOn' if laser else '_laserOff'
                            sTrials = np.intersect1d(statTrials, laserCondition[il])
                            rTrials = np.intersect1d(runTrials, laserCondition[il])
                            aTrials = laserCondition[il]
                                
                            self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf', trials=sTrials, saveTag=laserTag+'_stat', plot=plot)
                            self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf', trials=rTrials, saveTag=laserTag+'_run', plot=plot)
                            self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf', trials=aTrials, saveTag=laserTag+'_allTrials', plot=plot)
                   
                    else:
                        self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf', plot=plot)
    
                elif 'gratings_ori'==pro:
                    if splitRunning:
                        statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)  
                        laserTrials, controlTrials = self.findLaserTrials(protocol)
                        laserCondition = [controlTrials, laserTrials]
                        hasLaser = True if any([c in self.visstimData[str(protocol)] for c in ['laserPowerHistory', 'stimulusHistory_laserPower', 'trialLaserPower']]) else False
                        analyzeLaserTrials = (False,True) if hasLaser and np.any(self.visstimData[str(protocol)]['laserPower']>0) else (False,)
                        for il, laser in enumerate(analyzeLaserTrials):
                            laserTag = '_laserOn' if laser else '_laserOff'
                            sTrials = np.intersect1d(statTrials, laserCondition[il])
                            rTrials = np.intersect1d(runTrials, laserCondition[il])
                            aTrials = laserCondition[il]
                                
                            self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori', trials=sTrials, saveTag=laserTag+'_stat', plot=plot)
                            self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori', trials=rTrials, saveTag=laserTag+'_run', plot=plot)
                            self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori', trials=aTrials, saveTag=laserTag+'_allTrials', plot=plot)
                    
                    else:
                        self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori', plot=plot)
    
                elif 'sparseNoise' in pro:                                            
                    if ttx:
                        controlTrials, controlProtocol, ttxTrials, ttxProtocol = self.findTTXTrials(pro)
                        for pid, trials, tag in zip((controlProtocol, ttxProtocol), (controlTrials, ttxTrials), ('_control', '_ttx')):
                            trialStarts, trialEnds = self.getTrialStartsEnds(pid)
                            statTrials, runTrials, _ = self.parseRunning(pid, trialStarts=trialStarts, trialEnds=trialEnds)
                            sTrials = np.intersect1d(statTrials, trials).astype(np.int)
                            rTrials = np.intersect1d(runTrials, trials).astype(np.int)
                            aTrials = trials.astype(np.int)
                            self.findRF(units, protocol=pid, trials=sTrials, saveTag=tag+'_stat', plot=plot)
                            self.findRF(units, protocol=pid, trials=rTrials, saveTag=tag+'_run', plot=plot)
                            self.findRF(units, protocol=pid, trials=aTrials, saveTag=tag+'_allTrials', plot=plot)
                        
                    elif splitRunning:
                        statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                        laserTrials, controlTrials = self.findLaserTrials(protocol)
                        laserCondition = [controlTrials, laserTrials]
                        hasLaser = True if any([c in self.visstimData[str(protocol)] for c in ['laserPowerHistory', 'stimulusHistory_laserPower', 'trialLaserPower']]) else False
                        analyzeLaserTrials = (False,True) if hasLaser and np.any(self.visstimData[str(protocol)]['laserPower']>0) else (False,)
                        for il, laser in enumerate(analyzeLaserTrials):
                            laserTag = '_laserOn' if laser else '_laserOff'
                            sTrials = np.intersect1d(statTrials, laserCondition[il])
                            rTrials = np.intersect1d(runTrials, laserCondition[il])
                            aTrials = laserCondition[il]
                                
                            self.findRF(units, protocol=protocol, trials=sTrials, saveTag=laserTag+'_stat', plot=plot)
                            self.findRF(units, protocol=protocol, trials=rTrials, saveTag=laserTag+'_run', plot=plot)
                            self.findRF(units, protocol=protocol, trials=aTrials, saveTag=laserTag+'_allTrials', plot=plot)

                            
                            
                    else:                    
                        self.findRF(units, protocol=protocol, useCache=useCache, plot=plot)
                        
                elif 'flash' in pro:
                    if splitRunning:
                        statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)                 
                        self.analyzeFlash(units, protocol=protocol, useCache=useCache, trials=statTrials, saveTag=laserTag+'_stat', plot=plot)
                        self.analyzeFlash(units, protocol=protocol, useCache=useCache, trials=runTrials, saveTag=laserTag+'_run', plot=plot)
                    else:                    
                        self.analyzeFlash(units, protocol=protocol, useCache=useCache, plot=plot)
                        
                elif 'spots' in pro:
                    if splitRunning:
                        statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)                    
                        self.analyzeSpots(units, protocol=protocol, useCache=useCache, trials=statTrials, saveTag=laserTag+'_stat', plot=plot)
                        self.analyzeSpots(units, protocol=protocol, useCache=useCache, trials=runTrials, saveTag=laserTag+'_run', plot=plot)
                    else:
                        self.analyzeSpots(units, protocol=protocol, useCache=useCache, plot=plot)
                        
                elif 'checkerboard' in pro:
                    if ttx:
                        controlTrials, controlProtocol, ttxTrials, ttxProtocol = self.findTTXTrials(pro)
                        for pid, trials, tag in zip((controlProtocol, ttxProtocol), (controlTrials, ttxTrials), ('_control', '_ttx')):
                            trialStarts, trialEnds = self.getTrialStartsEnds(pid)
                            statTrials, runTrials, _ = self.parseRunning(pid, trialStarts=trialStarts, trialEnds=trialEnds)
                            sTrials = np.intersect1d(statTrials, trials).astype(np.int)
                            rTrials = np.intersect1d(runTrials, trials).astype(np.int)
                            aTrials = trials.astype(np.int)
                            self.analyzeCheckerboard(units, protocol=pid, trials=sTrials, saveTag=tag+'_stat', plot=plot)
                            self.analyzeCheckerboard(units, protocol=pid, trials=rTrials, saveTag=tag+'_run', plot=plot)
                            self.analyzeCheckerboard(units, protocol=pid, trials=aTrials, saveTag=tag+'_allTrials', plot=plot)
                            
                    elif splitRunning:
                        statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                        hasLaser = True if 'trialLaserPower' in self.visstimData[str(protocol)] else False
                        analyzeLaserTrials = (False,True) if hasLaser and np.any(self.visstimData[str(protocol)]['laserPower']>0) else (False,)
                        for laser in analyzeLaserTrials:
                            laserTag = '_laserOn' if laser else '_laserOff'
                            self.analyzeCheckerboard(units, protocol=protocol, trials=statTrials, laser=laser, saveTag=laserTag+'_stat', plot=plot)
                            self.analyzeCheckerboard(units, protocol=protocol, trials=runTrials, laser=laser, saveTag=laserTag+'_run', plot=plot)
                            self.analyzeCheckerboard(units, protocol=protocol, laser=laser, saveTag=laserTag+'_allTrials', plot=plot)
                    
                    else:
                        self.analyzeCheckerboard(units, protocol=protocol, plot=plot)
                elif 'loom' in pro:
                    if splitRunning:
                        statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                        laserTrials, controlTrials = self.findLaserTrials(protocol)
                        laserCondition = [controlTrials, laserTrials]
                        hasLaser = True if any([c in self.visstimData[str(protocol)] for c in ['laserPowerHistory', 'stimulusHistory_laserPower', 'trialLaserPower']]) else False
                        analyzeLaserTrials = (False,True) if hasLaser and np.any(self.visstimData[str(protocol)]['laserPower']>0) else (False,)
                        for il, laser in enumerate(analyzeLaserTrials):
                            laserTag = '_laserOn' if laser else '_laserOff'
                            sTrials = np.intersect1d(statTrials, laserCondition[il])
                            rTrials = np.intersect1d(runTrials, laserCondition[il])
                            aTrials = laserCondition[il]
                                
                            self.analyzeLoom(units, protocol=protocol, trials=sTrials, saveTag=laserTag+'_stat', plot=plot)
                            self.analyzeLoom(units, protocol=protocol, trials=rTrials, saveTag=laserTag+'_run', plot=plot)
                            self.analyzeLoom(units, protocol=protocol, trials=aTrials, saveTag=laserTag+'_allTrials', plot=plot)
                    else:
                        self.analyzeLoom(units, protocol=protocol, plot=plot)
        if plot:
            units, _ = self.getOrderedUnits(units)
            if len(units)==1:
                self.plotSpikeRateOverProtocols(units)
                self.plotAutoCorr(units[0])
                self.plotSpikeAmplitudes(units[0])
                _ = self.getSpikeTemplate(units[0])
            
    def plotSpikeRateOverProtocols(self, units=None, protocols = None, binsize = 10):        
        units, _ = self.getOrderedUnits(units)        

        if protocols is None:
            protocols = []
            for pro in self.kwdFileList:
                name = os.path.dirname(pro)
                pname = name[name.rfind('_')+1:]
                protocols.append(pname)
        
        if not isinstance(protocols,list):
            protocols = [protocols]
            
        proInds = []
        for pro in protocols:
            pi = self.getProtocolIndex(pro)
            if pi is not None:
                proInds.append(pi)                                                
        try:
            nSamps = np.array(self.nsamps)[proInds]
        except:
            nSamps = []
            for pro in proInds:
                nSamps.append(self.behaviorData[str(pro)]['running'].size * 500)
                
        for u in units:
            fig = plt.figure(u, facecolor='w', figsize=[8, 2*len(proInds)])
            axes = []
            maxSpikeRate = 0
            for ip, pro in enumerate(proInds):
                spikes = self.units[u]['times'][str(pro)]
                if spikes.size > 0 and nSamps[ip] > binsize*self.sampleRate:
                    ax = fig.add_subplot(len(proInds), 1, ip+1)
                    h, b = np.histogram(spikes, int(round(nSamps[ip]/(binsize*self.sampleRate))), [0, nSamps[ip]])
                    h_spr = h/binsize
                    if np.max(h_spr) > maxSpikeRate:
                        maxSpikeRate = np.max(h_spr)
                    ax.plot(b[1:], h_spr, 'k')
                    ax.set_title(self.getProtocolLabel(pro))
                    formatFigure(fig, ax, yLabel='Spike Rate (Hz)')
                    axes.append(ax)

            for a in axes:
                a.set_ylim([0, maxSpikeRate])
            fig.tight_layout()
            
    def getSpikeTemplate(self, unit, plot=True):
        temp = self.units[str(unit)]['template']
        maxChan = np.unravel_index(np.argmax(temp), temp.shape)[1]
        t = np.copy(temp[:, maxChan])
        x = 1000*np.arange(t.size)/self.sampleRate
        
        # convert template to real units
        if hasattr(self, 'digitalGain'):
            dg = self.digitalGain
        else:
            dg = 0.195
        
        if 'amplitudes' in self.units[str(unit)]:
            meanTempScaleFactor = np.mean(self.units[str(unit)]['amplitudes'])
            t *= meanTempScaleFactor*dg
        else:
            print('No spike amplitudes found for: ' + str(unit) + '. Displaying arb units')
        
        if plot:        
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(111)
            ax.plot(x, t, 'k')
            formatFigure(fig, ax, 'Spike Template for: ' + str(unit), xLabel='ms', yLabel='mV')
        
        return t
        
                                    
    def plotSpikeAmplitudes(self, unit):
        filelist = self.kwdFileList
        filePaths = [os.path.dirname(f) for f in filelist]
        fileDir=os.path.dirname(filePaths[0])
        _, kwdNsamplesList = getKwdInfo(dirPath=fileDir)        
        
        if 'amplitudes' in self.units[str(unit)]:
            
            fig = plt.figure(facecolor='w', figsize=[14, 4.5])                        
            ax = fig.add_subplot(111)        
            
            protocolEnds = np.cumsum(kwdNsamplesList)
            protocolStarts = np.insert(protocolEnds, 0, 0)[:-1] - 1
            spikeTimes = []
            for i, _ in enumerate(self.kwdFileList):
                sts = self.units[str(unit)]['times'][str(i)]                
                spikeTimes.extend(sts + protocolStarts[i])
            
            ax.plot(spikeTimes[::100], self.units[str(unit)]['amplitudes'][::100], 'ko', alpha=0.2)
            ax.set_xlim([0, protocolEnds[-1]])
            for i, p in enumerate(self.kwdFileList):
                name = os.path.dirname(p)
                ax.plot([protocolEnds[i], protocolEnds[i]], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--')
                ax.text(np.mean([protocolStarts[i], protocolEnds[i]]), 0.95*ax.get_ylim()[1], name[name.rfind('_')+1:], horizontalalignment='center')
            formatFigure(fig, ax, 'Spike Template Amplitudes for unit: ' + str(unit), xLabel='Sample #', yLabel='Template Scale Factor')
        else:
            print('No amplitudes found for unit: ' + str(unit))
                       
    def getTrialStartsEnds(self, protocol, timeUnit='samples'):
        if isinstance(protocol, str):
            protocol = self.getProtocolIndex(protocol)

        trialStarts = None
        trialEnds = None
        
        v = self.visstimData[str(protocol)]
        
        if 'loom' in self.getProtocolLabel(protocol):
            trialStarts = v['stimStartFrames']
            trialFrameLengths = (np.diff(v['stimStartFrames']) - v['interTrialInterval'])
            trialStarts = trialStarts[:-1]
            trialEnds = v['stimStartFrames'][:-1] + trialFrameLengths
        else:
            for param in ['trialStartFrame', 'stimStartFrames']:
                if param in v:
                    trialStarts = v[param]
            
            if trialStarts is None:
                print('Could not find trial starts: key must be either trialStartFrame or stimStartFrames')
                return
            
            for param in ['trialNumFrames', 'boxDuration', 'stimTime', 'stimDur']:
                if param in v:
                    if isinstance(v[param], int) or issubclass(type(v[param]),np.integer):
                        trialDuration = v[param]
                    else:
                        trialDuration = v[param][:len(trialStarts)].astype(np.int)
                    trialEnds = trialStarts + trialDuration
                    
            lastFullTrial = np.where(trialEnds<v['frameSamples'].size)[0][-1]
            trialStarts = trialStarts[:lastFullTrial+1]        
            trialEnds = trialEnds[:lastFullTrial+1]
        if timeUnit=='samples':    
            return v['frameSamples'][trialStarts], v['frameSamples'][trialEnds]
        elif timeUnit=='frames':
            return trialStarts, trialEnds
        else:
            print('Did not recognize timeUnit. Must be either samples or frames.')
            return
                
     
    def getProtocolIndex(self, label):
        protocol = []
        protocol.extend([i for i,f in enumerate(self.kwdFileList) if ntpath.dirname(str(f)).endswith(label)])
        if len(protocol)>1:
            raise ValueError('Multiple protocols found matching: '+label)
        elif len(protocol)<1:
            return None
        else:
            return protocol[0]
        
    def getProtocolLabel(self,protocolIndex):
        match = re.search('_\d{2,2}-\d{2,2}-\d{2,2}_.*',ntpath.dirname(self.kwdFileList[int(protocolIndex)]))
        return match.group()[10:]
        
    def getOrderedUnits(self,units=None):
        # orderedUnits, yPosition = self.getOrderedUnits(units)
        if units is None:
            units = [u for u in self.units]
        elif isinstance(units,(list,tuple)):
            units = [str(u) for u in units]
        else:
            units = [str(units)]
        for u in units:
            if u not in self.units:
                units.remove(u)
                print(str(u)+' not in units')
        if len(units)<1:
            raise ValueError('Found no matching units')
        orderedUnits = [(u,self.units[u]['ypos']) for u in self.units if u in units]
        orderedUnits.sort(key=lambda i: i[1], reverse=True)
        return zip(*orderedUnits)
    
    
    def getSingleUnits(self, fileDir = None, protocolsToAnalyze = None):
        if fileDir is None:
            fileDir = fileIO.getDir()
        fileList, nsamps = getKwdInfo(dirPath=fileDir)
        if protocolsToAnalyze is None:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir)
        else:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir, protocolsToAnalyze=protocolsToAnalyze)
    
    
#    def loadClusteredData(self, kwdNsamplesList = None, protocolsToAnalyze = None, fileDir = None):
#        from load_phy_template import load_phy_template
#                 
#        if fileDir is None:
#            fileDir = fileIO.getDir()
#        
#        if protocolsToAnalyze is None:
#            protocolsToAnalyze = np.arange(len(self.kwdFileList))
#        
#        self.units = load_phy_template(fileDir, sampling_rate = self.sampleRate)
#        for unit in self.units:
#            spikeTimes = (self.units[unit]['times']).astype(int)
#            
#            if kwdNsamplesList is not None:
#                self.units[unit]['times'] = {}
#                protocolEnds = np.cumsum(kwdNsamplesList)
#                protocolStarts = np.insert(protocolEnds, 0, 0)[:-1] - 1
#                for pro in protocolsToAnalyze:                    
#                    self.units[unit]['times'][str(pro)] = spikeTimes[np.logical_and(spikeTimes >= protocolStarts[pro], spikeTimes < protocolEnds[pro])]
#                    self.units[unit]['times'][str(pro)] -= protocolStarts[pro]
#            else:
#              self.units[unit]['times'] = spikeTimes       
    
    def loadClusteredData(self, kwdNsamplesList = None, protocolsToAnalyze = None, fileDir = None):
        from load_phy_template import load_phy_template
                 
        if fileDir is None:
            fileDir = fileIO.getDir()
        
        if protocolsToAnalyze is None:
            protocolsToAnalyze = np.arange(len(self.kwdFileList))
        
        units = load_phy_template(fileDir, sampling_rate = self.sampleRate)
        for unit in units:
            spikeTimes = (units[unit]['times']).astype(int)
            
            if kwdNsamplesList is not None:
                units[unit]['times'] = {}
                protocolEnds = np.cumsum(kwdNsamplesList)
                protocolStarts = np.insert(protocolEnds, 0, 0)[:-1] - 1
                for pro in protocolsToAnalyze:                    
                    units[unit]['times'][str(pro)] = spikeTimes[np.logical_and(spikeTimes >= protocolStarts[pro], spikeTimes < protocolEnds[pro])]
                    units[unit]['times'][str(pro)] -= protocolStarts[pro]
            else:
              units[unit]['times'] = spikeTimes       
        
        
        if hasattr(self, 'units'):
            for u in units:
                if u in self.units:
                    for key in units[u]:
                        self.units[u][key] = units[u][key]
                else:
                    self.units[u]= {}
                    self.units[u] = units[str(u)]
        else:
            self.units = units
                    

    def saveHDF5(self,filePath=None):
        fileIO.objToHDF5(self,filePath)
                    
                    
    def loadHDF5(self,filePath=None):
        fileIO.hdf5ToObj(self,filePath)
        if not hasattr(self, 'nsamps'):
            try:
                _, self.nsamps = getKwdInfo(os.path.dirname(os.path.dirname(self.kwdFileList[0])))
            except:
                print('Could not find number of samples for kwd files')
                pass
    
    def saveWorkspace(self, variables=None, saveGlobals = False, fileName=None, exceptVars = []):
        if fileName is None:
            fileName = fileIO.saveFile()
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
            fileName = fileIO.getFile()
            if fileName=='':
                return
        shelf = shelve.open(fileName)
        for key in shelf:
            setattr(self, key, shelf[key])
        shelf.close()
        
        
    def getExperimentInfo(self):
        expName = ntpath.basename(ntpath.dirname(ntpath.dirname(self.kwdFileList[0])))
        if isinstance(expName,bytes):
            expName = expName.decode('utf8')
        date,animalID = expName.split('_')[:2]
        return date,animalID


    def readExcelFile(self, sheetname = None, fileName = None):
        if sheetname is None:            
            expDate,animalID = self.getExperimentInfo()
            sheetname = expDate+'_'+animalID
        if fileName is None:        
            fileName = fileIO.getFile()
            if fileName=='':
                return        
        
        table = pandas.read_excel(fileName, sheetname=sheetname)
                
        if 'Genotype' in table.columns:
            self.genotype = table.Genotype[0]
        else:
            self.genotype = ''
            
        for u in range(table.shape[0]):
            unit = table.Cell[u]
            if not np.isnan(unit):
                self.units[str(int(unit))]['label'] = table.Label[u]
                if 'Laser Resp' in table.keys():
                    self.units[str(int(unit))]['laserResp'] = table['Laser Resp'][u]
                else:
                    self.units[str(int(unit))]['laserResp'] = np.nan
        assert(all('label' in self.units[u] for u in self.units.keys()))
            
        try:
            self.CCFTipPosition = np.array(table.Tip[0:3]).astype(float)
            self.CCFLPEntryPosition = np.array(table.Entry[0:3]).astype(float)
            self.findCCFCoords()
        except:
            for u in self.units:
                self.units[str(u)]['CCFCoords'] = np.full(3,np.nan)
            print('Could not find CCF Tip or Entry positions')
           
        if 'TTX_protocols' in table.columns:
            ttxPros = [a for a in table['TTX_protocols'] if not isinstance(a, float)]
            self.ttxDict = {}
            for i, pro in enumerate(ttxPros):
                self.ttxDict[pro] =     {'control_start': table['control_start'][i],
                                         'control_end': table['control_end'][i],
                                         'ttx_start' : table['TTX_start'][i],
                                         'ttx_end' : table['TTX_end'][i]}
                

    def getUnitsByLabel(self, labelID, criterion, notFlag=False):
        units = []
        for unit in self.units.keys():
            label = self.units[unit][labelID]
            if notFlag:                
                if label not in criterion:
                    units.append(unit)
            else:
                if label in criterion:
                    units.append(unit)
        return self.getOrderedUnits(units)
        
        
    def findCCFCoords(self, tipPos=None, entryPos=None, tipProbePos=-1300):
        if tipPos is None:
            tipPos = self.CCFTipPosition
        if entryPos is None:
            entryPos = self.CCFLPEntryPosition      
        
        xRange = entryPos[0] - tipPos[0]
        yRange = entryPos[1] - tipPos[1]
        zRange = entryPos[2] - tipPos[2]
        
        trackLength = math.sqrt(xRange**2 + yRange**2 + zRange**2)
        
        xSlope = xRange/trackLength
        ySlope = yRange/trackLength
        zSlope = zRange/trackLength
                        
        units, unitsYPos = self.getOrderedUnits()
        if min(unitsYPos)>0:
            unitsYPos = [y-1260 for y in unitsYPos]
        for i, unit in enumerate(units):
            distFromTip = unitsYPos[i] - tipProbePos
            pos = np.array([xSlope, ySlope, zSlope])*distFromTip
            pos += tipPos
            self.units[str(unit)]['CCFCoords'] = pos
            
    
    def comparisonPlot(self, unit, tagsToPlot=['stat', 'run'], protocolsToPlot=['sparseNoise', 'flash', 'gratings', 'gratings_ori', 'checkerboard'], colors = ['k', 'r', 'b', 'g'], cmap='bwr'):
        ### Tags to plot should be in order so that the reference tag is listed first. The reference tag will establish which trial conditions to use (usually, best conditions for that tag)
        data = self.units[str(unit)]
        for pro in protocolsToPlot:
            if pro is 'sparseNoise':
                for ir, resp in enumerate(['_sdfOn', '_sdfOff']):
                    fig = plt.figure(str(unit)+ ['On', 'Off'][ir], facecolor='w')
                    gs = gridspec.GridSpec(2, len(tagsToPlot))  
                    
                    axCreated = False
                    resps = np.array([np.nanmean(data['sparseNoise_' + tg][['onResp', 'offResp'][ir]], axis=0) for tg in tagsToPlot])
                    sizeTuning = 'sizeTuningOn' in data['sparseNoise_' + tagsToPlot[0]]
                    maxResp = np.nanmax(resps)
                    minResp = np.nanmin(resps)
                    rfAxes=[]
                    for it, tag in enumerate(tagsToPlot):
                        saveTag = '_' + tag
                        if (pro + saveTag) in data:
                            numTrials = len(data[pro + saveTag]['trials'])
                            sdf = data['sparseNoise' + saveTag][resp]    
                            try:
                                sdfMax = max([np.nanmax(data[pro +'_'+tg][resp]) for tg in tagsToPlot]) 
                                sdfMin = min([np.nanmin(data[pro +'_'+tg][resp]) for tg in tagsToPlot])
                            except:
                                sdfMax = np.nanmax(data[pro +'_'+tag][resp])
                                sdfMin = np.nanmin(data[pro +'_'+tag][resp])
                            sdfTime = data['sparseNoise' + saveTag]['_sdfTime']    
                            sdfMean = np.nanmean(sdf, axis=0)
                            ax = None
                            if it > 0 and axCreated:
                                ax = plt.gca()
                            self.plotSDF1Axis(sdfMean, sdfTime, sdfMax, ax=ax, lineColor=colors[it], figureTitle=str(unit)+resp)
                            axCreated = True
                            axRF=fig.add_subplot(gs[0, it])
                            im = axRF.imshow(np.nanmean(data['sparseNoise'+saveTag][['onResp', 'offResp'][ir]], axis=0), interpolation='none', origin='lower', clim=[minResp, maxResp])
                            axRF.set_title(str(unit)+''+tag + ': ' + str(numTrials) + ' trials', fontdict={'size':10})
                            axRF.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                            axRF.set_xticks([])
                            axRF.set_yticks([])
                            rfAxes.append(axRF)
                            if it==len(tagsToPlot)-1:
                                cb = plt.colorbar(im,ax=rfAxes,fraction=0.05,pad=0.04,shrink=0.5)
                                cb.ax.tick_params(length=0,labelsize='x-small')
                            
                            if sizeTuning:
                                axSize = fig.add_subplot(gs[1, :])
                                sizeTuningSize = [5, 10, 20, 40]
                                sizeTuningLabel = ['5', '10', '20', 'full']
                                axSize.plot([5, 10, 20, 40], data['sparseNoise'+saveTag][['sizeTuningOn', 'sizeTuningOff'][ir]], color=colors[it])
                                axSize.set_xticks(sizeTuningSize)
                                axSize.set_xticklabels(sizeTuningLabel)
                                axSize.set_ybound(lower=0)
                                axSize.set_xlabel('Size, deg')
                                axSize.set_ylabel('Spikes/s')
                            
                            
            if pro is 'gratings':
                fig = plt.figure(facecolor='w')
                gs = gridspec.GridSpec(1,len(tagsToPlot))  
                axes = []
                axCreated = False
                resps = np.array([data['gratings_stf_' + tg]['respMat'] for tg in tagsToPlot])
                centerPoints = [data['gratings_stf_' + tg]['spontRateMean'] if not np.isnan(data['gratings_stf_' + tg]['spontRateMean']) else np.nanmedian(data['gratings_stf_' + tg]['respMat']) for tg in tagsToPlot]
                cLims = [np.nanmax(abs(rMat-centerPoints[r])) for r, rMat in enumerate(resps)]
                for it, tag in enumerate(tagsToPlot):
                    saveTag = '_' + tag                    
                    if ('gratings_stf' + saveTag) in data:
                        numTrials = len(data['gratings_stf' + saveTag]['trials'])
                        sdf = data['gratings_stf' + saveTag]['_sdf']
                        try:
                            sdfMax = max([np.nanmax(data['gratings_stf'+'_'+tg]['_sdf']) for tg in tagsToPlot]) 
                            sdfMin = min([np.nanmin(data['gratings_stf'+'_'+tg]['_sdf']) for tg in tagsToPlot])
                        except:
                            sdfMax = np.nanmax(data['gratings_stf'+'_'+tag]['_sdf'])
                            sdfMin = np.nanmin(data['gratings_stf'+'_'+tag]['_sdf'])
                        sdfTime = data['gratings_stf' + saveTag]['_sdfTime']    
                        if it==0:
                            maxInd = np.unravel_index(np.nanargmax(sdf), sdf.shape)[2]                        
                        bestSDF = sdf[:,:,maxInd]
                        ax = None
                        if it > 0 and axCreated:
                            ax = plt.gca()
                        self.plotSDF1Axis(bestSDF, sdfTime, sdfMax, ax=ax, lineColor=colors[it])
                        axCreated = True
                        respMat = data['gratings_stf' + saveTag]['respMat']
                        ax = fig.add_subplot(gs[0, it])
                        im = ax.imshow(respMat[:, :, maxInd], clim=[centerPoints[it]-max(cLims), centerPoints[it]+max(cLims)], cmap=cmap, interpolation='none', origin='lower')
                        ax.set_title(str(unit)+' gratings stf: '+tag + ': ' + str(numTrials) + ' trials', fontdict={'size':10})
                        axes.append(ax)
                        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
                        cb.ax.tick_params(length=0,labelsize='x-small')                            

            if pro is 'gratings_ori':
                fig = plt.figure(facecolor='w')
                gs = gridspec.GridSpec(1,len(tagsToPlot))  
                ori = data['gratings_ori' + '_' + tagsToPlot[0]]['ori']  
                theta = ori * (np.pi/180.0)
                theta = np.append(theta, theta[0])
                axCreated = False
                sdfDict = {}
                for it, tag in enumerate(tagsToPlot):
                    saveTag = '_' + tag
                    if (pro + saveTag) in data:
                        numTrials = len(data[pro + saveTag]['trials'])
                        sdf = data['gratings_ori' + saveTag]['_sdf']
                        try:
                            sdfMax = max([np.nanmax(data[pro +'_'+tg]['_sdf']) for tg in tagsToPlot]) 
                            sdfMin = min([np.nanmin(data[pro +'_'+tg]['_sdf']) for tg in tagsToPlot])
                        except:
                            sdfMax = np.nanmax(data[pro +'_'+tag]['_sdf'])
                            sdfMin = np.nanmin(data[pro +'_'+tag]['_sdf'])
                       
                        sdfTime = data['gratings_ori' + saveTag]['_sdfTime']    
                        if it==0:
                            maxInd = np.unravel_index(np.nanargmax(sdf), sdf.shape)
                        bestSDF = sdf[maxInd[0], maxInd[1], :, :]
                        ax = None
                        if it > 0 and axCreated:
                            ax = plt.gca()
                        self.plotSDF1Axis(bestSDF, sdfTime, sdfMax, ax=ax, lineColor=colors[it])
                        axCreated = True
                        sdfDict[tag] = sdf                        
                        
                        axp = fig.add_subplot(gs[0, it], projection='polar')
                        respMat = data['gratings_ori' + saveTag]['respMat']    
                        maxInd = np.unravel_index(np.nanargmax(respMat), respMat.shape)                        
                        bestResp = respMat[maxInd[0],maxInd[1],:]
                        rho = np.append(bestResp, bestResp[0])
                        axp.plot(theta, rho, color=colors[it])
                        axp.set_rmax(np.nanmax(bestResp)*1.05)
                        axp.set_title(str(unit)+ pro +tag + ': ' + str(numTrials) + ' trials', fontdict={'size':10})
                
                fig = plt.figure(facecolor='w')
                ax = fig.add_subplot(111)
                for condi, cond in enumerate(sdfDict.keys()):
                    sdf = sdfDict[cond]
                    for cond2 in sdfDict.keys():
                        sdf2 = sdfDict[cond2]
                        sdf = nanMaskToMatch(sdf, sdf2)
                    ax.plot(np.nanmean(sdf, axis=(0, 1, 2)), color=colors[condi])
                    
            
            if pro is 'flash':
                fig = plt.figure(facecolor='w')
                axCreated = False
                sdfDict = {}
                for it, tag in enumerate(tagsToPlot):
                    saveTag = '_' + tag
                    if (pro + saveTag) in data:
                        numTrials = len(data[pro + saveTag]['trials'])                        
                        sdf = data['flash' + saveTag]['meanResp']
                        try:
                            sdfMax = max([np.nanmax(data[pro +'_'+tg]['_sdf']) for tg in tagsToPlot]) 
                        except:
                            sdfMax = np.nanmax(data[pro +'_'+tag]['_sdf'])
                        sdfTime = np.arange(sdf.shape[1])/1000.0
                        sdfDict[tag] = sdf
                        ax = None
                        if it > 0 and axCreated:
                            ax = plt.gca()
                        self.plotSDF1Axis(sdf, sdfTime, sdfMax, ax=ax, color=colors[it])
                        axCreated = True
                        ax = plt.gca()
                        ax.set_title(str(unit)+ pro +tag + ': ' + str(numTrials) + ' trials')
                
                fig = plt.figure(facecolor='w')
                ax = fig.add_subplot(111)
                for condi, cond in enumerate(sdfDict.keys()):
                    sdf = sdfDict[cond]
                    for cond2 in sdfDict.keys():
                        sdf2 = sdfDict[cond2]
                        sdf = nanMaskToMatch(sdf, sdf2)
                    ax.plot(np.nanmean(sdf, axis=0), lineColor = colors[condi])
                    
            if pro is 'checkerboard':
                fig = plt.figure(facecolor='w')
                gs = gridspec.GridSpec(1,len(tagsToPlot))  
                axes = []
                axCreated = False
                patchSpeed = data['checkerboard_' + tagsToPlot[0]]['patchSpeed']
                bckgndSpeed = data['checkerboard_' + tagsToPlot[0]]['bckgndSpeed']
                respMats = [data['checkerboard_' + tg]['respMat'] for tg in tagsToPlot]
                centerPoints = [rMat[patchSpeed==0,bckgndSpeed==0][0] if not np.isnan(rMat[patchSpeed==0,bckgndSpeed==0][0]) else np.nanmedian(rMat) for rMat in respMats]
                cLims = [np.nanmax(abs(rMat-centerPoints[r])) for r, rMat in enumerate(respMats)]
                for it, tag in enumerate(tagsToPlot):
                    saveTag = '_' + tag
                    if (pro + saveTag) in data:
                        numTrials = len(data[pro + saveTag]['trials'])       
                        sdf = data['checkerboard' + saveTag]['_sdf']    
                        try:
                            sdfMax = max([np.nanmax(data[pro +'_'+tg]['_sdf']) for tg in tagsToPlot]) 
                            sdfMin = min([np.nanmin(data[pro +'_'+tg]['_sdf']) for tg in tagsToPlot])
                        except:
                            sdfMax = np.nanmax(data[pro +'_'+tag]['_sdf'])
                            sdfMin = np.nanmin(data[pro +'_'+tag]['_sdf'])
                        sdfTime = data['checkerboard' + saveTag]['_sdfTime']    
                        maxInd = data['checkerboard' + saveTag]['bestPatchRespInd']    
                        bestSDF = sdf[:,:,maxInd[1], maxInd[2]]
                        ax = None
                        if it > 0 and axCreated:
                            ax = plt.gca()
                        self.plotSDF1Axis(bestSDF, sdfTime, sdfMax, ax=ax, lineColor=colors[it])
                        axCreated = True
                        respMat = data['checkerboard' + saveTag]['respMat']
                        ax = fig.add_subplot(gs[0, it])
                        im = ax.imshow(respMat,cmap=cmap,clim=(centerPoints[it]-max(cLims),centerPoints[it]+ max(cLims)),interpolation='none',origin='lower')
                        ax.set_title(str(unit)+' checkerboard: '+tag + ': ' + str(numTrials) + ' trials', fontdict={'size':10})
                        ax.set_xticks(range(bckgndSpeed.size))
                        ax.set_xticklabels(bckgndSpeed)
                        ax.set_yticks(range(patchSpeed.size))
                        ax.set_yticklabels(patchSpeed)
                        ax.set_xlabel('Background Speed')
                        if it==0:
                            ax.set_ylabel('Patch Speed')
                        axes.append(ax)
                        
                        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
#                            cb = plt.colorbar(im, ax=axes)
                        cb.ax.tick_params(length=0,labelsize='x-small')
                if pro is 'loom':
                    fig = plt.figure(facecolor='w')
                    ax = fig.add_subplot(111)
                    trialConditions = data['loom_'+tagsToPlot[0]]['trialConditions']
                    lvRatio = trialConditions[:, 0]                    
                    xpos = trialConditions[:,1]
                    ypos = trialConditions[:, 2]
                    loomColors = trialConditions[:,3]
                    
                    sdfs = [data['loom_' + tg]['_sdf'] for tg in tagsToPlot]
                    maxResp = max([np.nanmax(sdf) for sdf in sdfs])
                    maxSDF = np.argmax([np.nanmax(sdf) for sdf in sdfs])
                    bestCondition = np.argmax([np.nanmax(sdf) for sdf in sdfs[maxSDF]])
                    conditionsToPlot = [tc for tc in trialConditions if (tc[1:] == trialConditions[bestCondition][1:]).all()]
                    for it, tag in enumerate(tagsToPlot):
                        alphas = np.linspace(0.2, 1, np.unique(lvRatio).size)
                        sdf = sdfs[it]
                        for ic, c in enumerate(conditionsToPlot):
                            linecolor = colors[it]
                            alpha = alphas[ic]
                            thiscond = np.where([(c==tc).all() for tc in trialConditions])[0][0]
                            ax.plot(sdf[thiscond], color=linecolor, alpha=alpha)
                    ax.set_ylim([0, 1.1*maxResp])
                    ylims = ax.get_ylim()
                    ax.plot([sdf.shape[1]-400, sdf.shape[1]-400], [ylims[0], ylims[1]], 'k--')
                    formatFigure(fig, ax)
    

                            
    def getWaveforms(self):
        dataPath = os.path.dirname(os.path.dirname(self.kwdFileList[0]))
        self.mapChannels()
        
        #find channel positions
        imec_p2_positions = np.zeros((128,2))
        imec_p2_positions[:,0][::2] = 18
        imec_p2_positions[:,0][1::2] = 48
        imec_p2_positions[:,1] = np.floor(np.linspace(0,128,128)/2) * 20;imec_p2_positions[:,1][-1]=1260.
        
        #get waveforms for all the units
        waveforms = extractWaveforms.calculate_waveform(dataPath, clustersToAnalyze='good', numChannels=128, channelMap=self.channelMapping, returnWaveforms = True, save = False, saveFileDir = None, saveFileName='mean_waveforms.npy')

        #add waveforms to units dict and update xpos and ypos as position of channel with min        
        for u in self.units:        
            self.units[u].update(waveforms[u])
            minChannel = np.unravel_index(np.argmin(waveforms[u]['waveform']), waveforms[u]['waveform'].shape)[1]
            self.units[u]['xpos'] = imec_p2_positions[minChannel, 0]
            self.units[u]['ypos'] = imec_p2_positions[minChannel, 1]               

    
    def findAnalogPulses(self, channel, protocol, threshold=1000, refractoryPeriod=1000):
        data = self._d[protocol]['data'][:, channel]
        dataThresh = data>threshold
        
        pulseOn = np.where(dataThresh[1:]>dataThresh[:-1])[0]
        goodPoints = np.where(pulseOn[1:] - pulseOn[:-1] > refractoryPeriod)[0]
        goodPoints += 1
        goodPoints = np.insert(goodPoints, 0, 0)
        pulseStarts = pulseOn[goodPoints]
        
        dataThresh_rev = dataThresh[::-1]
        pulseOff = np.where(dataThresh_rev[1:]>dataThresh_rev[:-1])[0]
        goodPoints = np.where(pulseOff[1:] - pulseOff[:-1] > refractoryPeriod)[0]
        goodPoints += 1
        goodPoints = np.insert(goodPoints, 0, 0)
        pulseEnds = (dataThresh.size - pulseOff[goodPoints])[::-1]

        pulseAmps = np.array([np.mean(data[pulseStarts[u]:pulseEnds[u]]) for u, _ in enumerate(pulseStarts)])*self._d[protocol]['gains'][channel]
        
        self.behaviorData[str(protocol)][str(channel) + '_pulses'] = [pulseStarts, pulseEnds, pulseAmps]
        
        return pulseStarts, pulseEnds, pulseAmps


    def findLaserTrials(self, protocol):
        if isinstance(protocol, str):
            protocol = self.getProtocolIndex(protocol)
        
        v = self.visstimData[str(protocol)]
        trialLaser = None
        for p in ['laserPowerHistory', 'stimulusHistory_laserPower', 'trialLaserPower']:
            if p in v:
                if p is 'stimHistory':
                    trialLaser = v[p][6]
                else:
                    trialLaser = v[p]
            
        laserTrials = np.where(trialLaser>0)[0]
        key = 'trialStartFrame' if protocol=='checkerboard' else 'stimStartFrames'
        controlTrials = np.setdiff1d(np.arange(v[key].size), laserTrials)
        return laserTrials, controlTrials
    
    def findTTXTrials(self, protocolName):
        controlProtocol = self.getProtocolIndex(protocolName)
        ttxProtocol = self.getProtocolIndex(protocolName+'_TTX')
        if ttxProtocol is None:
            ttxProtocol = controlProtocol
        
        key = 'trialStartFrame' if protocolName=='checkerboard' else 'stimStartFrames'
        totalTrials = self.visstimData[str(controlProtocol)][key].size
        
        ttxDict = self.ttxDict[protocolName]
        controlStart = ttxDict['control_start'] if ttxDict['control_start']>0 else 0
        controlEnd = ttxDict['control_end'] if ttxDict['control_end']>0 else totalTrials
        controlTrials = np.arange(controlStart, controlEnd)
        
        totalTrials = self.visstimData[str(ttxProtocol)][key].size
        ttxStart = ttxDict['ttx_start'] if ttxDict['ttx_start']>0 else 0
        ttxEnd = ttxDict['ttx_end'] if ttxDict['ttx_end']>0 else totalTrials
        ttxTrials = np.arange(ttxStart, ttxEnd)
        
        return controlTrials, controlProtocol, ttxTrials, ttxProtocol
                
        
        
# utility functions

def getKwdInfo(dirPath=None):
    # kwdFiles, nSamples = getKwdInfo()
    # returns kwd file paths and number of samples in each file ordered by file start time
    if dirPath is None:    
        dirPath = fileIO.getDir(dataDir)
        if dirPath=='':
            return
    kwdFiles = []
    startTime = []
    nSamples = []
    for item in os.listdir(dirPath):
        itemPath = os.path.join(dirPath,item)
        if os.path.isdir(itemPath):
            for f in os.listdir(itemPath):
                if f[-4:]=='.kwd':
                    startTime.append(datetime.datetime.strptime(os.path.basename(itemPath)[:19],'%Y-%m-%d_%H-%M-%S'))
                    kwdFiles.append(os.path.join(itemPath,f))
                    kwd = h5py.File(kwdFiles[-1],'r')
                    nSamples.append(kwd['recordings']['0']['data'].shape[0])
    if len(kwdFiles)>1:
        kwdFiles,nSamples = zip(*[n[1:] for n in sorted(zip(startTime,kwdFiles,nSamples),key=lambda z: z[0])])
    return kwdFiles,nSamples


def makeDat(kwdFiles=None, saveDir=None, copyToSortComputer=True):
    if kwdFiles is None:
        kwdFiles, _ = getKwdInfo()
    dirPath = os.path.dirname(os.path.dirname(kwdFiles[0]))
    if saveDir is None:
        saveDir = dirPath
    datFilePath = os.path.join(saveDir,os.path.basename(dirPath)+'.dat')
    datFile = open(datFilePath,'wb')
    for filenum, filePath in enumerate(kwdFiles):
        print('Copying kwd file ' + str(filenum + 1) + ' of ' + str(len(kwdFiles)) + ' to dat file')
        kwd = h5py.File(filePath,'r')
        dset = kwd['recordings']['0']['data']
        i = 0
        while i<dset.shape[0]:
            (dset[i:i+dset.chunks[0],:128]).tofile(datFile)                        
            i += dset.chunks[0]
    datFile.close()
    if copyToSortComputer:    
        copyPath = r'\\10.128.38.3\data_local_1\corbett'
        print('Copying dat file to ' + copyPath)
        shutil.copy(datFilePath,copyPath)
    
    
def gauss2D(xyTuple,x0,y0,sigX,sigY,theta,amplitude,offset):
    x,y = xyTuple # (x,y)
    y = y[:,None]                                                                                                             
    a = (math.cos(theta)**2)/(2*sigX**2)+(math.sin(theta)**2)/(2*sigY**2)   
    b = (math.sin(2*theta))/(4*sigX**2)-(math.sin(2*theta))/(4*sigY**2)    
    c = (math.sin(theta)**2)/(2*sigX**2)+(math.cos(theta)**2)/(2*sigY**2)   
    z = offset + amplitude * np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))                                   
    return z.ravel()


def fitRF(x,y,data,initialParams,maxOffGrid):
    gridSize = max(x[-1]-x[0],y[-1]-y[0])
    lowerBounds = np.array([x[0]-maxOffGrid,y[0]-maxOffGrid,0,0,0,data.min(),data.min()])
    upperBounds = np.array([x[-1]+maxOffGrid,y[-1]+maxOffGrid,0.5*gridSize,0.5*gridSize,2*math.pi,1.5*data.max(),data.mean()])
    try:
        fitParams,fitCov = scipy.optimize.curve_fit(gauss2D,(x,y),data.flatten(),p0=initialParams,bounds=(lowerBounds,upperBounds))
    except Exception as e:
#        print('fit failed because of '+str(type(e)))
        return None,None
    if not all([lowerBounds[i]+1<fitParams[i]<upperBounds[i]-1 for i in (0,1)]):
        fitParams = fitError = None # if fit center on edge of boundaries
    else:
        fitData = gauss2D((x,y),*fitParams).reshape(y.size,x.size)
        fitError = np.sqrt(np.mean(np.square(data-fitData)))/data.max() # normalized root mean squared error
    return fitParams,fitError


def getEllipseXY(x,y,a,b,angle):
    sinx = np.sin(np.arange(0,361)*math.pi/180)
    cosx = np.cos(np.arange(0,361)*math.pi/180)
    X = x+a*cosx*math.cos(angle)-b*sinx*math.sin(angle)
    Y = y+a*cosx*math.sin(angle)+b*sinx*math.cos(angle)
    return X,Y
    

def stfLogGauss2D(stfTuple,sf0,tf0,sigSF,sigTF,speedTuningIndex,amplitude,offset):
    sf,tf = stfTuple
    tf = tf[:,None]
    z = offset + amplitude * np.exp(-((np.log2(sf)-np.log2(sf0))**2)/(2*sigSF**2)) * np.exp(-((np.log2(tf)-(speedTuningIndex*(np.log2(sf)-np.log2(sf0))+np.log2(tf0)))**2)/(2*sigTF**2))
    return z.ravel()


def fitStf(sf,tf,data,initialParams):
    lowerBounds = np.array([0.75*sf[0],0.75*tf[0],0.5,0.5,-0.5,0,data.min()])
    upperBounds = np.array([1.25*sf[-1],1.25*tf[-1],0.5*sf.size,0.5*tf.size,1.5,1.5*data.max(),np.median(data)])
    try:
        fitParams,fitCov = scipy.optimize.curve_fit(stfLogGauss2D,(sf,tf),data.flatten(),p0=initialParams,bounds=(lowerBounds,upperBounds))
    except Exception as e:
#        print('fit failed because of '+str(type(e)))
        return None,None
    fitData = stfLogGauss2D((sf,tf),*fitParams).reshape(tf.size,sf.size)
    fitError = np.sqrt(np.mean(np.square(data-fitData)))/data.max() # normalized root mean squared error
    return fitParams,fitError


def getStfContour(sf,tf,fitParams):
    intpPts = 100
    sfIntp = np.logspace(np.log2(sf[0]*0.5),np.log2(sf[-1]*2),intpPts,base=2)
    tfIntp = np.logspace(np.log2(tf[0]*0.5),np.log2(tf[-1]*2),intpPts,base=2)
    intpFit = stfLogGauss2D((sfIntp,tfIntp),*fitParams).reshape(intpPts,intpPts)
    intpFit -= intpFit.min()
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
    prefTheta = ((math.atan2(sumY,sumX)*180/math.pi)+360)%360
#    prefTheta = math.atan(sumY/sumX)*180/math.pi
#    if sumX<0:
#        if sumY>0:
#            prefTheta += 180
#        else:
#            prefTheta += 180
#    elif sumY<0:
#        prefTheta += 360
    return dsi, prefTheta


def gaussianConvolve3D(data, sigma):
    gaussian2DKernel = Gaussian2DKernel(stddev=sigma)
    data_c = np.zeros_like(data)
    for z in range(data.shape[2]):
        data_c[:, :, z] = convolve(data[:, :, z], gaussian2DKernel, boundary='extend')

    gaussian1DKernel = Gaussian1DKernel(stddev=sigma)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            data_c[x, y, :] = convolve(data_c[x, y, :], gaussian1DKernel, boundary='extend')
    
    return data_c    
    
def nanMaskToMatch(dataToMask, dataToMatch):
    mask = np.full_like(dataToMatch, np.nan)    
    mask[dataToMatch==dataToMatch] = 1
    masked = dataToMask * mask

    return masked
    
    
def formatFigure(fig, ax, title=None, xLabel=None, yLabel=None, xTickLabels=None, yTickLabels=None, blackBackground=False, saveName=None):
    fig.set_facecolor('w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    
    if title is not None:
        ax.set_title(title)
    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
        
    if blackBackground:
        ax.set_axis_bgcolor('k')
        ax.tick_params(labelcolor='w', color='w')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        for side in ('left','bottom'):
            ax.spines[side].set_color('w')

        fig.set_facecolor('k')
        fig.patch.set_facecolor('k')
    if saveName is not None:
        fig.savefig(saveName, facecolor=fig.get_facecolor())


def runningAverage(data, bins=10, xDataLabel = '', yDataLabel = '', plot=True):
    #if data is 1D take the index as x data
    if data.ndim == 1:
        xData = np.arange(len(data))
        yData = data
    elif data.ndim ==2:
        xData = data[:, 0]
        yData = data[:, 1]
    else:
        print('Data must be 1D or 2D (nX2)')
#        return
    xMin = np.nanmin(xData)
    xMax = np.nanmax(xData)
    stepSize = (xMax-xMin)/bins
    xBinned = np.digitize(xData, np.arange(xMin, xMax, stepSize))
    
    runningAverageY = []
    runningAverageX = []
    runningSEM = []
    for ib, b in enumerate(np.unique(xBinned)):
        thisBinIndex = xBinned == b
        runningAverageY.append(np.nanmean(yData[thisBinIndex]))
        runningSEM.append(np.nanstd(yData[thisBinIndex])/np.sum(~np.isnan(yData[thisBinIndex]))**0.5)
        runningAverageX.append(np.arange(xMin, xMax, stepSize)[ib] + stepSize/2)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(data[:,0], data[:,1], 'ko', alpha=0.2)
        ax.plot(runningAverageX, runningAverageY, 'ko')
#        ax.plot(runningAverageX, runningAverageY, 'k')
        ax.errorbar(runningAverageX, runningAverageY, runningSEM, color='k')
        formatFigure(fig, ax, '', xDataLabel, yDataLabel)
    
    return runningAverageX, runningAverageY, runningSEM, xBinned

    
    
    
    


if __name__=="__main__":
    pass       