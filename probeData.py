# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:19:20 2016

@author: SVC_CCG
"""

from __future__ import division
import fileIO
import copy, datetime, h5py, json, math, ntpath, os, shelve, shutil
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


dataDir = r'C:\Users\SVC_CCG\Desktop\Data'

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
    
    
    def getEyeTrackData(self, filePath=None, protocol=0):
        if filePath is None:        
            filePath = fileIO.getFile()
        
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
            frameSampleAdjustment = np.round((4.5/60.0) * self.sampleRate) 
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
     
               
    def findSpikesPerTrial(self, trialStarts, trialEnds, spikes): 
        spikesPerTrial = np.zeros(trialStarts.size)
        for trialNum in range(trialStarts.size):
            spikesPerTrial[trialNum] = np.count_nonzero(np.logical_and(spikes>=trialStarts[trialNum],spikes<=trialEnds[trialNum]))
        return spikesPerTrial
        
            
    def findRF(self, units=None, usePeakResp = True, sigma = 1, plot = True, minLatency = 0.05, maxLatency = 0.15, trials = None, protocol=None, fit=True, saveTag='', useCache=False):

        units, unitsYPos = self.getOrderedUnits(units)
        
        if protocol is None:
            protocol = self.getProtocolIndex('sparseNoise')
        protocol = str(protocol)
            
        if trials is None:
            trials = np.arange(self.visstimData[protocol]['stimStartFrames'].size-1)
        else:
            trials = np.array(trials)
        
        minLatencySamples = minLatency*self.sampleRate
        maxLatencySamples = maxLatency*self.sampleRate
        
        stimStartFrames = self.visstimData[protocol]['stimStartFrames'][trials]
        stimStartSamples = self.visstimData[protocol]['frameSamples'][stimStartFrames]
        
        posHistory = self.visstimData[protocol]['boxPositionHistory'][trials]
        xpos = np.unique(posHistory[:,0])
        ypos = np.unique(posHistory[:,1])
        pixPerDeg = self.visstimData[str(protocol)]['pixelsPerDeg']
        elev, azim = ypos/pixPerDeg, xpos/pixPerDeg
        gridExtent = self.visstimData[protocol]['gridBoundaries']
        
        colorHistory = self.visstimData[protocol]['boxColorHistory'][trials, 0]
        
        boxSizeHistory = self.visstimData[protocol]['boxSizeHistory'][trials]/pixPerDeg
        boxSize = self.visstimData[protocol]['boxSize']
        sizeTuningOn = np.full((len(units),boxSize.size),np.nan)
        sizeTuningOff = np.copy(sizeTuningOn)
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
            onFit = np.full((len(units),7),np.nan)
            offFit = np.copy(onFit)
        
        if plot:
            fig = plt.figure(figsize=(15,6*len(units)), facecolor='w')
            gs = gridspec.GridSpec(3*len(units), 6*len(boxSize))
        
        for uindex, unit in enumerate(units):
            spikes = self.units[unit]['times'][str(protocol)]
            gridOnSpikes = np.full((len(boxSize),ypos.size,xpos.size),np.nan)
            gridOffSpikes = np.copy(gridOnSpikes)
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
                            gridOnSpikes[sizeInd,i,j] = np.mean(self.findSpikesPerTrial(posOnSamples+minLatencySamples,posOnSamples+maxLatencySamples,spikes))
                            sdfOn[sizeInd,i,j,:],_ = self.getSDF(spikes,posOnSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
                        
                        posOffSamples = stimStartSamples[np.logical_and(posOffTrials,boxSizeTrials)]
                        if any(posOffSamples):                            
                            gridOffSpikes[sizeInd,i,j] = np.mean(self.findSpikesPerTrial(posOffSamples+minLatencySamples,posOffSamples+maxLatencySamples,spikes))
                            sdfOff[sizeInd,i,j,:],sdfTime = self.getSDF(spikes,posOffSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
            
            # convert spike count to spike rate
            gridOnSpikes /= maxLatency-minLatency
            gridOffSpikes /= maxLatency-minLatency
            
            # get full field flash resp
            fullFieldOnSpikes = fullFieldOffSpikes = np.nan
            fullFieldTrials = boxSizeHistory>100
            if any(fullFieldTrials):
                ffOnSamples = stimStartSamples[np.logical_and(fullFieldTrials,colorHistory==1)]
                if any(ffOnSamples):
                    fullFieldOnSpikes = np.mean(self.findSpikesPerTrial(ffOnSamples+minLatencySamples,ffOnSamples+maxLatencySamples,spikes))
                    fullFieldOnSpikes /= maxLatency-minLatency
                    fullFieldOnSDF,_ = self.getSDF(spikes,ffOnSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
                ffOffSamples = stimStartSamples[np.logical_and(fullFieldTrials,colorHistory==-1)]
                if any(ffOffSamples):
                    fullFieldOffSpikes = np.mean(self.findSpikesPerTrial(ffOffSamples+minLatencySamples,ffOffSamples+maxLatencySamples,spikes))
                    fullFieldOffSpikes /= maxLatency-minLatency
                    fullFieldOffSDF,_ = self.getSDF(spikes,ffOffSamples-minLatencySamples,sdfSamples,sdfSigma,sdfSampInt)
            
            # optionally use peak resp instead of mean rate
            inAnalysisWindow = np.logical_and(sdfTime>minLatency*2,sdfTime<minLatency+maxLatency)
            if usePeakResp:
                gridOnSpikes = np.nanmax(sdfOn[:,:,:,inAnalysisWindow],axis=3)
                gridOffSpikes = np.nanmax(sdfOff[:,:,:,inAnalysisWindow],axis=3)
                if not np.isnan(fullFieldOnSpikes):
                    fullFieldOnSpikes = np.nanmax(fullFieldOnSDF[inAnalysisWindow])
                if not np.isnan(fullFieldOffSpikes):
                    fullFieldOffSpikes = np.nanmax(fullFieldOffSDF[inAnalysisWindow])
                        
            # calculate size tuning
            sizeTuningOn[uindex,:boxSize.size] = np.nanmax(np.nanmax(gridOnSpikes,axis=2),axis=1)
            sizeTuningOff[uindex,:boxSize.size] = np.nanmax(np.nanmax(gridOffSpikes,axis=2),axis=1)
            if any(fullFieldTrials):
                sizeTuningOn[uindex,-1] = fullFieldOnSpikes
                sizeTuningOff[uindex,-1] = fullFieldOffSpikes
            
            # estimate spontRate using random trials and interval 0:minLatency
            nTrialTypes = xpos.size*ypos.size*boxSize.size*2
            nTrials = int(np.count_nonzero(boxSizeHistory<100)/nTrialTypes)
            spontRate = np.zeros(nTrialTypes)
            for n in range(nTrialTypes):
                randTrials = np.random.choice(trials,nTrials,replace=False)
                spontRate[n] = np.max(self.getSDF(spikes,stimStartSamples[randTrials],minLatencySamples,sdfSigma,sdfSampInt))
            peakSpontRateMean = spontRate.mean()
            peakSpontRateStd = spontRate.std()
            if usePeakResp:
                spontRateMean,spontRateStd = peakSpontRateMean,peakSpontRateStd
            else:
                spontRate = self.findSpikesPerTrial(stimStartSamples,stimStartSamples+minLatencySamples,spikes)/(minLatency)
                spontRateMean = spontRate.mean()
                spontRateStd = spontRate.std()            
            
            # average significant responses                
            respThresh = spontRateMean+5*spontRateStd
            hasOnResp = np.zeros(len(boxSize),dtype=bool)
            hasOffResp = np.copy(hasOnResp)
            for sizeInd,_ in enumerate(boxSize):
                hasOnResp[sizeInd] = np.nanmax(gridOnSpikes[sizeInd])>respThresh
                hasOffResp[sizeInd] = np.nanmax(gridOffSpikes[sizeInd])>respThresh
            meanOnResp = np.nanmean(gridOnSpikes[hasOnResp],axis=0) if any(hasOnResp) else None
            meanOffResp = np.nanmean(gridOffSpikes[hasOffResp],axis=0) if any(hasOffResp) else None
            
            # filter mean response
            if meanOnResp is not None:
                meanOnResp = convolve(meanOnResp, gaussianKernel, boundary='extend')
            if meanOffResp is not None:
                meanOffResp = convolve(meanOffResp, gaussianKernel, boundary='extend')
            
            # fit mean response
            fitParams = []
            if fit:
                maxOffGrid = 15
                for resp in (meanOnResp,meanOffResp):
                    # params: x0 , y0, sigX, sigY, theta, amplitude, offset
                    if resp is None or np.any(np.isnan(resp)):
                        fitParams.append(None)
                    else:
                        i,j = np.unravel_index(np.argmax(resp),resp.shape)
                        sigmaGuess = (azim[1]-azim[0])*0.5*np.sqrt(np.count_nonzero(resp>resp.min()+0.5*(resp.max()-resp.min())))
                        initialParams = (azim[j],elev[i],sigmaGuess,sigmaGuess,0,resp.max(),np.percentile(resp,10))
                        fitParams.append(fitRF(azim,elev,resp,initialParams,maxOffGrid))
            if fitParams[0] is not None:
                onFit[uindex] = fitParams[0]
            if fitParams[1] is not None:
                offFit[uindex] = fitParams[1]
                
            # compare on and off resp magnitude (max across all box sizes)
            onMax = np.nanmax(gridOnSpikes)
            offMax = np.nanmax(gridOffSpikes)
            onVsOff[uindex] = (onMax-offMax)/(onMax+offMax)
            
            # calculate response latency and duration
            # SDF time is minLatency before stim onset through 2*maxLatency
            # Hence stim starts at minLatency and analysisWindow starts at 2*minLatency
            # Search analysisWindow for peak but allow searching outside analaysisWindow for halfMax
            sdfMaxInd = np.zeros((2,4),dtype=int)
            halfMaxInd = np.zeros((2,2),dtype=int)
            respLatencyInd = np.zeros(2,dtype=int)
            latencyThresh = peakSpontRateMean+5*peakSpontRateStd
            for i,sdf in enumerate((sdfOn,sdfOff)):
                if not np.any(sdf[:,:,:,inAnalysisWindow]>latencyThresh):
                    continue
                sdfMaxInd[i,:] = np.unravel_index(np.nanargmax(sdf[:,:,:,inAnalysisWindow]),sdf[:,:,:,inAnalysisWindow].shape)
                sdfMaxInd[i,3] += np.where(inAnalysisWindow)[0][0]
                bestSDF = np.copy(sdf[sdfMaxInd[i,0],sdfMaxInd[i,1],sdfMaxInd[i,2],:])
                maxInd = sdfMaxInd[i,3]
                # find last 3*std cross before peak for latency
                respLatencyInd[i] = np.where(bestSDF[:maxInd]<latencyThresh)[0][-1]+1
                respLatency[uindex,i] = respLatencyInd[i]*sdfSampInt-minLatency
                # subtract min for calculating resp duration
                bestSDF -= np.min(bestSDF[inAnalysisWindow])
                # respNormArea = (area under SDF in analysisWindow) / (peak * analysisWindow duration)
                respNormArea[uindex,i] = np.trapz(bestSDF[inAnalysisWindow])*sdfSampInt/(bestSDF[maxInd]*(maxLatency-minLatency))                 
                # find last half-max cross before peak
                halfMax = 0.5*bestSDF[maxInd]
                halfMaxInd[i,0] = np.where(bestSDF[:maxInd]<halfMax)[0][-1]+1
                # find first half-max cross after peak
                postHalfMax = np.where(bestSDF[maxInd:]<halfMax)[0]
                halfMaxInd[i,1] = maxInd+postHalfMax[0]-1 if any(postHalfMax) else bestSDF.size
                respHalfWidth[uindex,i] = (halfMaxInd[i,1]-halfMaxInd[i,0])*sdfSampInt
            
            # cache results
            self.units[unit]['sparseNoise' + saveTag] = {'gridExtent': gridExtent,
                                                         'elev': elev,
                                                         'azim': azim,
                                                         'boxSize': boxSize,
                                                         'gridOnSpikes': gridOnSpikes,
                                                         'gridOffSpikes': gridOffSpikes,
                                                         'spontRateMean': spontRateMean,
                                                         'spontRateStd': spontRateStd,
                                                         'meanOnResp': meanOnResp,
                                                         'meanOffResp': meanOffResp,
                                                         'onFit': onFit[uindex],
                                                         'offFit': offFit[uindex],
                                                         'sizeTuningOn': sizeTuningOn[uindex],
                                                         'sizeTuningOff': sizeTuningOff[uindex],
                                                         'onVsOff': onVsOff[uindex],
                                                         'respLatency': respLatency[uindex],
                                                         'respNormArea': respNormArea[uindex],
                                                         'respHalfWidth': respHalfWidth[uindex]}
            
            if plot:
                maxVal = max(np.nanmax(gridOnSpikes), np.nanmax(gridOffSpikes))
                minVal = min(np.nanmin(gridOnSpikes), np.nanmin(gridOffSpikes))
                sdfMax = max(np.nanmax(sdfOn),np.nanmax(sdfOff))
                spacing = 0.2
                sdfXMax = sdfTime[-1]
                sdfYMax = sdfMax
                for sizeInd,size in enumerate(boxSize):
                    for ind,(sdf,resp,hasResp,onOrOff) in enumerate(zip((sdfOn[sizeInd],sdfOff[sizeInd]),(gridOnSpikes[sizeInd],gridOffSpikes[sizeInd]),(hasOnResp[sizeInd],hasOffResp[sizeInd]),('On','Off'))):
                        axIndex = sizeInd*6+ind*3
                        ax = fig.add_subplot(gs[uindex*3,axIndex:axIndex+2])
                        x = 0
                        y = 0
                        for i,_ in enumerate(ypos):
                            for j,_ in enumerate(xpos):
                                ax.plot(x+sdfTime,y+sdf[i,j,:],color='k')
                                if not np.isnan(respLatency[uindex,ind]) and all((sizeInd,i,j)==sdfMaxInd[ind,:3]):
                                    ax.plot(x+sdfTime[halfMaxInd[ind]],y+sdf[i,j,halfMaxInd[ind]],color='r',linewidth=2)
                                    ax.plot(x+sdfTime[respLatencyInd[ind]],y+sdf[i,j,respLatencyInd[ind]],'bo')
                                x += sdfXMax*(1+spacing)
                            x = 0
                            y += sdfYMax*(1+spacing)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                        ax.set_xticks([minLatency,minLatency+0.1])
                        ax.set_xticklabels(['','100 ms'])
                        ax.set_yticks([0,int(sdfMax)])
                        ax.set_xlim([-sdfXMax*spacing,sdfXMax*(1+spacing)*xpos.size])
                        ax.set_ylim([-sdfYMax*spacing,sdfYMax*(1+spacing)*ypos.size])
                        if sizeInd==0 and ind==0:
                            ax.set_ylabel('Unit '+str(unit), fontsize='medium')
                        if uindex==0:
                            ax.set_title(onOrOff,fontsize='medium')                    
                        
                        ax = fig.add_subplot(gs[uindex*3,axIndex+2])
                        im = ax.imshow(resp, cmap='jet', clim=(minVal,maxVal), interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]])
                        ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        title = '' if hasResp else 'no resp'
                        ax.set_title(title,fontsize='x-small')
                
                minOn,maxOn = (0,0) if meanOnResp is None else (np.nanmin(meanOnResp),np.nanmax(meanOnResp))
                minOff,maxOff = (0,0) if meanOffResp is None else (np.nanmin(meanOffResp),np.nanmax(meanOffResp))
                minVal,maxVal = min(minOn,minOff),max(maxOn,maxOff)
                for ind,(resp,fitParams,onOrOff) in enumerate(zip((meanOnResp,meanOffResp),(onFit[uindex],offFit[uindex]),('On','Off'))):
                    ax = fig.add_subplot(gs[uindex*3+1:uindex*3+3,ind*3+1:ind*3+3])
                    if resp is None:
                        ax.set_title('no '+onOrOff+' resp',fontsize='small')
                        ax.set_axis_off()
                    else:
                        im = ax.imshow(resp, cmap='jet', clim=(minVal,maxVal), interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]])
                        if not all(np.isnan(fitParams)):
                            ax.plot(fitParams[0],fitParams[1],'kx',markeredgewidth=2)
                            fitX,fitY = getEllipseXY(*fitParams[:-2])
                            ax.plot(fitX,fitY,'k',linewidth=2)
                            ax.set_xlim(gridExtent[[0,2]]-0.5)
                            ax.set_ylim(gridExtent[[1,3]]-0.5)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                        ax.set_xticks(np.round(azim[[0,-1]]))
                        ax.set_yticks(np.round(elev[[0,-1]]))
                        ax.set_title('mean '+onOrOff+' resp',fontsize='small')
                        cb = plt.colorbar(im, ax=ax, fraction=0.05, shrink=0.5, pad=0.04)
                        cb.ax.tick_params(length=0,labelsize='x-small')
                        cb.set_ticks([math.ceil(minVal),int(maxVal)])
                
                if len(boxSize)>1:
                    if any(fullFieldTrials):
                        sizeTuningSize = np.append(boxSize,boxSize[-1]*2.5)
                        sizeTuningLabel = list(sizeTuningSize)
                        sizeTuningLabel[-1] = 'full'
                    else:
                        sizeTuningSize = sizeTuningLabel = boxSize
                    ax = fig.add_subplot(gs[uindex*3+1:uindex*3+3,7:11])
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
                    
        if plot and len(units)>1:
            
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
                        ax.set_ylabel('Norm Spikes/s',fontsize='small')
                    else:
                        ax.set_yticklabels([])
                    ax.set_title(onOrOff,fontsize='medium')
                    
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
                    ax.set_xlabel('Size',fontsize='small')
                    if ind==0:
                        ax.set_ylabel('Best Size Count',fontsize='small')
            
            plt.figure(facecolor='w')
            gspec = gridspec.GridSpec(3,2)
            for i,(ydata,ylabel) in enumerate(zip((respLatency,respNormArea,respHalfWidth),('Latency','Resp Area','Resp Half-width'))):
                for j,title in enumerate(('On','Off')):                
                    ax = plt.subplot(gspec[i,j])
                    ax.plot(onVsOff,ydata[:,j],'ko',markerfacecolor='none')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                    ax.set_xlim([-1,1])
                    ax.set_ylim([0,1.1*np.nanmax(ydata)])
                    ax.set_xticks([-1,0,1])
                    if i==0:
                        ax.set_title(title,fontsize='small')
                    if i==2:
                        ax.set_xlabel('On vs Off Index',fontsize='small')
                    else:
                        ax.set_xticklabels([])
                    if j==0:
                        ax.set_ylabel(ylabel,fontsize='small')
                    else:
                        ax.set_yticklabels([])
            
            if fit:
                plt.figure(facecolor='w')
                ax = plt.subplot(1,1,1)
                ax.plot(gridExtent[[0,2,2,0,0]],gridExtent[[1,1,3,3,1]],color='0.6')
                ax.plot(onFit[:,0],onFit[:,1],'o',markeredgecolor='r',markerfacecolor='none')
                ax.plot(offFit[:,0],offFit[:,1],'o',markeredgecolor='b',markerfacecolor='none')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                ax.set_xlim(gridExtent[[0,2]]+[-maxOffGrid,maxOffGrid])
                ax.set_ylim(gridExtent[[1,3]]+[-maxOffGrid,maxOffGrid])
                ax.set_xlabel('Azimuth',fontsize='small')
                ax.set_ylabel('Elevation',fontsize='small')
                ax.set_title('RF center (red = on, blue = off)',fontsize='small')
                
                # comparison of RF and probe position
                plt.figure(facecolor='w')
                gspec = gridspec.GridSpec(2,2)
                unitsYPos = np.array(unitsYPos)
                xlim = np.array([min(unitsYPos)-10,max(unitsYPos)+10])
                for j,(rfCenters,onOrOff) in enumerate(zip((onFit,offFit),('On','Off'))):
                    for i,azimOrElev in enumerate(('Azimuth','Elevation')):
                        ax = plt.subplot(gspec[i,j])
                        hasRF = np.logical_not(np.isnan(rfCenters[:,i]))
                        if np.count_nonzero(hasRF)>1:
                            # linFit = (slope, intercept, r-value, p-value, stderror)
                            linFit = scipy.stats.linregress(unitsYPos[hasRF],rfCenters[hasRF,i])
                            ax.plot(xlim,linFit[0]*xlim+linFit[1],color='0.6')
                            ax.text(0.5,0.95,'$\mathregular{r^2}$ = '+str(round(linFit[2]**2,2))+', p = '+str(round(linFit[3],2)),
                                    transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',fontsize='xx-small',color='0.6')
                        ax.plot(unitsYPos,rfCenters[:,i],'ko',markerfacecolor='none')
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                        ax.set_xlim(xlim)
                        if i==0:
                            ax.set_title(onOrOff,fontsize='medium')
                            ax.set_ylim(gridExtent[[0,2]]+[-maxOffGrid,maxOffGrid])
                            ax.set_xticklabels([])
                        else:
                            ax.set_xlabel('Probe Y Pos',fontsize='medium')
                            ax.set_ylim(gridExtent[[1,3]]+[-maxOffGrid,maxOffGrid])
                        if j==0:
                            ax.set_ylabel(azimOrElev,fontsize='medium')
                        else:
                            ax.set_yticklabels([])
                            
                # comparision of RF position and area
                plt.figure(facecolor='w')
                gspec = gridspec.GridSpec(2,2)
                rfArea = np.full((len(units),2),np.nan)
                rfArea[:,0] = np.pi*np.prod(onFit[:,2:4],axis=1)
                rfArea[:,1] = np.pi*np.prod(offFit[:,2:4],axis=1)
                for i,(rfCenters,onOrOff) in enumerate(zip((onFit,offFit),('On','Off'))):
                    for j,azimOrElev in enumerate(('Azimuth','Elevation')):
                        ax = plt.subplot(gspec[i,j])
                        hasRF = np.logical_not(np.isnan(rfCenters[:,j]))
                        xlim = gridExtent[[0,2]] if j==0 else gridExtent[[1,3]]
                        xlim += [-maxOffGrid,maxOffGrid]
                        if np.count_nonzero(hasRF)>1:
                            # linFit = (slope, intercept, r-value, p-value, stderror)
                            linFit = scipy.stats.linregress(rfCenters[hasRF,j],rfArea[hasRF,i])
                            ax.plot(xlim,linFit[0]*xlim+linFit[1],color='0.6')
                            ax.text(0.5,0.95,'$\mathregular{r^2}$ = '+str(round(linFit[2]**2,2))+', p = '+str(round(linFit[3],2)),
                                    transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',fontsize='xx-small',color='0.6')
                        ax.plot(rfCenters[:,j],rfArea[:,i],'ko',markerfacecolor='none')
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
                        ax.set_xlim(xlim)
                        ax.set_ylim([0,np.nanmax(rfArea)*1.1])
                        if i==0:
                            ax.set_xticklabels([])
                        else:
                            ax.set_xlabel(azimOrElev,fontsize='medium')
                        if j==0:
                            ax.set_ylabel(onOrOff+' RF Area',fontsize='medium')
                        else:
                            ax.set_yticklabels([])
    
    
    def analyzeFlash(self, units=None, trials=None, protocol=None, responseLatency=0.25, plot=True, sdfSigma=0.005):
        units, unitsYPos = self.getOrderedUnits(units) 
            
        if protocol is None:
            label = 'flash'
            protocol = self.getProtocolIndex(label)
            
        protocol = str(protocol)
        
        if trials is None:
            trials = np.arange(self.visstimData[str(protocol)]['stimStartFrames'].size-1)
        else:
            trials = np.array(trials)
        
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
                if any(lumTrials):
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
     
              
    def analyzeGratings(self, units=None, trials=None, responseLatency=0.25, usePeakResp=True, plot=True, protocol=None, protocolType='stf', fit=True, saveTag='', useCache=False):
    
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
        sdfSamples = round((preTime+stimTime+postTime)*self.sampleRate)
        sdfSigma = 0.02
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
            gridWidth = 3*len(ori)+1 if protocolType=='stf' else len(tf)*len(sf)+1
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
            preTrialMat = np.copy(respMat)
            sdf = np.full((len(tf),len(sf),len(ori),round(sdfSamples/self.sampleRate/sdfSampInt)),np.nan)
            f1Mat = np.full((len(tf),len(sf),len(ori)),np.nan)
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
                            sdf[tfInd,sfInd,oriInd,:],sdfTime = self.getSDF(spikes,trialStartSamples[trialIndex]-int(preTime*self.sampleRate),sdfSamples,sigma=sdfSigma,sampInt=sdfSampInt)
                            if inAnalysisWindow is None:
                                inAnalysisWindow = np.logical_and(sdfTime>preTime+responseLatency,sdfTime<preTime+stimTime)
                            s = sdf[tfInd,sfInd,oriInd,inAnalysisWindow]
                            f,pwr = scipy.signal.welch(s,1/sdfTime[1],nperseg=s.size,detrend='constant',scaling='spectrum')
                            pwr **= 0.5
                            f1Ind = np.argmin(np.absolute(f-thisTF))
                            f1Mat[tfInd,sfInd,oriInd] = pwr[f1Ind-1:f1Ind+2].max()/s.mean()
            
            # calculate spontRate from gray screen trials
            grayTrials = trialContrast<0+tol
            if usePeakResp:
                respMat = np.nanmax(sdf[:,:,:,inAnalysisWindow],axis=3)
                spontRateMean,spontRateStd = self.getSDFNoise(spikes,trialStartSamples[grayTrials],max(trialEndSamples[grayTrials]-trialStartSamples[grayTrials]),sigma=sdfSigma,sampInt=sdfSampInt)
            else:
                spontRateMean = trialResp[grayTrials].mean()
                spontRateStd = trialResp[grayTrials].std()
            
            # find significant responses
            hasResp = respMat>spontRateMean+5*spontRateStd
            tfHasResp,sfHasResp,oriHasResp = [np.unique(np.where(hasResp)[i]) for i in (0,1,2)]
            
            meanResp = None
            if protocolType=='stf':
                # fit stf matrix for avg of ori's with resp
                if fit and oriHasResp.size>0:
                    # params: sf0 , tf0, sigSF, sigTF, speedTuningIndex, amplitude, offset
                    meanResp = np.nanmean(respMat[:,:,oriHasResp],axis=2)
                    i,j = np.unravel_index(np.argmax(meanResp),meanResp.shape)
                    initialParams = (sf[j], tf[i], 1, 1, 0.5, meanResp.max(), meanResp.min())
                    fitPrms = fitStf(sf,tf,meanResp,initialParams)
                    if fitPrms is not None:
                        stfFitParams[uindex] = fitPrms
            elif protocolType=='ori':
                # calculate DSI and OSI for avg of sf/tf's with resp
                if tfHasResp.size>0 and sfHasResp.size>0:
                    meanResp = np.nanmean(np.nanmean(respMat[tfHasResp,:,:],axis=0)[sfHasResp,:],axis=0)
                    dsi[uindex],prefDir[uindex] = getDSI(meanResp,ori)
                    osi[uindex],prefOri[uindex] = getDSI(meanResp,2*ori)
                    prefOri[uindex] /= 2
            
            # cache results
            tag = 'gratings_' + protocolType + saveTag
            self.units[str(unit)][tag] = {'sf': sf,
                                          'tf': tf,
                                          'ori': ori,
                                          'spontRateMean': spontRateMean,
                                          'spontRateStd': spontRateStd,
                                          'respMat': respMat}
            if protocolType=='stf':
                self.units[str(unit)][tag]['stfFitParams'] = stfFitParams[uindex]
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
                    
                    ax = fig.add_subplot(gs[uindex,-1])
                    if meanResp is not None:
                        im = ax.imshow(meanResp, clim=(centerPoint-cLim, centerPoint+cLim), cmap='bwr', origin = 'lower', interpolation='none')
                        if fit and not all(np.isnan(stfFitParams[uindex])):
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
                        ax.set_title('mean',fontsize='x-small')
                        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
                        cb.set_ticks([math.ceil(centerPoint-cLim),round(centerPoint),math.floor(centerPoint+cLim)])
                        cb.ax.tick_params(length=0,labelsize='xx-small')
                    else:
                        ax.set_axis_off()
                        ax.set_title('no resp',fontsize='small')
                        
                else:
                    for i,_ in enumerate(tf):
                        for j,_ in enumerate(sf):
                            ax = fig.add_subplot(gs[uindex,i*len(tf)+j], projection='polar')
                            theta = ori * (np.pi/180.0)
                            theta = np.append(theta, theta[0])
                            rho = np.append(respMat[i,j,:], respMat[i,j,0])
                            ax.plot(theta, rho)
                            ax.set_rmax(np.nanmax(respMat)*1.05)
                            if i==0 and j==0:
                                ax.set_ylabel('Unit '+str(unit))
                    
                    ax = fig.add_subplot(gs[uindex,i*len(tf)+j+1], projection='polar')
                    if meanResp is not None:
                        theta = ori * (np.pi/180.0)
                        theta = np.append(theta, theta[0])
                        rho = np.append(meanResp,meanResp[0])
                        ax.plot(theta, rho)
                        ax.set_rmax(np.nanmax(meanResp)*1.05)
                        ax.set_title('meanResp'+'\n'+'DSI = '+str(round(dsi[uindex],2))+', prefDir = '+str(round(prefDir[uindex]))+'\n'+'OSI = '+str(round(osi[uindex],2))+', prefOri = '+str(round(prefOri[uindex])),fontsize='x-small')
                    else:
                        ax.set_axis_off()
                        ax.set_title('no resp',fontsize='small')
                        
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

                                    
    def analyzeCheckerboard(self, units=None, protocol=None, trials=None, usePeakResp=True, plot=True, saveTag='', useCache=False):
        
        units, unitsYPos = self.getOrderedUnits(units)
        
        if protocol is None:
            protocol = self.getProtocolIndex('checkerboard')
        protocol = str(protocol)          
        p = self.visstimData[protocol]
        assert(set(p['bckgndDir'])=={0,180} and set(p['patchDir'])=={0,180} and 0 in p['bckgndSpeed'] and 0 in p['patchSpeed'])
        
        trialStartFrame = p['trialStartFrame']
        trialNumFrames = p['trialNumFrames'].astype(int)
        trialEndFrame = trialStartFrame+trialNumFrames
        lastFullTrial = np.where(trialEndFrame<p['frameSamples'].size)[0][-1]
        if trials is None:
            trials = np.arange(lastFullTrial+1)
        trialStartSamples = p['frameSamples'][trialStartFrame[trials]]
        trialEndSamples = p['frameSamples'][trialEndFrame[trials]]
        minInterTrialTime = p['interTrialInterval'][0]
        minInterTrialSamples = int(minInterTrialTime*self.sampleRate)
        latency = 0.25
        latencySamples = int(latency*self.sampleRate)
        
        bckgndSpeed = np.concatenate((-p['bckgndSpeed'][:0:-1],p['bckgndSpeed']))
        patchSpeed = np.concatenate((-p['patchSpeed'][:0:-1],p['patchSpeed']))
        
        sdfSigma = 0.1
        sdfSampInt = 0.001
        sdfTime = np.arange(0,2*minInterTrialTime+max(trialEndSamples-trialStartSamples)/self.sampleRate,sdfSampInt)
        
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
                                peakResp[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd] = s[np.logical_and(t>minInterTrialTime+latency,t<t[-1]-minInterTrialTime)].max()
                                sdf[pchSpeedInd,bckSpeedInd,pchSizeInd,pchElevInd,:s.size] = s
            
            # fill in resp for patch and bckgnd speeds not tested for every patch size and elevation
            for r in (meanResp,peakResp,sdf):
                for pchSizeInd,_ in enumerate(p['patchSize']):
                    for pchElevInd,_ in enumerate(p['patchElevation']):
                        r[patchSpeed==0,:,pchSizeInd,pchElevInd] = r[patchSpeed==0,:,0,0]
                for pchSpeedInd,pchSpeed in enumerate(patchSpeed):
                    for bckSpeedInd,bckSpeed in enumerate(bckgndSpeed):
                        if pchSpeed==bckSpeed:
                            r[pchSpeedInd,bckSpeedInd] = r[patchSpeed==0,bckSpeedInd]
            
            # get spont rate and find best resp over all patch sizes and elevations
            spontTrials = np.logical_and(p['trialBckgndSpeed'][trials]==0,p['trialPatchSpeed'][trials]==0)
            if usePeakResp:
                spontRateMean,spontRateStd = self.getSDFNoise(spikes,trialStartSamples[spontTrials],max(trialEndSamples[spontTrials]-trialStartSamples[spontTrials]),sigma=sdfSigma,sampInt=sdfSampInt)
                respMat = peakResp.copy()
            else:
                spontRateMean = np.nanmean(trialSpikeRate[spontTrials])
                spontRateStd = np.nanstd(trialSpikeRate[spontTrials])
                respMat = meanResp.copy()
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
                                                            'respMat': respMat}
            
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
                        x += sdfXMax*(1+spacing)
                    x = 0
                    y += sdfYMax*(1+spacing)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                ax.set_xticks([minInterTrialTime,sdfTime[-1]-2*minInterTrialTime])
                ax.set_xticklabels(['0',str(int(sdfTime[-1]-2*minInterTrialTime))+' s'])
                ax.set_yticks([0,int(sdfYMax)])
                ax.set_xlim([-sdfXMax*spacing,sdfXMax*(1+spacing)*bckgndSpeed.size])
                ax.set_ylim([-sdfYMax*spacing,sdfYMax*(1+spacing)*patchSpeed.size])
                ax.set_ylabel('Unit '+str(u), fontsize='medium')
                
                ax = fig.add_subplot(gs[uInd,2])
                centerPoint = respMat[patchSpeed==0,bckgndSpeed==0][0] if not np.isnan(respMat[patchSpeed==0,bckgndSpeed==0][0]) else np.nanmedian(respMat)
                cLim = np.nanmax(abs(respMat-centerPoint))
                im = ax.imshow(respMat,cmap='bwr',clim=(centerPoint-cLim,centerPoint+cLim),interpolation='none',origin='lower')
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='x-small')
                ax.set_xticks(range(bckgndSpeed.size))
                ax.set_xticklabels(bckgndSpeed)
                ax.set_yticks(range(patchSpeed.size))
                ax.set_yticklabels(patchSpeed)
                ax.set_xlabel('Background Speed')
                ax.set_ylabel('Patch Speed')
                ax.set_title('Spikes/s',fontsize='small')
                cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
                cb.set_ticks([math.ceil(centerPoint-cLim),round(centerPoint),math.floor(centerPoint+cLim)])
                cb.ax.tick_params(length=0,labelsize='xx-small')
    
    
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
        if not 'running' in self.behaviorData[str(protocol)]:
            self.decodeWheel(self.d[protocol]['data'][::500, self.wheelChannel]*self.d[protocol]['gains'][self.wheelChannel])
        
        wheelData = -self.behaviorData[str(protocol)]['running']
        
        if trialStarts is not None:
            runningTrials = []
            stationaryTrials = []
            for trial in range(trialStarts.size):
                trialSpeed = np.mean(wheelData[round(trialStarts[trial]/wheelDownsampleFactor):round(trialEnds[trial]/wheelDownsampleFactor)])
                if trialSpeed >= runThresh:
                    runningTrials.append(trial)
                elif trialSpeed <= statThresh:
                    stationaryTrials.append(trial)
        return stationaryTrials, runningTrials, trialSpeed
    
    
    def analyzeRunning(self, units, protocol, plot=True):
        
        units, unitsYPos = self.getOrderedUnits(units)
        
        if plot:
            plt.figure(figsize=(10,3*len(units)),facecolor='w')
            gs = gridspec.GridSpec(len(units),1)
                        
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
            
            fr_binned = np.array(fr_binned)
            fr_std = np.array(fr_std)
            self.units[str(u)]['runModulation'] = {}
            self.units[str(u)]['runModulation'][str(protocol)] = [speedBins, fr_binned, fr_std] 
            
            if plot:
                ax = plt.subplot(gs[uindex, 0])
                ax.plot(speedBins, fr_binned)
#                ax.set_xscale('log', basex=2)
                plt.fill_between(speedBins, fr_binned+fr_std, fr_binned-fr_std, alpha=0.3)
    
    
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
            nReps = int(windowSamples/self.sampleRate/sampInt)
        bufferTime = sigma*5
        bufferSamples = int(bufferTime*self.sampleRate)
        sdf,t = self.getSDF(spikes,startSamples-bufferSamples,windowSamples+2*bufferSamples,sigma=sigma,sampInt=sampInt,avg=False)
        sdf = sdf[:,np.logical_and(t>bufferTime,t<t[-1]-bufferTime)]
        peaks = np.zeros(nReps)
        for n in range(nReps):
            for i,_ in enumerate(sdf):
                sdf[i] = np.roll(sdf[i],np.random.randint(0,sdf.shape[1]))
            peaks[n] = sdf.mean(axis=0).max()
        return peaks.mean(),peaks.std()
    
    
    def plotRaster(self,unit,protocol,startSamples=None,offset=0,windowDur=None,paramNames=None,paramColors=None,grid=False):
        # offset and windowDur input in seconds then converted to samples
        protocol = str(protocol)
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
        startSamples += offset
        spikes = self.units[str(unit)]['times'][protocol]
        
        if paramColors is None:
            paramColors = [None]*len(params)
        else:
            for i,c in enumerate(paramColors):
                if c=='auto' and i<len(params):
                    paramColors[i] = cm.Dark2(range(0,256,int(256/len(set(params[i])))))
                    break
        grid = True if grid and len(paramNames)==2 else False
         
        plt.figure(facecolor='w')
        if grid:
            axes = []
            rows = []
            gs = gridspec.GridSpec(len(set(params[1])),len(set(params[0])))
        else:
            axes = [plt.subplot(1,1,1)]
            rows = [0]
            gs = None
        if len(params)<1:
            self.appendToRaster(axes,spikes,startSamples,offset,windowDur,rows=rows)
        else:
            self.parseRaster(axes,spikes,startSamples,offset,windowDur,params,paramColors,rows=rows,grid=grid,gs=gs)
        for ax,r in zip(axes,rows):
            ax.set_xlim([offset/self.sampleRate,(max(windowDur)+offset)/self.sampleRate])
            ax.set_ylim([-0.5,r+0.5])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            if grid:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Trial')
        axes[-1].set_title('Unit '+str(unit))
         
         
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
    
    
    def runAllAnalyses(self, units=None, protocolsToRun = ['sparseNoise', 'gratings', 'gratings_ori', 'spots', 'checkerboard'], splitRunning = False, useCache=False):

        for pro in protocolsToRun:
            protocol = self.getProtocolIndex(pro)
            
            if 'gratings'==pro:
                if splitRunning:
                    trialStartFrames = self.visstimData[str(protocol)]['stimStartFrames'][:-1]
                    trialEndFrames = trialStartFrames + self.visstimData[str(protocol)]['stimTime']
                    trialStarts = self.visstimData[str(protocol)]['frameSamples'][trialStartFrames]
                    trialEnds = self.visstimData[str(protocol)]['frameSamples'][trialEndFrames]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf', trials=statTrials, saveTag='_stat')
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf', trials=runTrials, saveTag='_run')
                else:
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='stf')

            elif 'gratings_ori'==pro:
                if splitRunning:
                    trialStartFrames = self.visstimData[str(protocol)]['stimStartFrames'][:-1]
                    trialEndFrames = trialStartFrames + self.visstimData[str(protocol)]['stimTime']
                    trialStarts = self.visstimData[str(protocol)]['frameSamples'][trialStartFrames]
                    trialEnds = self.visstimData[str(protocol)]['frameSamples'][trialEndFrames]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori', trials=statTrials, saveTag='_stat')
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori', trials=runTrials, saveTag='_run')
                else:
                    self.analyzeGratings(units, protocol = protocol, useCache=useCache, protocolType='ori')

            elif 'sparseNoise' in pro:
                if splitRunning:
                    trialStartFrames = self.visstimData[str(protocol)]['stimStartFrames'][:-1]
                    trialEndFrames = trialStartFrames + self.visstimData[str(protocol)]['boxDuration']
                    trialStarts = self.visstimData[str(protocol)]['frameSamples'][trialStartFrames]
                    trialEnds = self.visstimData[str(protocol)]['frameSamples'][trialEndFrames]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.findRF(units, protocol=protocol, useCache=useCache, trials=statTrials, saveTag='_stat')
                    self.findRF(units, protocol=protocol, useCache=useCache, trials=runTrials, saveTag='_run')
                else:                    
                    self.findRF(units, protocol=protocol, useCache=useCache)
            elif 'spots' in pro:
                if splitRunning:
                    trialStartFrames = self.visstimData[str(protocol)]['trialStartFrame'][:-1]
                    trialDuration = (self.visstimData[str(protocol)]['trialNumFrames']).astype(np.int)
                    trialEndFrames = trialStartFrames + trialDuration[:-1]
                    frameSamples = self.visstimData[str(protocol)]['frameSamples']     
                    trialStarts = frameSamples[trialStartFrames]
                    trialEnds = frameSamples[trialEndFrames]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.analyzeSpots(units, protocol=protocol, useCache=useCache, trials=statTrials, saveTag='_stat')
                    self.analyzeSpots(units, protocol=protocol, useCache=useCache, trials=runTrials, saveTag='_run')
                else:
                    self.analyzeSpots(units, protocol=protocol, useCache=useCache)
            elif 'checkerboard' in pro:
                if splitRunning:
                    trialStartFrame = self.visstimData[str(protocol)]['trialStartFrame'][:-1]
                    trialDuration = (self.visstimData[str(protocol)]['trialNumFrames']).astype(int)
                    trialStarts = self.visstimData[str(protocol)]['frameSamples'][trialStartFrame]
                    trialEnds = self.visstimData[str(protocol)]['frameSamples'][trialStartFrame+trialDuration]
                    statTrials, runTrials, _ = self.parseRunning(protocol, trialStarts=trialStarts, trialEnds=trialEnds)
                    
                    self.analyzeCheckerboard(units, protocol=protocol, trials=statTrials, saveTag='_stat')
                    self.analyzeCheckerboard(units, protocol=protocol, trials=runTrials, saveTag='_run')
                else:
                    self.analyzeCheckerboard(units, protocol=protocol)
            else:
                print("Couldn't find analysis script for protocol type:", pro)
                
     
    def getProtocolIndex(self, label):
        protocol = []
        protocol.extend([i for i,f in enumerate(self.kwdFileList) if ntpath.dirname(str(f)).endswith(label)])
        if len(protocol)>1:
            raise ValueError('Multiple protocols found matching: '+label)
        elif len(protocol)<1:
            return None
        else:
            return protocol[0]
        
          
    def getOrderedUnits(self,units=None):
        # orderedUnits, yPosition = self.getOrderedUnits(units)
        if units is None:
            units = [u for u in self.units]
        elif isinstance(units,list):
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
    
    
    def loadClusteredData(self, kwdNsamplesList = None, protocolsToAnalyze = None, fileDir = None):
        from load_phy_template import load_phy_template
                 
        if fileDir is None:
            fileDir = fileIO.getDir()
        
        if protocolsToAnalyze is None:
            protocolsToAnalyze = np.arange(len(self.d))
        
        self.units = load_phy_template(fileDir, sampling_rate = self.sampleRate)
        for unit in self.units:
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


    def saveHDF5(self,filePath=None):
        fileIO.objToHDF5(self,filePath)
                    
                    
    def loadHDF5(self,filePath=None):
        fileIO.hdf5ToObj(self,filePath)
                    
    
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
        date,animalID = expName.split('_')
        return date,animalID


    def readExcelFile(self, sheetname = None, fileName = None):
        if sheetname is None:            
            expDate,_ = self.getExperimentInfo()
        if fileName is None:        
            fileName = fileIO.getFile()
            if fileName=='':
                return        
        
        table = pandas.read_excel(fileName, sheetname=expDate)
        for u in range(table.shape[0]):
            unit = table.Cell[u]
            label = table.Label[u]
            self.units[str(unit)]['label'] = label
            
        try:
            self.CCFTipPosition = np.array(table.Tip[0:3])
            self.CCFLPEntryPosition = np.array(table.Entry[0:3])
            self.findCCFCoords()
        except:
            for u in self.units:
                self.units[str(u)]['CCFCoords'] = np.full(3,np.nan)
            print('Could not fine CCF Tip or Entry positions')


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
        return units
        
        
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
        for i, unit in enumerate(units):
            distFromTip = unitsYPos[i] - tipProbePos
            pos = np.array([xSlope, ySlope, zSlope])*distFromTip
            pos+=tipPos
            self.units[str(unit)]['CCFCoords'] = pos
            
        
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
                    startTime.append(datetime.datetime.strptime(os.path.basename(itemPath)[0:19],'%Y-%m-%d_%H-%M-%S'))
                    kwdFiles.append(os.path.join(itemPath,f))
                    kwd = h5py.File(kwdFiles[-1],'r')
                    nSamples.append(kwd['recordings']['0']['data'].shape[0])
    return zip(*[n[1:] for n in sorted(zip(startTime,kwdFiles,nSamples),key=lambda z: z[0])])


def makeDat(kwdFiles=None):
    if kwdFiles is None:
        kwdFiles, _ = getKwdInfo()
    dirPath = os.path.dirname(os.path.dirname(kwdFiles[0]))
    datFilePath = os.path.join(dirPath,os.path.basename(dirPath)+'.dat')
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
    try:
        gridSize = max(x[-1]-x[0],y[-1]-y[0])
        lowerBounds = np.array([x[0]-maxOffGrid,y[0]-maxOffGrid,0,0,0,data.min(),data.min()])
        upperBounds = np.array([x[-1]+maxOffGrid,y[-1]+maxOffGrid,0.5*gridSize,0.5*gridSize,2*math.pi,1.5*data.max(),np.median(data)])
        fitParams,fitCov = scipy.optimize.curve_fit(gauss2D,(x,y),data.flatten(),p0=initialParams,bounds=(lowerBounds,upperBounds))
        if not all([lowerBounds[i]+1<fitParams[i]<upperBounds[i]-1 for i in (0,1)]):
            fitParams = None # if fit center on edge of boundaries
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
    

def stfLogGauss2D(stfTuple,sf0,tf0,sigSF,sigTF,speedTuningIndex,amplitude,offset):
    sf,tf = stfTuple
    tf = tf[:,None]
    z = offset + amplitude * np.exp(-((np.log2(sf)-np.log2(sf0))**2)/(2*sigSF**2)) * np.exp(-((np.log2(tf)-(speedTuningIndex*(np.log2(sf)-np.log2(sf0))+np.log2(tf0)))**2)/(2*sigTF**2))
    return z.ravel()


def fitStf(sf,tf,data,initialParams):
    try:
        lowerBounds = np.array([sf[0]-0.25*sf[0],tf[0]-0.25*tf[0],0,0,-0.25,0,data.min()])
        upperBounds = np.array([sf[-1]+0.25*sf[-1],tf[-1]+0.25*tf[-1],0.5*tf[-1],0.5*tf[-1],1.25,1.5*data.max(),np.median(data)])
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
    
if __name__=="__main__":
    pass       