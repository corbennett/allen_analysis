# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:19:20 2016

@author: SVC_CCG
"""

from __future__ import division
import datetime, h5py, itertools, json, math, os, shelve, shutil
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.ndimage.filters
from matplotlib import pyplot as plt 
import matplotlib.gridspec as gridspec
from PyQt4 import QtGui


class probeData():
    
    def __init__(self):
        self.dataDir = r'C:\Users\SVC_CCG\Desktop\Data'
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
        
        
    def loadExperiment(self):
        self.kwdFileList, nsamps = self.getKwdInfo()
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
        
        self.behaviorData[str(protocol)] = {}        
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
    
    
    def decodeWheel(self, kernelLength = 0.5, protocol=0):
        wheelData = -self.data[str(protocol)]['data'][:, self.wheelChannel] * self.gains[self.wheelChannel]
        wheelData = wheelData - np.min(wheelData)
        wheelData = 2*np.pi*wheelData/np.max(wheelData)
      
        smoothFactor = 500.0       

        wD = wheelData
        angularWheelData = np.arctan2(np.sin(wD), np.cos(wD))
        angularWheelData = np.convolve(angularWheelData, np.ones(smoothFactor), 'same')/smoothFactor
       
        angularDisplacement = (np.diff(angularWheelData) + np.pi)%(2*np.pi) - np.pi
        angularDisplacement[np.abs(angularDisplacement) > 0.001] = 0
        self.wheelData = np.convolve(angularDisplacement, np.ones(kernelLength*self.sampleRate), 'same')/(kernelLength*self.sampleRate)
        self.wheelData *= 7.6*self.sampleRate
        self.wheelData = np.insert(self.wheelData, 0, self.wheelData[0])


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
            for trial in xrange(trialStarts.size):
                trialSpeed = np.mean(self.wheelData[trialStarts[trial]:trialEnds[trial]])
                if trialSpeed >= runThresh:
                    self.runningTrials.append(trial)
                elif trialSpeed <= statThresh:
                    self.stationaryTrials.append(trial)
     
               
    def findSpikesPerTrial(self, trialStarts, trialEnds, spikes):
        
        spikesPerTrial = np.zeros(trialStarts.size)
        for trialNum in xrange(trialStarts.size):
            trialSamples = np.arange(trialStarts[trialNum], trialEnds[trialNum])    
            spikesPerTrial[trialNum] = np.intersect1d(trialSamples, spikes).size
        
        return spikesPerTrial
        
        
    def findRF(self, spikes, sigma = 2, plot = True, noiseStim = 'sparse', minLatency = 0.03, maxLatency = 0.13, trials = None, protocol=1):
        minLatency *= self.sampleRate
        maxLatency *= self.sampleRate
        if noiseStim == 'sparse':        
            xpos = np.sort(np.unique(self.visstimData[str(protocol)]['boxPositionHistory'][:, 0]))
            ypos = np.sort(np.unique(self.visstimData[str(protocol)]['boxPositionHistory'][:, 1]))
            
            posHistory = self.visstimData[str(protocol)]['boxPositionHistory'][:]
            colorHistory = self.visstimData[str(protocol)]['boxColorHistory'][:, 0:1]        
            
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
                    gridOnSpikes[index] += np.intersect1d(np.arange(p+minLatency, p+maxLatency), spikes).size
                
                for p in posOffSamples:
                    gridOffSpikes[index] += np.intersect1d(np.arange(p+minLatency, p+maxLatency), spikes).size
                
                gridOnSpikes[index] = gridOnSpikes[index]/float(posOnTrials.size)
                gridOffSpikes[index] = gridOffSpikes[index]/float(posOffTrials.size)
                
            gridOnSpikes = gridOnSpikes.reshape(xpos.size,ypos.size).T    
            gridOffSpikes = gridOffSpikes.reshape(xpos.size,ypos.size).T
                        
            gridExtent = self.visstimData[str(protocol)]['gridBoundaries']
            
            if plot:
                gs = gridspec.GridSpec(1, 2)
                gridOnSpikes_filter = scipy.ndimage.filters.gaussian_filter(gridOnSpikes, sigma)
                gridOffSpikes_filter = scipy.ndimage.filters.gaussian_filter(gridOffSpikes, sigma)

                maxVal = max(np.max(gridOnSpikes_filter), np.max(gridOffSpikes_filter))
                minVal = min(np.min(gridOnSpikes_filter), np.min(gridOffSpikes_filter))
                
                plt.figure()
                a1 = plt.subplot(gs[0, 0])
                a1.imshow(gridOnSpikes_filter, clim=[minVal, maxVal], interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]] )
                a1.set_title('on response')
           
                a2 = plt.subplot(gs[0, 1])
                im = a2.imshow(gridOffSpikes_filter, clim=[minVal, maxVal], interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]])
                a2.set_title('off response')
                plt.colorbar(im, ax=[a1, a2], fraction=0.05, pad=0.04)
            return gridOnSpikes, gridOffSpikes, xpos, ypos
            
        elif noiseStim == 'dense':
            a = np.random.RandomState(self.visstimData[str(protocol)]['numpyRandomSeed'])
            gridBoundaries = self.visstimData[str(protocol)]['gridBoundaries']
            boxSize = float(self.visstimData[str(protocol)]['boxSize'])
            gridXlength = np.ceil((gridBoundaries[2] - gridBoundaries[0])/boxSize)
            gridYlength = np.ceil((gridBoundaries[3] - gridBoundaries[1])/boxSize)
            rf = []
            
            for frame in self.visstimData[str(protocol)]['noiseStartFrames']:
                frameSamples = np.arange(self.visstimData[str(protocol)]['frameSamples'][frame]+minLatency, self.visstimData[str(protocol)]['frameSamples'][frame] + maxLatency)
                spikesThisFrame = np.intersect1d(frameSamples, spikes).size
                noiseImage = a.randint(0,2, [gridYlength, gridXlength])*2 - 1
                noiseImage = np.repeat(noiseImage, boxSize, 0)
                noiseImage = np.repeat(noiseImage, boxSize, 1)
                rf.append(spikesThisFrame*noiseImage)
                  
            rf_array = np.array(rf)   
            self.rf_mean = np.sum(rf_array, axis=0)
            plt.figure()
            plt.imshow(scipy.ndimage.filters.gaussian_filter(self.rf_mean, sigma), origin='lower')
            plt.colorbar()
      
    
    def findRF_units(self, units = None, plot=True, sigma=1, protocol=1, fit=False):
        sortedUnits = [(u[0], u[1]['ypos'], u[1]['times'][str(protocol)]) for u in self.units.iteritems()]
        sortedUnits.sort(key=lambda i: -i[1])
        if units is not None:
            sU = [sortedUnits[ind] for ind, u in enumerate(sortedUnits) if int(u[0]) in units]
        else:
            sU = sortedUnits
        
        if plot:        
            plt.figure(figsize = (3.0, 50.0), tight_layout = True)
            gridExtent = self.visstimData[str(protocol)]['gridBoundaries']
            gs = gridspec.GridSpec(len(sU), 4)
        
        for index in xrange(len(sU)):
            if 'rf_on' in self.units[sU[index][0]].keys():
                gon = self.units[sU[index][0]]['rf']['on']
                goff = self.units[sU[index][0]]['rf']['off']
                xpos = self.units[sU[index][0]]['rf']['xpos']
                ypos = self.units[sU[index][0]]['rf']['ypos']
            else:
                gon, goff, xpos, ypos = self.findRF(sU[index][2], sigma=sigma, minLatency = 0.05, maxLatency = 0.15, plot=False, protocol=protocol)
                self.units[sU[index][0]]['rf'] = {'gon':gon,'goff':goff,'xpos':xpos,'ypos':ypos}
                
            gridOnSpikes_filter = scipy.ndimage.filters.gaussian_filter(gon, sigma)
            gridOffSpikes_filter = scipy.ndimage.filters.gaussian_filter(goff, sigma)

            maxVal = max((np.max(gridOnSpikes_filter), np.max(gridOffSpikes_filter)))
            minVal = min((np.min(gridOnSpikes_filter), np.min(gridOffSpikes_filter)))
            
            if fit:
                fitParams = []
                pixPerDeg = self.visstimData[str(protocol)]['pixelsPerDeg']
                for data in (gridOnSpikes_filter,gridOffSpikes_filter):
                    # params: x0 , y0, sigX, sigY, theta, amplitude
                    elev, azi = ypos/pixPerDeg, xpos/pixPerDeg
                    j,i = np.unravel_index(np.argmax(data),data.shape)
                    initialParams = (azi[j], elev[i], azi[1]-azi[0], elev[1]-elev[0], 0, data.max())
                    fitParams.append(fitGauss2D(azi,elev,data,initialParams))
                onFit,offFit = fitParams
                            
            if plot:        
                a1 = plt.subplot(gs[index, :2])
                a1.imshow(gridOnSpikes_filter, clim=(minVal, maxVal), interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]])
                if fit and onFit is not None:
                    xlim = a1.get_xlim
                    ylim = a1.get_ylim
                    a1.plot(onFit[0],onFit[1],'kx',markeredgewidth=2)
                    fitX,fitY = getEllipseXY(*onFit[:-1])
                    a1.plot(fitX,fitY,'k',linewidth=2)
                    a1.set_xlim(xlim)
                    a1.set_ylim(ylim)
                a1.xaxis.set_visible(False)
                a1.yaxis.set_visible(False)
                a1.text(-5.5, 0.5, str(sU[index][0]))
                
                a2 = plt.subplot(gs[index, 2:])
                a2.imshow(gridOffSpikes_filter, clim=(minVal, maxVal), interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]])
                if fit and offFit is not None:
                    xlim = a2.get_xlim
                    ylim = a2.get_ylim
                    a2.plot(offFit[0],offFit[1],'kx',markeredgewidth=2)
                    fitX,fitY = getEllipseXY(*offFit[:-1])
                    a2.plot(fitX,fitY,'k',linewidth=2)
                    a2.set_xlim(xlim)
                    a2.set_ylim(ylim)
                a2.xaxis.set_visible(False)
                a2.yaxis.set_visible(False)
    
    
    def analyzeGratings(self, spikes, trials = None, responseLatency = 0.05, plot=True, protocol=3):
        trialSF = self.visstimData[str(protocol)]['stimulusHistory_sf']
        trialTF = self.visstimData[str(protocol)]['stimulusHistory_tf']
        trialOri = self.visstimData[str(protocol)]['stimulusHistory_ori']
        trialContrast = self.visstimData[str(protocol)]['stimulusHistory_contrast']

        sf = np.unique(trialSF)
        tf = np.unique(trialTF)
        ori = np.unique(trialOri)
        
        #spontaneous firing rate taken from interspersed gray trials
        spontRate = 0
        spontCount = 0
        
        stfMat = np.zeros([tf.size, sf.size])
        stfCountMat = np.zeros([sf.size, tf.size])
        oriVect = np.zeros(ori.size)
        oriCountVect = np.zeros(ori.size)
        
        responseLatency *= self.sampleRate
        
        #find and record spikes for every trial
        self.trialResponse = np.zeros(trialSF.size)
        for trial in xrange(trialSF.size-1):
            trialStartFrame = self.visstimData[str(protocol)]['stimStartFrames'][trial]
            trialEndFrame = trialStartFrame + self.visstimData[str(protocol)]['stimTime']
            trialSamples = np.arange(self.visstimData[str(protocol)]['frameSamples'][trialStartFrame] + responseLatency, self.visstimData[str(protocol)]['frameSamples'][trialEndFrame] + responseLatency)    
            
            spikesThisTrial = np.intersect1d(spikes, trialSamples).size
            self.trialResponse[trial] = self.sampleRate*spikesThisTrial/(float(trialSamples.size))
        
        
        #make STF mat for specified trials (default all trials)
        if trials is None:
            trials = np.arange(trialSF.size - 1)
        
        for trial in trials:
            spikeRateThisTrial = self.trialResponse[trial]
            
            if trialContrast[trial] > 0:
                sfIndex = int(np.where(sf == trialSF[trial])[0])
                tfIndex = int(np.where(tf == trialTF[trial])[0])
                oriIndex = int(np.where(ori == trialOri[trial])[0])
                    
                stfMat[tfIndex, sfIndex] += spikeRateThisTrial
                stfCountMat[tfIndex, sfIndex] += 1
                
                oriVect[oriIndex] += spikeRateThisTrial
                oriCountVect[oriIndex] += 1        
            else:
                spontRate += spikeRateThisTrial
                spontCount += 1
        
        spontRate /= spontCount
        stfMat /= stfCountMat
        stfMat -= spontRate
        
        
        if plot:
            xyNan = np.transpose(np.where(np.isnan(stfMat)))
            stfMat[np.isnan(stfMat)] = 0
           
            gs = gridspec.GridSpec(2, 3)
            plt.figure()
            a1 = plt.subplot(gs[:, :-1])
            plt.xlabel('sf')
            plt.ylabel('tf')
            im = a1.imshow(stfMat, clim=(0,stfMat.max()), cmap='gray', origin = 'lower', interpolation='none')
            for xypair in xyNan:    
                a1.text(xypair[1], xypair[0], 'no trials', color='white', ha='center')
            a1.set_xticklabels(np.insert(sf, 0, 0))
            a1.set_yticklabels(np.insert(tf, 0, 0))
            plt.colorbar(im, ax=a1, fraction=0.05, pad=0.04)
            
            a2 = plt.subplot(gs[0,2])
            values = np.mean(stfMat, axis=0)
            error = np.std(stfMat, axis=0)
            a2.plot(sf, values)
            plt.fill_between(sf, values+error, values-error, alpha=0.3)
            plt.xlabel('sf')
            plt.ylabel('spikes')
            plt.xticks(sf)
            
            a3 = plt.subplot(gs[1, 2])
            values = np.mean(stfMat, axis=1)
            error = np.std(stfMat, axis=1)
            a3.plot(tf, values)
            plt.fill_between(tf, values+error, values-error, alpha=0.3)
            plt.xlabel('tf')
            plt.ylabel('spikes')
            plt.xticks(tf)
            
            plt.tight_layout()
        
        return stfMat, sf, tf
    
    
    def analyzeGratings_units(self, units = None, protocol=2, trials=None, plot=True, fit=False):
        sortedUnits = [(u[0], u[1]['ypos'], u[1]['times'][str(protocol)]) for u in self.units.iteritems()]
        sortedUnits.sort(key=lambda i: -i[1])    
        
        if units is not None:
            sU = [sortedUnits[ind] for ind, u in enumerate(sortedUnits) if int(u[0]) in units]
        else:
            sU = sortedUnits        

        if plot:
            gs = gridspec.GridSpec(len(sU), 1)
            plt.figure(figsize = (1.725, 17.9), tight_layout = True)
            
        for index in xrange(len(sU)):
            if 'stf_dict' in self.units[sU[index][0]].keys():
                stfMat = self.units[sU[index][0]]['stf']['stfMat']
                sf = self.units[sU[index][0]]['stf']['sf']
                tf = self.units[sU[index][0]]['stf']['tf']
            else:
                stfMat, sf, tf = self.analyzeGratings(sU[index][2], protocol=protocol, plot=False, trials=trials)
                self.units[sU[index][0]]['stf'] = {'stfMat':stfMat,'sf':sf,'tf':tf}
                
            if fit:
                # params: sf0 , tf0, sigSF, sigTF, speedTuningIndex, amplitude
                j,i = np.unravel_index(np.argmax(stfMat),stfMat.shape)
                initialParams = (sf[j], tf[i], 1, 1, 0.5, stfMat.max())
                fitParams = fitStfLogGauss2D(sf,tf,stfMat,initialParams)

            if plot:
                a1 = plt.subplot(gs[index, 0])
        
                xyNan = np.transpose(np.where(np.isnan(stfMat)))
                stfMat[np.isnan(stfMat)] = 0
                cLim = max(1,np.max(abs(stfMat)))
                plt.imshow(stfMat, clim=(-cLim,cLim), cmap='bwr', origin = 'lower', interpolation='none')
                for xypair in xyNan:    
                    a1.text(xypair[1], xypair[0], 'no trials', color='white', ha='center')
                if fit and fitParams is not None:
                    xlim = a1.get_xlim()
                    ylim = a1.get_ylim()
                    a1.plot(np.log2(fitParams[0])-np.log2(sf[0]),np.log2(fitParams[1])-np.log2(tf[0]),'kx',markeredgewidth=2)
                    fitX,fitY = getStfContour(sf,tf,fitParams)
                    a1.plot(fitX,fitY,'k',linewidth=2)
                    a1.set_xlim(xlim)
                    a1.set_ylim(ylim)
                a1.xaxis.set_visible(False)
                a1.yaxis.set_visible(False)
                a1.text(-3.5, 0.5, str(sU[index][0]))
        
                cb = plt.colorbar()
                cb.set_ticks([stfMat.min(), stfMat.max()])
                
    
    def analyzeSpots(self, spikes, protocol = 3, plot=True, trials=None):
        if trials is None:
            trials = np.arange((self.visstimData[str(protocol)]['trialStartFrame'][:-1]).size)
        
        trialStartFrames = self.visstimData[str(protocol)]['trialStartFrame'][trials]
        trialDuration = (self.visstimData[str(protocol)]['trialNumFrames'][trials]).astype(np.int)
        trialEndFrames = trialStartFrames + trialDuration
        frameSamples = self.visstimData[str(protocol)]['frameSamples']     
        trialStartSamples = frameSamples[trialStartFrames]
        trialEndSamples = frameSamples[trialEndFrames]
        
        spikesPerTrial = self.findSpikesPerTrial(trialStartSamples, trialEndSamples, spikes)
        trialSpikeRate = spikesPerTrial/((1/self.visstimData[str(protocol)]['frameRate'])*trialDuration)

        
        trialPos = self.visstimData[str(protocol)]['trialSpotPos'][trials]
        trialColor = self.visstimData[str(protocol)]['trialSpotColor'][trials]
        trialSize = self.visstimData[str(protocol)]['trialSpotSize'][trials]
        trialDir = self.visstimData[str(protocol)]['trialSpotDir'][trials]
        trialSpeed = self.visstimData[str(protocol)]['trialSpotSpeed'][trials]
        
        for d in np.unique(trialDir):
            pass 
            
        horTrials = np.in1d(trialDir,[0,180])
        vertTrials = np.in1d(trialDir,[90,270])
#        horTrials = np.logical_or(trialDir==0, trialDir==180)
#        vertTrials = np.logical_or(trialDir==270, trialDir==90)
        
        azimuths = np.unique(trialPos[vertTrials])
        elevs = np.unique(trialPos[horTrials])
        
        # get RF         
        azimuthSpikeRate = np.zeros(azimuths.size)        
        elevSpikeRate = np.zeros(elevs.size)
        azimuthTrialCount = np.zeros(azimuths.size)        
        elevTrialCount = np.zeros(elevs.size)
        for trial in xrange(trialPos.size):
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
            
#        xv, yv = np.meshgrid(elevSpikeRate, azimuthSpikeRate) # should this be (azi,elev)?
#        spotRF = (xv+yv)/2.0-itiRate.mean()
            
#        elevSpikeRate -= itiRate.mean()
#        azimuthSpikeRate -= itiRate.mean()
#        minSpikeRate = min(elevSpikeRate.min(),azimuthSpikeRate.min())
#        spotRF = (elevSpikeRate[:,None]-minSpikeRate)*(azimuthSpikeRate-minSpikeRate)
            
        x,y = np.meshgrid(azimuthSpikeRate-meanItiRate,elevSpikeRate-meanItiRate)
        spotRF = np.sqrt(abs(x*y))*np.sign(x+y)
        responseDict['spotRF'] = spotRF
        
        spotRF_zscore = (spotRF-spotRF.mean())/spotRF.std()
        if plot:
            
            gs = gridspec.GridSpec(1, 4)
            plt.figure(figsize = (13.075, 2.7625), tight_layout = True)
            
            a1 = plt.subplot(gs[0, 0])            
#            climMax = max(2, np.max(spotRF_zscore))
#            climMin = min(-2, np.min(spotRF_zscore))
            cLim = max(2,np.max(abs(spotRF)))
            im = a1.imshow(spotRF, clim = (-cLim,cLim), cmap='bwr', interpolation='none', origin='lower')
            plt.colorbar(im, ax=a1, fraction=0.05, pad=0.04)
            
            for paramnum, param in enumerate(['trialSpotSize', 'trialSpotDir', 'trialSpotSpeed']):
                    a = plt.subplot(gs[0, paramnum+1])
                    values = responseDict[param]['tuningCurve']['mean_spontSubtracted'] 
                    error = responseDict[param]['tuningCurve']['sem'] 
                    a.plot(responseDict[param]['tuningCurve']['paramValues'], values)
                    plt.fill_between(responseDict[param]['tuningCurve']['paramValues'], values+error, values-error, alpha=0.3)
                    a.plot(responseDict[param]['tuningCurve']['paramValues'], np.zeros(values.size), 'r--')
                    plt.xlabel(param) 
                    plt.ylim(min(-0.1, np.min(values - error)), max(np.max(values + error), 0.1))
                    plt.locator_params(axis = 'y', nbins = 3)
                    a.set_xticks(responseDict[param]['tuningCurve']['paramValues'])
            
        return responseDict
    
        
    def analyzeSpots_units(self, units = None, protocol=3, trials=None, plot=True):
        sortedUnits = [(u[0], u[1]['ypos'], u[1]['times'][str(protocol)]) for u in self.units.iteritems()]
        sortedUnits.sort(key=lambda i: -i[1])    
        
        if units is not None:
            sU = [sortedUnits[ind] for ind, u in enumerate(sortedUnits) if int(u[0]) in units]
        else:
            sU = sortedUnits
        
        if plot:
            gs = gridspec.GridSpec(len(sU), 4)
            plt.figure(figsize = (6.0, 17.9), tight_layout = True)
            
        for index in xrange(len(sU)):
            if 'spotRF_responseDict' in self.units[sU[index][0]].keys():
                responseDict = self.units[sU[index][0]]['spot_responseDict']
                spotRF = responseDict['spotRF']
            else:
                responseDict = self.analyzeSpots(sU[index][2], protocol=protocol, plot=False, trials=trials)
                self.units[sU[index][0]]['spot'] = responseDict
                spotRF = responseDict['spotRF']
                
            if plot:
                a1 = plt.subplot(gs[index, 0])
        
                xyNan = np.transpose(np.where(np.isnan(spotRF)))
                spotRF[np.isnan(spotRF)] = 0
                
#                climMax = max(2, np.max(spotRF))
#                climMin = min(-2, np.min(spotRF))
                cLim = max(2,np.max(abs(spotRF)))
                plt.imshow(spotRF, clim=(-cLim,cLim), cmap='bwr', origin = 'lower', interpolation='none')
                for xypair in xyNan:    
                    a1.text(xypair[1], xypair[0], 'no trials', color='white', ha='center')

                a1.xaxis.set_visible(False)
                a1.yaxis.set_visible(False)
                a1.text(-5.5, 0.5, str(sU[index][0]))
        
                cb = plt.colorbar()
                cb.set_ticks([spotRF.min(), spotRF.max()])
                
                for paramnum, param in enumerate(['trialSpotSize', 'trialSpotDir', 'trialSpotSpeed']):
                    a = plt.subplot(gs[index, paramnum+1])
                    values = responseDict[param]['tuningCurve']['mean_spontSubtracted'] 
                    error = responseDict[param]['tuningCurve']['sem'] 
                    a.plot(responseDict[param]['tuningCurve']['paramValues'], values)
                    plt.fill_between(responseDict[param]['tuningCurve']['paramValues'], values+error, values-error, alpha=0.3)
                    a.plot(responseDict[param]['tuningCurve']['paramValues'], np.zeros(values.size), 'r--')
                    plt.xlabel(param) 
                    plt.ylim(min(-0.1, np.min(values - error)), max(np.max(values + error), 0.1))
                    plt.locator_params(axis = 'y', nbins = 3)
                    if index < len(sU) -1:
                        a.xaxis.set_visible(False)
                    else:
                        a.set_xticks(responseDict[param]['tuningCurve']['paramValues'])
                     
#                a4 = plt.subplot(gs[index,3], projection='polar')
#                theta = (np.pi/180.)*(responseDict['trialSpotDir']['tuningCurve'][0]).astype(float)
#                rho = responseDict['trialSpotDir']['tuningCurve'][3]
#                theta = np.append(theta, theta[0])
#                rho = np.append(rho, rho[0])
#                a4.plot(theta, rho)
#                a4.xaxis.set_visible(False)
                     
                                        
    def analyzeCheckerboard(self, units, protocol=None, trials=None, plot=False):
        if not isinstance(units,list):
            units = [units]
        for u in units[:]:
            if str(u) not in self.units.keys():
                units.remove(u)
                print(str(u)+' not in units.keys()')
        if len(units)<1:
            return
        
        if protocol is None:
            protocol = self.getProtocolIndex('checkerboard')            
        p = self.visstimData[str(protocol)]
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
            plt.figure(facecolor='w')
            gs = gridspec.GridSpec(2*len(units),4)
            row = 0
        bckgndSpeed = np.concatenate((-p['bckgndSpeed'][:0:-1],p['bckgndSpeed']))
        patchSpeed = np.concatenate((-p['patchSpeed'][:0:-1],p['patchSpeed']))
        resp = np.full((bckgndSpeed.size,patchSpeed.size,p['patchSize'].size,p['patchElevation'].size),np.nan)
        resp = np.tile(resp[:,:,:,:,None],math.ceil(trials.size/(resp.size-2*p['patchSpeed'].size*p['patchSize'].size))+3)
        for u in units:
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
        if plot:
            plt.tight_layout()
    
    
    def getProtocolIndex(self, label):
        protocol = []
        protocol.extend([i for i,f in enumerate(self.kwdFileList) if os.path.dirname(f).endswith(label)])
        if len(protocol)<1:
            raise ValueError('No protocols found matching: '+label)
        elif len(protocol)>1:
            raise ValueError('Multiple protocols found matching: '+label)
        return protocol[0]
    
    
    def runAllAnalyses_units(self, units=None, protocolsToRun = ['sparseNoise', 'gratings', 'spots', 'checkerboard']):
        if units is None:
            units = self.units.keys()
        if type(units) is int:
            units = [units]
        
        for pro in protocolsToRun:
#            protocol = [index for index, f in enumerate(self.kwdFileList) if pro in f]
#            if len(protocol) == 0:
#                print "No protocols found matching:", pro
#                continue
#            if len(protocol) > 1:
#                print "Multiple protocols found matching:", pro, "Analyzing first only (", self.kwdFileList[protocol[0]], ")"
#    
#            protocol = protocol[0]  
            protocol = self.getProtocolIndex(pro)
            
            if len(units) > 1:
                if 'gratings' in pro:
                    self.analyzeGratings_units(units, protocol = protocol)
                elif 'sparseNoise' in pro:
                    self.findRF_units(units, protocol=protocol, fit=False)
                elif 'spots' in pro:
                    self.analyzeSpots_units(units, protocol=protocol)
                else:
                    print("Couldn't find analysis script for protocol type:", pro)
            else:
                spikes = self.units[str(units[0])]['times'][str(protocol)]
                if 'gratings' in pro:
                    responseDict = self.analyzeGratings(spikes, protocol = protocol)
                    self.units[str(units[0])]['stf_dict'] = responseDict
                elif 'sparseNoise' in pro:
                    g, gon, goff = self.findRF(spikes, protocol=protocol)
                    self.units[str(units[0])]['rf_on'] = gon
                    self.units[str(units[0])]['rf_off'] = goff
                elif 'spots' in pro:
                    responseDict = self.analyzeSpots(spikes, protocol=protocol)
                    self.units[str(units[0])]['spot_responseDict'] = responseDict
                elif 'checkerboard' in pro:
                    self.analyzeCheckerboard(units, protocol=protocol, plot=True)
                else:
                    print("Couldn't find analysis script for protocol type:", pro)            
                
        
    def saveWorkspace(self, variables=None, saveGlobals = False, filename=None, exceptVars = []):
        if filename is None:
            ftemp = self.filePath[:self.filePath.rfind('/')]
            ftemp = ftemp[ftemp.rfind('/')+1:]
            filename = self.filePath[:self.filePath.rfind('/')+1] + ftemp + '.out' 
        shelf = shelve.open(filename, 'n')
        
        if variables is None:
            if not saveGlobals:
                variables = self.__dict__.keys()
            else:
                variables = self.__dict__.keys() + globals().keys()
        
        for key in variables:
#            if key in exceptVars:
#                continue
            try:
                if key in self.__dict__.keys():
                    shelf[key] = self.__dict__[key]
                else:
                    shelf[key] = globals()[key]    
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
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

            
    def getKwdInfo(self, fileDir=None):
        fileList, nsamps = getKwdFiles(fileDir)
        return fileList, nsamps
    
    
    def getSingleUnits(self, fileDir = None, protocolsToAnalyze = None):
        if fileDir is None:
            fileDir = getDir()
        fileList, nsamps = self.getKwdInfo(fileDir=fileDir)
        if protocolsToAnalyze is None:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir)
        else:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir, protocolsToAnalyze=protocolsToAnalyze)


# utility functions

def getFile():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return QtGui.QFileDialog.getOpenFileName(None,'Choose File')
    
    
def getDir():
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return QtGui.QFileDialog.getExistingDirectory(None,'Choose Directory') 


def getKwdFiles(dirPath=None):
    # kwdFiles,nSamples = getKwdFiles()
    # returns kwd file paths and number of samples in each file ordered by file start time
    if dirPath is None:
        dirPath = getDir()
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
    for filePath in kwdFiles:
        kwd = h5py.File(filePath,'r')
        dset = kwd['recordings']['0']['data']
        i = 0
        while i<dset.shape[0]:
            (dset[i:i+dset.chunks[0],:128]).tofile(datFile)                        
            i += dset.chunks[0]
    datFile.close()
    shutil.copy(datFilePath,r'\\10.128.38.3\data_local_1\corbett')
    
    
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
        fitParams,fitCov = scipy.optimize.curve_fit(gauss2D,(x,y),data.flatten(),p0=initialParams)
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
        fitParams,fitCov = scipy.optimize.curve_fit(stfLogGauss2D,(sf,tf),data.flatten(),p0=initialParams)
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


   
if __name__=="__main__":
    pass       