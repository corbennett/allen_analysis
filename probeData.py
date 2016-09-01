# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:19:20 2016

@author: SVC_CCG
"""

#newFile = open(p.filePath + '.dat', 'wb')
#np.array(p.data['data'][:, :128]).tofile(newFile)

from __future__ import division
import h5py, os, scipy.signal, json
import Tkinter as tk
import tkFileDialog
import numpy as np
import itertools, math
from matplotlib import pyplot as plt 
import scipy.ndimage.filters
import shelve
import matplotlib.gridspec as gridspec

class probeData():
    
    def __init__(self):
         
        self.dataDir = 'C:\Users\SVC_CCG\Desktop\Data'
        self.recording = 0
        self.TTLChannelLabels = ['VisstimOn', 'CamExposing', 'CamSaving', 'OrangeLaserShutter']
        self.channelMapFile = 'C:\Users\SVC_CCG\Documents\PythonScripts\imec_channel_map.prb'
        self.wheelChannel = 134
        self.diodeChannel = 135
        self.sampleRate = 30000
         
    def getFile(self):
        root = tk.Tk()
        root.withdraw()
        return tkFileDialog.askopenfilename(initialdir = self.dataDir, parent = root)
    
    def getDir(self):
        root = tk.Tk()
        root.withdraw()
        return tkFileDialog.askdirectory(initialdir = self.dataDir, parent = root)
             
   
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
                print 'No vis stim data found for ' + os.path.basename(proPath)
            if not eyeDataFound:
                print 'No eye tracking data found for ' + os.path.basename(proPath)
            
        
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
            filePath = self.getFile()
        
        dataFile = h5py.File(filePath)        
        self.visstimData[str(protocol)] = {}
        for params in dataFile.keys():
            if dataFile[params].size > 1:
                self.visstimData[str(protocol)][params] = dataFile[params][:]
            else:
                self.visstimData[str(protocol)][params] = dataFile[params][()]   
        
        self.visStimDataFile = filePath
     
    def getEyeTrackData(self, filePath=None, protocol=0):
        if filePath is None:        
            filePath = self.getFile()
        
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
        

    def findRF(self, spikes, sigma = 2, plot = True, noiseStim = 'sparse', minLatency = 0.03, maxLatency = 0.15, trials = None, protocol=1):

        minLatency *= self.sampleRate
        maxLatency *= self.sampleRate
        if noiseStim == 'sparse':        
            xpos = np.sort(np.unique(self.visstimData[str(protocol)]['boxPositionHistory'][:, 0]))
            ypos = np.sort(np.unique(self.visstimData[str(protocol)]['boxPositionHistory'][:, 1]))
            
            posHistory = self.visstimData[str(protocol)]['boxPositionHistory'][:]
            colorHistory = self.visstimData[str(protocol)]['boxColorHistory'][:, 0:1]        
            
            grid = list(itertools.product(xpos, ypos))
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
                
            gridOnSpikes = np.reshape(gridOnSpikes, [xpos.size, ypos.size])    
            gridOffSpikes = np.reshape(gridOffSpikes, [xpos.size, ypos.size])
                        
            gridExtent = self.visstimData[str(protocol)]['gridBoundaries']
            
            if plot:  
                gridOnSpikes_filter = scipy.ndimage.filters.gaussian_filter(gridOnSpikes, sigma)
                gridOffSpikes_filter = scipy.ndimage.filters.gaussian_filter(gridOffSpikes, sigma)
                
                plt.figure()
                plt.imshow(gridOnSpikes_filter.T, interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]] )
                plt.colorbar()
                
                plt.figure()
                plt.imshow(gridOffSpikes_filter.T, interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]])
                plt.colorbar()
            return grid, gridOnSpikes, gridOffSpikes
            
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
      
    
    def findRF_units(self, units = None, plot=True, sigma=1, protocol=1):
        sortedUnits = [(u[0], u[1]['ypos'], u[1]['times'][str(protocol)]) for u in self.units.iteritems()]
        sortedUnits.sort(key=lambda i: -i[1])
        if units is not None:
            sU = [sortedUnits[u] for u in units]
        else:
            sU = sortedUnits
        
        if plot:        
            plt.figure(figsize = (1.725, 17.9), tight_layout = True)
            gridExtent = self.visstimData[str(protocol)]['gridBoundaries']
            gs = gridspec.GridSpec(len(sU), 4)
        
        for index in xrange(len(sU)):
            if 'rf_on' in self.units[sU[index][0]].keys():
                gon = self.units[sU[index][0]]['rf_on']
                goff = self.units[sU[index][0]]['rf_off']
            else:
                g, gon, goff = self.findRF(sU[index][2], sigma=sigma, minLatency = 0.05, maxLatency = 0.15, plot=False, protocol=protocol)
                self.units[sU[index][0]]['rf_on'] = gon
                self.units[sU[index][0]]['rf_off'] = goff
                
            gridOnSpikes_filter = scipy.ndimage.filters.gaussian_filter(gon, sigma)
            gridOffSpikes_filter = scipy.ndimage.filters.gaussian_filter(goff, sigma)
            
            maxVal = max((np.max(gridOnSpikes_filter), np.max(gridOffSpikes_filter)))
            minVal = min((np.min(gridOnSpikes_filter), np.min(gridOffSpikes_filter)))
                            
            if plot:        
                a1 = plt.subplot(gs[index, :2])
                a2 = plt.subplot(gs[index, 2:])
            
                a1.imshow(gridOnSpikes_filter.T, clim=(minVal, maxVal), interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]] )
                a1.xaxis.set_visible(False)
                a1.yaxis.set_visible(False)
                a1.text(1.1, 0.5, str(sU[index][0]))
                
                a2.imshow(gridOffSpikes_filter.T, clim=(minVal, maxVal), interpolation = 'none', origin = 'lower', extent = [gridExtent[0], gridExtent[2], gridExtent[1], gridExtent[3]])
                a2.xaxis.set_visible(False)
                a2.yaxis.set_visible(False)
                
            
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
        
    def analyzeGratings(self, spikes, trials = None, responseLatency = 0.05, plot=True, protocol=3):
        trialSF = self.visstimData[str(protocol)]['stimulusHistory_sf']
        trialTF = self.visstimData[str(protocol)]['stimulusHistory_tf']
        trialOri = self.visstimData[str(protocol)]['stimulusHistory_ori']
        trialContrast = self.visstimData[str(protocol)]['stimulusHistory_contrast']

        sfs = np.unique(trialSF)
        tfs = np.unique(trialTF)
        oris = np.unique(trialOri)
        
        #spontaneous firing rate taken from interspersed gray trials
        spontRate = 0
        spontCount = 0
        
        stfMat = np.zeros([sfs.size, tfs.size])
        stfCountMat = np.zeros([sfs.size, tfs.size])
        oriVect = np.zeros(oris.size)
        oriCountVect = np.zeros(oris.size)
        
        responseLatency *= self.sampleRate
        
        #find and record spikes for every trial
        self.trialResponse = np.zeros(trialSF.size)
        for trial in xrange(trialSF.size-1):
            trialStartFrame = self.visstimData[str(protocol)]['stimStartFrames'][trial]
            trialEndFrame = trialStartFrame + self.visstimData[str(protocol)]['stimTime']
            trialSamples = np.arange(self.visstimData[str(protocol)]['frameSamples'][trialStartFrame] + responseLatency, self.visstimData[str(protocol)]['frameSamples'][trialEndFrame] + responseLatency)    
            
            spikesThisTrial = np.intersect1d(spikes, trialSamples).size
            self.trialResponse[trial] = spikesThisTrial
        
        
        #make STF mat for specified trials (defaul all trials)
        if trials is None:
            trials = np.arange(trialSF.size - 1)
        
        for trial in trials:
            spikesThisTrial = self.trialResponse[trial]
            
            if trialContrast[trial] > 0:
                sfIndex = int(np.where(sfs == trialSF[trial])[0])
                tfIndex = int(np.where(tfs == trialTF[trial])[0])
                oriIndex = int(np.where(oris == trialOri[trial])[0])
                    
                stfMat[sfIndex, tfIndex] += spikesThisTrial
                stfCountMat[sfIndex, tfIndex] += 1
                
                oriVect[oriIndex] += spikesThisTrial
                oriCountVect[oriIndex] += 1        
            else:
                spontRate += spikesThisTrial
                spontCount += 1
        
        spontRate /= spontCount
        stfMat /= stfCountMat
        stfMat -= spontRate
        
        
        if plot:
            xyNan = np.transpose(np.where(np.isnan(stfMat)))
            stfMat[np.isnan(stfMat)] = 0
           
            gs = gridspec.GridSpec(2, 3)
            plt.figure(figsize = (17, 8), tight_layout = True)
            a1 = plt.subplot(gs[:, :-1])
            plt.xlabel('tf')
            plt.ylabel('sf')
            plt.imshow(stfMat, clim=(0,stfMat.max()), cmap='gray', origin = 'lower', interpolation='none')
            for xypair in xyNan:    
                a1.text(xypair[1], xypair[0], 'no trials', color='white', ha='center')
            a1.set_xticklabels(np.insert(tfs, 0, 0))
            a1.set_yticklabels(np.insert(sfs, 0, 0))
            plt.colorbar()
            
            a2 = plt.subplot(gs[0,2])
            plt.xlabel('sf')
            plt.ylabel('spikes')
            a2.plot(sfs, np.mean(stfMat, axis = 1))
            plt.xticks(sfs)
            
            a3 = plt.subplot(gs[1, 2])
            plt.xlabel('tf')
            plt.ylabel('spikes')
            a3.plot(tfs, np.mean(stfMat, axis = 0))
            plt.xticks(tfs)
    
        return sfs, tfs, stfMat
        
    def analyzeGratings_units(self, units = None, protocol=2, trials=None, plot=True):
        sortedUnits = [(u[0], u[1]['ypos'], u[1]['times'][str(protocol)]) for u in self.units.iteritems()]
        sortedUnits.sort(key=lambda i: -i[1])    
        
        if units is not None:
            sU = [sortedUnits[u] for u in units]
        else:
            sU = sortedUnits        

        if plot:
            gs = gridspec.GridSpec(len(sU), 1)
            plt.figure(figsize = (1.725, 17.9), tight_layout = True)
            
        for index in xrange(len(sU)):
            if 'stf' in self.units[sU[index][0]].keys():
                stfMat = self.units[sU[index][0]]['stf']
            else:
                sfs, tfs, stfMat = self.analyzeGratings(sU[index][2], protocol=protocol, plot=False, trials=trials)
                self.units[sU[index][0]]['stf'] = stfMat
            
            if plot:
                a1 = plt.subplot(gs[index, 0])
        
                xyNan = np.transpose(np.where(np.isnan(stfMat)))
                stfMat[np.isnan(stfMat)] = 0

                plt.imshow(stfMat, clim=(stfMat.min(),stfMat.max()), cmap='gray', origin = 'lower', interpolation='none')
                for xypair in xyNan:    
                    a1.text(xypair[1], xypair[0], 'no trials', color='white', ha='center')

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
        trialSpikeRate = spikesPerTrial/((1/60.0)*trialDuration)
        
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
            responseDict[param]['tuningCurve'] = []
            responseDict[param]['tuningCurve'].append(possibleValues)
            responseDict[param]['tuningCurve'].append(meanResponse)
            responseDict[param]['tuningCurve'].append(semResponse)
            responseDict[param]['tuningCurve'].append(spontSubtracted)
            responseDict[param]['tuningCurve'].append(zscored)            
            
            
        xv, yv = np.meshgrid(elevSpikeRate, azimuthSpikeRate)
        spotRF = (xv+yv)/2.0-itiRate.mean()
        spotRF_zscore = (spotRF - np.mean(spotRF))/np.std(spotRF)
        if plot:
            plt.figure()
            climMax = max(2, np.max(spotRF_zscore))
            climMin = min(-2, np.min(spotRF_zscore))
            plt.imshow(spotRF_zscore, clim = (climMin, climMax), cmap='gray', interpolation='none', origin='lower')
        
        return spotRF_zscore, trialSpikeRate-np.mean(itiRate), responseDict
            
    def analyzeSpots_units(self, units = None, protocol=3, trials=None, plot=True):
        sortedUnits = [(u[0], u[1]['ypos'], u[1]['times'][str(protocol)]) for u in self.units.iteritems()]
        sortedUnits.sort(key=lambda i: -i[1])    
        
        if units is not None:
            sU = [sortedUnits[u] for u in units]
        else:
            sU = sortedUnits
        
        if plot:
            gs = gridspec.GridSpec(len(sU), 4)
            plt.figure(figsize = (6.0, 17.9), tight_layout = True)
            
        for index in xrange(len(sU)):
            if 'spotRF' in self.units[sU[index][0]].keys():
                spotRF = self.units[sU[index][0]]['spotRF']
            else:
                spotRF, trialSpikeRate, responseDict = self.analyzeSpots(sU[index][2], protocol=protocol, plot=False, trials=trials)
                self.units[sU[index][0]]['spotRF'] = spotRF
            
            if plot:
                a1 = plt.subplot(gs[index, 0])
        
                xyNan = np.transpose(np.where(np.isnan(spotRF)))
                spotRF[np.isnan(spotRF)] = 0
                
                climMax = max(2, np.max(spotRF))
                climMin = min(-2, np.min(spotRF))
                plt.imshow(spotRF, clim=(climMin,climMax), cmap='gray', origin = 'lower', interpolation='none')
                for xypair in xyNan:    
                    a1.text(xypair[1], xypair[0], 'no trials', color='white', ha='center')

                a1.xaxis.set_visible(False)
                a1.yaxis.set_visible(False)
                a1.text(-5.5, 0.5, str(sU[index][0]))
        
                cb = plt.colorbar()
                cb.set_ticks([spotRF.min(), spotRF.max()])
                
                for paramnum, param in enumerate(['trialSpotSize', 'trialSpotDir', 'trialSpotSpeed']):
                    a = plt.subplot(gs[index, paramnum+1])
                    values = responseDict[param]['tuningCurve'][3] 
                    error = responseDict[param]['tuningCurve'][2] 
                    a.plot(responseDict[param]['tuningCurve'][0], values)
                    plt.fill_between(responseDict[param]['tuningCurve'][0], values+error, values-error, alpha=0.3)
                    a.plot(responseDict[param]['tuningCurve'][0], np.zeros(values.size), 'r--')
                    plt.xlabel(param) 
                    plt.ylim(min(-0.1, np.min(values - error)), max(np.max(values + error), 0.1))
                    plt.locator_params(axis = 'y', nbins = 3)
                    if index < len(sU) -1:
                        a.xaxis.set_visible(False)
                    else:
                        a.set_xticks(responseDict[param]['tuningCurve'][0])
                    
#                
#                a2 = plt.subplot(gs[index, 1])
#                a2.plot(responseDict['trialSpotSpeed']['tuningCurve'][0], responseDict['trialSpotSpeed']['tuningCurve'][3])
#                a2.plot(responseDict['trialSpotSpeed']['tuningCurve'][0], )
#                plt.xlabel('speed') 
#                plt.ylim(min(0, np.min(responseDict['trialSpotSpeed']['tuningCurve'][3])), np.max(responseDict['trialSpotSpeed']['tuningCurve'][3]))
#                
#                a3 = plt.subplot(gs[index,2])
#                a3.plot(responseDict['trialSpotSize']['tuningCurve'][0], responseDict['trialSpotSize']['tuningCurve'][3])
#                plt.xlabel('size')
#                plt.ylim(min(0, np.min(responseDict['trialSpotSize']['tuningCurve'][3])), np.max(responseDict['trialSpotSize']['tuningCurve'][3]))
#                
#                a4 = plt.subplot(gs[index,3], projection='polar')
#                theta = (np.pi/180.)*(responseDict['trialSpotDir']['tuningCurve'][0]).astype(float)
#                rho = responseDict['trialSpotDir']['tuningCurve'][3]
#                theta = np.append(theta, theta[0])
#                rho = np.append(rho, rho[0])
#                a4.plot(theta, rho)
#                a4.xaxis.set_visible(False)
#                
#                if index < len(sU) -1:
#                    a2.xaxis.set_visible(False)                    
#                    a3.xaxis.set_visible(False)
                    
    def analyzeCheckerboard(self, units, protocol=None, trials=None, plot=False):
        if protocol is None:
            protocol = self.getProtocolIndex(['checkerboard'])
        else:
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
        
        bckgndSpeed = np.concatenate((-p['bckgndSpeed'][:0:-1],p['bckgndSpeed']))
        patchSpeed = np.concatenate((-p['patchSpeed'][:0:-1],p['patchSpeed']))
        resp = np.full((bckgndSpeed.size,patchSpeed.size,p['patchSize'].size,p['patchElevation'].size),np.nan)
        resp = np.tile(resp[:,:,:,:,None],math.ceil(trials.size/(resp.size-2*p['patchSpeed'].size*p['patchSize'].size))+1)
        if plot:
            plt.figure(facecolor='w')
            row = 0
        for u in units:
            spikesPerTrial = self.findSpikesPerTrial(trialStartSamples, trialEndSamples, self.units[str(u)]['times'][protocol])
            trialSpikeRate = spikesPerTrial/((1/p['frameRate'])*trialDuration)
            for n in trials:
                i = patchSpeed==p['trialPatchSpeed'][n] if p['trialPatchDir'][n]==0 else patchSpeed==-p['trialPatchSpeed'][n]
                j = bckgndSpeed==p['trialBckgndSpeed'][n] if p['trialBckgndDir'][n]==0 else bckgndSpeed==-p['trialBckgndSpeed'][n]
                k = p['patchSize']==p['trialPatchSize'][n]
                l = p['patchElevation']==p['trialPatchPos'][n]
                resp[i,j,k,l,np.count_nonzero(np.logical_not(np.isnan(resp[i,j,k,l,:])))] = trialSpikeRate[n]
            meanResp = np.nanmean(resp,axis=4)
            meanResp -= np.nanmean(resp[patchSpeed.size//2,bckgndSpeed.size//2,:,:,:])
            meanResp /= np.nanstd(resp[patchSpeed.size//2,bckgndSpeed.size//2,:,:,:])
            for k in range(p['patchSize'].size):
                for l in range(p['patchElevation'].size):
                    meanResp[patchSpeed.size//2,:,k,l] = meanResp[patchSpeed.size//2,:,0,0]
            for i in range(patchSpeed.size):
                for j in range(bckgndSpeed.size):
                    if patchSpeed[i]==bckgndSpeed[j]:
                        meanResp[i,j,:,:] = meanResp[patchSpeed.size//2,j]            
            self.units[str(u)]['checkerboard'] = {'meanResp':meanResp}
            resp[:,:,:,:,:] = np.nan
            
            if plot:
                gs = gridspec.GridSpec(2*len(units),4)
                ax = plt.subplot(gs[row:row+2,0:2])
                r = np.nanmean(np.nanmean(meanResp,axis=3),axis=2)
                cLim = max(1,np.max(abs(r)))
                plt.imshow(r,cmap='bwr',clim=(-cLim,cLim),interpolation='none',origin='lower')
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if row>2*len(units)-3:
                    ax.xaxis.set_ticks_position('bottom')
                    ax.xaxis.set_tick_params(direction='out')
                    ax.set_xticks(range(bckgndSpeed.size))
                    ax.set_xticklabels(bckgndSpeed[:])
                    ax.set_xlabel('Background Speed')
                else:
                    ax.set_xticks([])
                if row==0:
                    ax.yaxis.set_ticks_position('left')
                    ax.yaxis.set_tick_params(direction='out')
                    ax.set_yticks(range(patchSpeed.size))
                    ax.set_yticklabels(patchSpeed[:])
                    ax.set_ylabel('Patch Speed')
                else:
                    ax.set_yticks([])
                ax.set_title('Unit '+str(u))
                cb = plt.colorbar(fraction=0.05,pad=0.04)
                cb.set_ticks([int(r.min()),int(r.max())])
                
                ax = plt.subplot(gs[row:row+2,2])
                r = [np.nanmean(meanResp[patchSpeed!=0,bckgndSpeed.size//2,k,:]) for k in range(p['patchSize'].size)]
                plt.plot(p['patchSize'],r)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.yaxis.set_tick_params(direction='out')
                ax.set_yticks([int(min(r)),int(max(r))])
                if row>2*len(units)-3:
                    ax.xaxis.set_ticks_position('bottom')
                    ax.xaxis.set_tick_params(direction='out')
                    ax.set_xticks(p['patchSize'])
                    ax.set_xlabel('Patch Size')
                else:
                    ax.set_xticks([])
                if row==0:
                    ax.set_ylabel('Z Score')
                
                ax = plt.subplot(gs[row:row+2,3])
                r = [np.nanmean(meanResp[patchSpeed!=0,bckgndSpeed.size//2,:,l]) for l in range(p['patchElevation'].size)]
                plt.plot(p['patchElevation'],r)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.yaxis.set_tick_params(direction='out')
                ax.set_yticks([int(min(r)),int(max(r))])
                if row>2*len(units)-3:
                    ax.xaxis.set_ticks_position('bottom')
                    ax.xaxis.set_tick_params(direction='out')
                    ax.set_xticks(p['patchElevation'])
                    ax.set_xlabel('Patch Elevation')
                else:
                    ax.set_xticks([])
                row += 2
        if plot:
            plt.tight_layout()
        
    def getProtocolIndex(self,labels):
        if isinstance(labels,str):
            labels = [labels]
        protocol = []
        for pro in labels:
            protocol.extend([str(index) for index, f in enumerate(self.kwdFileList) if pro in f])
        if len(protocol)<1:
            raise ValueError('No protocols found matching: '+pro)
        elif len(protocol)>1:
            raise ValueError('Multiple protocols found matching: '+pro)
        return str(protocol[0])
    
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
        
        print variables
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
            fileName = self.getFile()
        shelf = shelve.open(fileName)
        
        for key in shelf:
            setattr(self, key, shelf[key])
        
        shelf.close()
    
    
    def loadClusteredData(self, kwdNsamplesList = None, protocolsToAnalyze = None, fileDir = None):
        from load_phy_template import load_phy_template
                 
        if fileDir is None:
            fileDir = self.getDir()
        
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

            
    def getKwdInfo(self, fileDir = None):
        from makeDat import getKwdFiles
        if fileDir is None:
            fileList, nsamps = getKwdFiles()
        else:
            fileList, nsamps = getKwdFiles(fileDir)
        
        return fileList, nsamps
        
    def getSingleUnits(self, fileDir = None, protocolsToAnalyze = None):
        if fileDir is None:
            fileDir = self.getDir()
        fileList, nsamps = self.getKwdInfo(fileDir=fileDir)
        if protocolsToAnalyze is None:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir)
        else:
            self.loadClusteredData(kwdNsamplesList=nsamps, fileDir=fileDir, protocolsToAnalyze=protocolsToAnalyze)
        
if __name__=="__main__":
    pass
        