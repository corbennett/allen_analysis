# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:07:49 2016

@author: svc_ccg
"""

from __future__ import division
import clust, fileIO, probeData
import cv2, datetime, math, nrrd, os, re, itertools
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from matplotlib import cm


class popProbeData():
    
    def __init__(self):
        self.experimentFiles = None
        self.excelFile = None
        self.data = None
    
    
    def getExperimentFiles(self,append=False):
        filePaths = fileIO.getFiles(caption='Choose experiments',fileType='*.hdf5')
        if len(filePaths)<1:
            return
        if append and self.experimentFiles is not None:
            self.experimentFiles.append(filePaths)
        else:
            self.experimentFiles = filePaths
        # sort by date
        expDates,_ = self.getExperimentInfo()
        self.experimentFiles = [z[0] for z in sorted(zip(filePaths,[datetime.datetime.strptime(date,'%m%d%Y') for date in expDates]),key=lambda i: i[1])]
        
        
    def getExperimentInfo(self):
        expDate =[]
        anmID = []
        for exp in self.experimentFiles:
            match = re.search('\d{8,8}_\d{6,6}',os.path.basename(exp))
            expDate.append(match.group()[:8])
            anmID.append(match.group()[9:])
        return expDate,anmID
        
        
    def analyzeExperiments(self,exps=None,protocols=None,save=False):
        if exps is None:
            exps = self.experimentFiles
        for ind,exp in enumerate(exps):
            print('Analyzing experiment '+str(ind+1)+' of '+str(len(exps)))
            p = self.getProbeDataObj(exp)
            self.getUnitLabels(p)
            p.runAllAnalyses(splitRunning=True,protocolsToRun=protocols,plot=False)
            if save:
                p.saveHDF5(exp)
        
            
    def getProbeDataObj(self,experimentFilePath):
        p = probeData.probeData()
        p.loadHDF5(experimentFilePath)
        return p
    
        
    def getUnitLabels(self,probeDataObj=None,save=False):
        if self.excelFile is None:
            filePath = fileIO.getFile(caption='Choose excel file with unit labels',fileType='*.xlsx')
            if filePath=='':
                return
            self.excelFile = filePath
        if probeDataObj is None:
            for exp in self.experimentFiles:
                p = self.getProbeDataObj(exp)
                p.readExcelFile(fileName=self.excelFile)
                if save:
                    p.saveHDF5(exp)
        else:
            probeDataObj.readExcelFile(fileName=self.excelFile)
    
    
    def makeDataFrame(self,analyzeExperiments=False):
        # determine which experiments to append to dataframe
        if self.experimentFiles is None:
            self.getExperimentFiles()
        if self.data is None:
            exps = self.experimentFiles
        else:
            expDate,anmID = self.getExperimentInfo()
            dataFrameExps = set(self.data.index.get_level_values('experimentDate'))
            dataFrameAnms = set(self.data.index.get_level_values('animalID'))
            exps = [self.experimentFiles[i] for i,(date,anm) in enumerate(zip(expDate,anmID)) if date not in dataFrameExps and anm not in dataFrameAnms]
            if len(exps)<1:
                return
        
        if analyzeExperiments:
            self.analyzeExperiments(exps)
            
        # dataFrame rows
        rowNames = ('experimentDate','animalID','unitID','unitLabel','ccfX','ccfY','ccfZ')
        experimentDate = []
        animalID = []
        unitID = []
        unitLabel = []
        ccfX = []
        ccfY = []
        ccfZ = []
        
        # dataFrame columns
        columnNames = ('laserState','runState','paramType','paramName')
        laserState = []
        runState = []
        paramType = []
        paramName = []
        
        # data is dictionary of paramter type (protocol) dictionaries that are converted to dataframe
        # each parameter type dictionary has keys corresponding to parameters
        # the value for each parameter is a len(units) list
        laserLabels = ('laserOff','laserOn')
        runLabels = ('allTrials','stat','run')
        protocols = ('sparseNoise','gratings','gratings_ori','checkerboard','loom')
        data = {laserLabel: {runLabel: {protocol: {} for protocol in protocols} for runLabel in runLabels} for laserLabel in laserLabels}
        for exp in exps:
            p = self.getProbeDataObj(exp)
            expDate,anmID = p.getExperimentInfo()
            units = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            units,ypos = p.getOrderedUnits(units)
            for u in units:
                experimentDate.append(expDate)
                animalID.append(anmID)
                unitID.append(u)
                unitLabel.append(p.units[u]['label'])
                for i,c in enumerate((ccfX,ccfY,ccfZ)):
                    c.append(p.units[u]['CCFCoords'][i])
                for laserLabel in laserLabels:
                    for runLabel in runLabels:
                        for protocol in protocols:
                            tag = 'gratings_stf' if protocol=='gratings' else protocol
                            tag += '_'+laserLabel+'_'+runLabel
                            if tag not in p.units[u]:
                                for prm in data[laserLabel][runLabel][protocol]:
                                    data[laserLabel][runLabel][protocol][prm].append(np.nan)
                            else:
                                for prm,val in p.units[u][tag].items():
                                    if prm not in data[laserLabel][runLabel][protocol]:
                                        laserState.append(laserLabel)
                                        runState.append(runLabel)
                                        paramType.append(protocol)
                                        paramName.append(prm)
                                        data[laserLabel][runLabel][protocol][prm] = [np.nan for _ in range(len(unitID)-1)]
                                    data[laserLabel][runLabel][protocol][prm].append(val)
        
        # build dataframe
        rows = pd.MultiIndex.from_arrays([experimentDate,animalID,unitID,unitLabel,ccfX,ccfY,ccfZ],names=rowNames)
        cols = pd.MultiIndex.from_arrays([laserState,runState,paramType,paramName],names=columnNames)
        dframe = pd.DataFrame(index=rows,columns=cols)
        for laserLabel in data:
            for runLabel in data[laserLabel]:
                for prmType in data[laserLabel][runLabel]:
                    for prmName in data[laserLabel][runLabel][prmType]:
                        dframe[laserLabel,runLabel,prmType,prmName] = data[laserLabel][runLabel][prmType][prmName]
        
        self.data = dframe if self.data is None else pd.concat((self.data,dframe))
    
    
    def loadDataFrame(self, filePath=None):
        filePath = fileIO.getFile(fileType='*.hdf5')
        if filePath=='':
            return
        self.data = pd.read_hdf(filePath,'table')
    
    
    def saveDataFrame(self, filePath=None):
        filePath = fileIO.saveFile(fileType='*.hdf5')
        if filePath=='':
            return
        self.data.to_hdf(filePath,'table')
        
        
    def getSCAxons(self):
        ccfX = np.array(self.data.index.get_level_values('ccfX'))    
        ccfZ = np.array(self.data.index.get_level_values('ccfZ'))
        return np.logical_and(ccfX<=170*25,ccfZ>=300*25)
    
                
    def analyzeRF(self):

        inSCAxons = self.getSCAxons()
        
        data = self.data.laserOff.allTrials.sparseNoise        
        
        isOnOff = self.data.index.get_level_values('unitLabel')=='on off'
        isOn = self.data.index.get_level_values('unitLabel')=='on'
        isOff = self.data.index.get_level_values('unitLabel')=='off'  
        noRF = np.logical_not(isOnOff | isOn | isOff) 
        
        onVsOff = data.onVsOff
        onVsOff[noRF] = np.nan
        
        respLatency = np.stack(data.respLatency)*1000
        respLatency[isOff | noRF,0] = np.nan
        respLatency[isOn | noRF,1] = np.nan
        latencyCombined = np.nanmean(respLatency,axis=1)
        
        onFit = np.full((self.data.shape[0],7),np.nan)
        offFit = onFit.copy()
        sizeUsed = np.full(self.data.shape[0],np.nan)
        for u in range(self.data.shape[0]):
            for size in (10,5,20):
                sizeInd = data.boxSize[u]==size
                if sizeInd.any():
                    onFit[u] = data.onFit[u][sizeInd]
                    offFit[u] = data.offFit[u][sizeInd]
                    if not np.all(np.isnan(onFit[u])) or not np.all(np.isnan(offFit[u])):
                        sizeUsed[u] = size
                        break
        sizeUsed[noRF] = np.nan
        onFit[isOff | noRF,:] = np.nan
        offFit[isOn | noRF,:] = np.nan
        minRFCutoff = 100
        maxRFCutoff = 5000
        minAspectCutoff = 0.25
        maxAspectCutoff = 4
        onArea = np.pi*np.prod(onFit[:,2:4],axis=1)
        onAspect = onFit[:,2]/onFit[:,3]
        badOn = (onArea<minRFCutoff) | (onArea>maxRFCutoff) | (onAspect<minAspectCutoff) | (onAspect>maxAspectCutoff)
        onArea[badOn] = np.nan
        onAspect[badOn] = np.nan
        offArea = np.pi*np.prod(offFit[:,2:4],axis=1)
        offAspect = offFit[:,2]/offFit[:,3]
        badOff = (offArea<minRFCutoff) | (offArea>maxRFCutoff) | (offAspect<minAspectCutoff) | (offAspect>maxAspectCutoff)
        offArea[badOff] = np.nan
        offAspect[badOff] = np.nan
        rfArea = offArea.copy()
        rfArea[onVsOff>0] = onArea[onVsOff>0].copy()
        rfAspect = offAspect.copy()
        rfAspect[onVsOff>0] = onAspect[onVsOff>0].copy()
        
        # plot on vs off resp index
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        counts,bins,bars = ax.hist(onVsOff[~np.isnan(onVsOff)],bins=np.arange(-1,1,0.1),color='k')
        ax.plot([0,0],[0,1.1*max(counts)],'--',color='0.5',linewidth=3)
        ax.text(-0.5,max(counts),'Off',fontsize='x-large',horizontalalignment='center')
        ax.text(0.5,max(counts),'On',fontsize='x-large',horizontalalignment='center')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='large')
        ax.set_xlabel('Off vs On Index',fontsize='x-large')
        ax.set_ylabel('Number of Cells',fontsize='x-large')
        plt.tight_layout()
        
        # plot response latency, on/off averaged
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.hist(latencyCombined[~np.isnan(latencyCombined)],bins=np.arange(0,160,10),color='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='large')
        ax.set_xlabel('Response Latency (ms)',fontsize='x-large')
        ax.set_ylabel('Number of Cells',fontsize='x-large')
        plt.tight_layout()
        
        # plot receptive field area
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.hist(rfArea[~np.isnan(rfArea)],bins=np.arange(0,maxRFCutoff,200),color='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='large')
        ax.set_xlabel('Receptive Field Area ($\mathregular{degrees^2}$)',fontsize='x-large')
        ax.set_ylabel('Number of Cells',fontsize='x-large')
        plt.tight_layout()
        
        # plot asepect ratio
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        counts,bins,bars = ax.hist(rfAspect[~np.isnan(rfAspect)],bins=np.arange(0,maxAspectCutoff,0.2),color='k')
        ax.plot([1,1],[0,1.1*max(counts)],'--',color='0.5',linewidth=3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='large')
        ax.set_xlabel('Aspect Ratio',fontsize='x-large')
        ax.set_ylabel('Number of Cells',fontsize='x-large')
        plt.tight_layout()
        
        # plot RF area and aspect ration in and out of SC axons
        for rfa in (rfArea[inSCAxons],rfArea[~inSCAxons]):
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            ax.hist(rfa[~np.isnan(rfa)],bins=np.arange(0,maxRFCutoff,200),color='k')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xlabel('Receptive Field Area ($\mathregular{degrees^2}$)',fontsize=20)
            ax.set_ylabel('Number of Cells',fontsize=20)
            plt.tight_layout()
        
        for aspect in (rfAspect[inSCAxons],rfAspect[~inSCAxons]):
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            counts,bins,bars = ax.hist(aspect[~np.isnan(aspect)],bins=np.arange(0,maxAspectCutoff,0.2),color='k')
            ax.plot([1,1],[0,1.1*max(counts)],'--',color='0.5',linewidth=3)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xlabel('Aspect Ratio',fontsize=20)
            ax.set_ylabel('Number of Cells',fontsize=20)
            plt.tight_layout()
            
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        rfAreaSC = rfArea[~np.isnan(rfArea) & inSCAxons]
        rfAreaNonSC  = rfArea[~np.isnan(rfArea) & ~inSCAxons]
        cumProbSC = [np.count_nonzero(rfAreaSC<=i)/rfAreaSC.size for i in np.sort(rfAreaSC)]
        cumProbNonSC = [np.count_nonzero(rfAreaNonSC<=i)/rfAreaNonSC.size for i in np.sort(rfAreaNonSC)]
        ax.plot([0,0],[0,1],'k--')
        ax.plot(np.sort(rfAreaNonSC),cumProbNonSC,'0.6',linewidth=3)
        ax.plot(np.sort(rfAreaSC),cumProbSC,'k',linewidth=3)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([0,maxRFCutoff])
        ax.set_ylim([0,1.01])
        ax.set_yticks([0,0.5,1])
        ax.set_xlabel('RF Area',fontsize=20)
        ax.set_ylabel('Cumulative Probability',fontsize=20)
        plt.tight_layout()
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        aspectSC = rfAspect[~np.isnan(rfAspect) & inSCAxons]
        aspectNonSC  = rfAspect[~np.isnan(rfAspect) & ~inSCAxons]
        cumProbSC = [np.count_nonzero(aspectSC<=i)/aspectSC.size for i in np.sort(aspectSC)]
        cumProbNonSC = [np.count_nonzero(aspectNonSC<=i)/aspectNonSC.size for i in np.sort(aspectNonSC)]
        ax.plot([0,0],[0,1],'k--')
        ax.plot(np.sort(aspectNonSC),cumProbNonSC,'0.6',linewidth=3)
        ax.plot(np.sort(aspectSC),cumProbSC,'r',linewidth=3)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([0,maxAspectCutoff])
        ax.set_ylim([0,1.01])
        ax.set_yticks([0,0.5,1])
        ax.set_xlabel('Aspect Ratio',fontsize=20)
        ax.set_ylabel('Cumulative Probability',fontsize=20)
        plt.tight_layout()
        
        # plot on and off resp latency
        for i,clr in zip((0,1),('r','b')):
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            ax.hist(respLatency[~np.isnan(respLatency[:,i]),i],bins=np.arange(0,160,10),color=clr)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='large')
            ax.set_xlabel('Response Latency (ms)',fontsize='x-large')
            ax.set_ylabel('Number of Cells',fontsize='x-large')
            plt.tight_layout()
        
        # plot on and off rf area
        for rfA,clr in zip((onArea,offArea),('r','b')):
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            ax.hist(rfA[~np.isnan(rfA)],bins=np.arange(0,maxRFCutoff,200),color=clr)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='large')
            ax.set_xlabel('Receptive Field Area ($\mathregular{degrees^2}$)',fontsize='x-large')
            ax.set_ylabel('Number of Cells',fontsize='x-large')
            plt.tight_layout()
            
        # size tuning plots: all trials, SC vs nonSC
        sizeTuningSize = [5,10,20,50]
        sizeTuningLabel = [5,10,20,'full']
        allSizesInd = np.zeros(data.shape[0],dtype=bool)
        for i in range(data.shape[0]):
            if data.sizeTuningOff[i].size>3:
                allSizesInd[i] = True      
        for scInd in (inSCAxons,~inSCAxons):
            ind = allSizesInd & scInd
            sizeTuningOn = np.stack(data.sizeTuningOn[ind])[isOn[ind] | isOnOff[ind]]
            sizeTuningOff = np.stack(data.sizeTuningOff[ind])[isOff[ind] | isOnOff[ind]]
            
            sizeResp = sizeTuningOff.copy()
            sizeResp[onVsOff>0] = sizeTuningOn[onVsOff>0].copy()
            
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            sizeRespNorm = sizeResp/np.nanmax(sizeResp,axis=1)[:,None]
            sizeRespMean = np.nanmean(sizeRespNorm,axis=0)
            sizeRespStd = np.nanstd(sizeRespNorm,axis=0)
            ax.plot(sizeTuningSize,sizeRespMean,color='k',linewidth=3)
            ax.fill_between(sizeTuningSize,sizeRespMean+sizeRespStd,sizeRespMean-sizeRespStd,color='k',alpha=0.3)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xlim([0,55])
            ax.set_xticks(sizeTuningSize)
            ax.set_xticklabels(sizeTuningLabel)
            ax.set_yticks([0,0.5,1])
            ax.set_xlabel('Size (degrees)',fontsize=20)
            ax.set_ylabel('Norm. Response',fontsize=20)
            plt.tight_layout()
            
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            sizeRespNorm[sizeRespNorm<1] = 0
            bestSizeCount = np.nansum(sizeRespNorm,axis=0)
            ax.bar(sizeTuningSize,bestSizeCount,color='k')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xlim([0,55])
            ax.set_xticks(sizeTuningSize)
            ax.set_xticklabels(sizeTuningLabel)
            ax.set_xlabel('Best Size (degrees)',fontsize=20)
            ax.set_ylabel('Number of Cells',fontsize=20)
            plt.tight_layout()
        
        # size tuning plots: run vs stat comparison
        sizeTuningSize = [5,10,20,50]
        sizeTuningLabel = [5,10,20,'full']
        allSizesInd = [u for u, _ in enumerate(data.sizeTuningOff) if data.sizeTuningOff[u].size > 3]
        sizeTuningOn = np.stack(data.sizeTuningOn[allSizesInd])[isOn[allSizesInd] | isOnOff[allSizesInd]]
        sizeTuningOff = np.stack(data.sizeTuningOff[allSizesInd])[isOff[allSizesInd] | isOnOff[allSizesInd]]
        goodUnits = []
        self.data.index.get_level_values('unitLabel')
        for sub,tuning in zip(['on', 'off'], ['sizeTuningOn', 'sizeTuningOff']):
            good = []
            for u in xrange(data.shape[0]):
                st_r = self.data.laserOff.run.sparseNoise[tuning][u]
                st_s = self.data.laserOff.stat.sparseNoise[tuning][u]
                
                if st_r.size > 3 and st_s.size > 3:
                    if ~any(np.isnan(st_r)) and ~any(np.isnan(st_s)):
                        label = self.data.index.get_level_values('unitLabel')[u]
                        if sub in label:                
                            good.append(u)
            goodUnits.append(good)
            
        for stateData,state in zip([self.data.laserOff.stat.sparseNoise, self.data.laserOff.run.sparseNoise], ['stat', 'run']):
            sizeTuningOn = np.stack(stateData.sizeTuningOn[goodUnits[0]])
            sizeTuningOff = np.stack(stateData.sizeTuningOff[goodUnits[1]])
            
            fig = plt.figure(state, facecolor='w')
            for axind,(sizeResp,clr) in enumerate(zip((sizeTuningOn,sizeTuningOff),('r','b'))):
                
                ax = fig.add_subplot(2,2,2*axind + 1)
                sizeRespNorm = sizeResp/np.nanmax(sizeResp,axis=1)[:,None]
                sizeRespMean = np.nanmean(sizeRespNorm,axis=0)
                sizeRespStd = np.nanstd(sizeRespNorm,axis=0)
                ax.plot(sizeTuningSize,sizeRespMean,color=clr,linewidth=3)
                ax.fill_between(sizeTuningSize,sizeRespMean+sizeRespStd,sizeRespMean-sizeRespStd,color=clr,alpha=0.3)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='large')
                ax.set_xlim([0,55])
                ax.set_xticks(sizeTuningSize)
                ax.set_xticklabels(sizeTuningLabel)
                ax.set_yticks([0,0.5,1])
                ax.set_xlabel('Size (degrees)',fontsize='x-large')
                ax.set_ylabel('Norm. Response',fontsize='x-large')
                plt.tight_layout()
                
                ax = fig.add_subplot(2,2,2*axind + 2)
                sizeRespNorm[sizeRespNorm<1] = 0
                bestSizeCount = np.nansum(sizeRespNorm,axis=0)
                ax.bar(sizeTuningSize,bestSizeCount,color=clr)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='large')
                ax.set_xlim([0,55])
                ax.set_xticks(sizeTuningSize)
                ax.set_xticklabels(sizeTuningLabel)
                ax.set_xlabel('Size (degrees)',fontsize='x-large')
                ax.set_ylabel('Best Size (# Units)',fontsize='x-large')
        
        #center of mass for size tuning curves
        sizeTuningCOM = []
        sizes = [5, 10, 20]
        for u, (on, off) in enumerate(zip(data.sizeTuningOn, data.sizeTuningOff)):
            if len(on) > 3:
                if isOn[u]:
                    sizeTuningCOM.append(np.average(sizes, weights=on[:3]))
                elif isOff[u]:
                    sizeTuningCOM.append(np.average(sizes, weights=off[:3]))
                elif isOnOff[u]:
                    if data.onVsOff[u] <= 0:
                        sizeTuningCOM.append(np.average(sizes, weights=off[:3]))
                    else:
                        sizeTuningCOM.append(np.average(sizes, weights=on[:3]))
                else:
                    sizeTuningCOM.append(np.nan)
            else:
                sizeTuningCOM.append(np.nan)
        sizeTuningCOM = np.array(sizeTuningCOM)
        plt.figure()        
        plt.hist(sizeTuningCOM[~np.isnan(sizeTuningCOM)])
        
        # plot all rf centers
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        onInd = (onFit[:,0]>-30) & (onFit[:,0]<130) & (onFit[:,1]>-40) & (onFit[:,1]<95)
        offInd = (offFit[:,0]>-30) & (offFit[:,0]<130) & (offFit[:,1]>-40) & (offFit[:,1]<95)
        plt.plot(onFit[onInd,0],onFit[onInd,1],'ro',label='On')
        plt.plot(offFit[offInd,0],offFit[offInd,1],'bo',label='Off')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlabel('Azimuth',fontsize=20)
        ax.set_ylabel('Elevation',fontsize=20)
        # plt.legend(loc='upper left',numpoints=1,frameon=False,fontsize=18)
        ax.set_aspect('equal')
        plt.tight_layout()
        
        # plot rf center vs probe position
        expDate = self.data.index.get_level_values('experimentDate')
        for exp in expDate.unique():
            ind = expDate==exp
            depth = ccfY[ind]
            minDepth = depth.min()
            maxDepth = depth.max()
            depthRange = maxDepth-minDepth
            xlim = np.array([minDepth-0.05*depthRange,maxDepth+0.05*depthRange])
            plt.figure(facecolor='w')
            gs = gridspec.GridSpec(2,2)
            for i,azimOrElev in enumerate(('Azimuth','Elevation')):
                for j,(rf,clr) in enumerate(zip((onFit,offFit),('r','b'))):
                    ax = plt.subplot(gs[i,j])
                    hasRF = np.logical_not(np.isnan(rf[ind,i]))
                    linFit = scipy.stats.linregress(depth[hasRF],rf[ind,i][hasRF])
                    ax.plot(xlim,linFit[0]*xlim+linFit[1],color='0.6',linewidth=2)
                    ax.plot(depth,rf[ind,i],'o',color=clr,markersize=10)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='large')
                    ax.set_xlim(xlim)
                    ylim = (-20,120) if azimOrElev=='Azimuth' else (-40,80)
                    ax.set_ylim(ylim)
                    ax.set_xticks(ax.get_xticks()[[0,-1]])
                    ax.set_yticks(ax.get_yticks()[[0,-1]])
                    if i==1:
                        ax.set_xlabel('Depth (microns)',fontsize='x-large')
                    else:
                        ax.set_xticklabels([])
                    if j==0:
                        ax.set_ylabel(azimOrElev+' (degrees)',fontsize='x-large')
                    else:
                        ax.set_yticklabels([])
                    plt.tight_layout()
        
        # rf bubble plots
        colors = [(1, 0, 0), (0, 0, 1)]
        for i, fit in enumerate([onFit, offFit]):
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(111, aspect='equal')
            
            for sub in fit:
                x,y = probeData.getEllipseXY(*sub[:5])
                if sub[3]<40:
                    e = Ellipse(xy=[sub[0], sub[1]], width=sub[2], height=sub[3], angle=sub[4])
                    e.set_edgecolor('none')
                    e.set_alpha(0.5)
                    color = cm.jet((sub[1] + 40)/110.0)
                    e.set_facecolor(color)
                    ax.add_artist(e)
        
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='large')
            ax.set_xlabel('Azimuth (degrees)',fontsize='x-large')
            ax.set_ylabel('Elevation (degrees)',fontsize='x-large')
            
            ax.set_xlim(-30, 120)
            ax.set_ylim(-45, 85)
            fig.tight_layout()
            
            
    def compareAdjustedRFArea(self):
        rfAreaAdjusted = []
        for i,exp in enumerate(self.experimentFiles):
            print('analyzing experiment '+str(i+1)+' of '+str(len(self.experimentFiles)))
            p = self.getProbeDataObj(exp)
            units = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            rfAreaAdjusted.append(p.findRF(units,adjustForPupil=True,plot=False))
        onAreaAdjusted,offAreaAdjusted = np.concatenate(rfAreaAdjusted).T
            
        isOnOff = self.data.index.get_level_values('unitLabel')=='on off'
        isOn = self.data.index.get_level_values('unitLabel')=='on'
        isOff = self.data.index.get_level_values('unitLabel')=='off'  
        noRF = np.logical_not(isOnOff | isOn | isOff)
        
        data = self.data.laserOff.allTrials.sparseNoise
        onFit = np.full((self.data.shape[0],7),np.nan)
        offFit = onFit.copy()
        for u in range(self.data.shape[0]):
            sizeInd = data.boxSize[u]==10
            onFit[u] = data.onFit[u][sizeInd]
            offFit[u] = data.offFit[u][sizeInd]
        onFit[isOff | noRF,:] = np.nan
        offFit[isOn | noRF,:] = np.nan
        minRFCutoff = 100
        maxRFCutoff = 5000
        onArea = np.pi*np.prod(onFit[:,2:4],axis=1)
        onArea[np.logical_or(onArea<minRFCutoff,onArea>maxRFCutoff)] = np.nan
        offArea = np.pi*np.prod(offFit[:,2:4],axis=1)
        offArea[np.logical_or(offArea<minRFCutoff,offArea>maxRFCutoff)] = np.nan
        
        onAreaAdjusted[isOff | noRF] = np.nan
        offAreaAdjusted[isOn | noRF] = np.nan
        onAreaAdjusted[np.logical_or(onAreaAdjusted<minRFCutoff,onAreaAdjusted>maxRFCutoff)] = np.nan
        offAreaAdjusted[np.logical_or(offAreaAdjusted<minRFCutoff,offAreaAdjusted>maxRFCutoff)] = np.nan
        
        onInd = np.logical_not(np.logical_or(np.isnan(onArea),np.isnan(onAreaAdjusted)))
        offInd = np.logical_not(np.logical_or(np.isnan(offArea),np.isnan(offAreaAdjusted)))
        _,pvalOn = scipy.stats.wilcoxon(onArea[onInd],onAreaAdjusted[onInd])
        _,pvalOff = scipy.stats.wilcoxon(offArea[offInd],offAreaAdjusted[offInd])
        print('median On RF Area = '+str(np.median(onArea[onInd]))+', adjusted = '+str(np.median(onAreaAdjusted[onInd]))+', p = '+str(pvalOn))
        print('median Off RF Area = '+str(np.median(offArea[offInd]))+', adjusted = '+str(np.median(offAreaAdjusted[offInd]))+', p = '+str(pvalOff))
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,maxRFCutoff],[0,maxRFCutoff],'k--')
        ax.plot(onArea,onAreaAdjusted,'ro',label='On')
        ax.plot(offArea,offAreaAdjusted,'bo',label='Off')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([0,maxRFCutoff])
        ax.set_ylim([0,maxRFCutoff])
        ax.set_xlabel('RF Area ($\mathregular{deg^2}$)',fontsize=20)
        ax.set_ylabel('Adjusted RF Area ($\mathregular{deg^2}$)',fontsize=20)
        plt.legend(loc='lower right',numpoints=1,frameon=False,fontsize=18)
        ax.set_aspect('equal')
        plt.tight_layout()
                
    
    def makeRFVolume(self, padding=10, sigma=1, annotationDataFile=None, weighted=False):
        if annotationDataFile is None:
            annotationDataFile = fileIO.getFile() 
        
        annotationData,_ = nrrd.read(annotationDataFile)
        annotationData = annotationData.transpose((1,2,0))
        
        inLPbinary = annotationData==218
        inLPbinary[:,inLPbinary.shape[1]//2:,:] = False
        inLP = np.where(inLPbinary)
        
        #find left hemisphere region for xRange
#        maxProj = np.max(inLPbinary, axis=2).astype(int)                
#        cnts,_ = cv2.findContours(maxProj.copy(order='C').astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#        leftSide = np.argmin([np.min(u[:, :, 0]) for u in cnts])
        
#        xRange = [np.min(cnts[leftSide][:, :, 0]) - padding, np.max(cnts[leftSide][:, :, 0]) + padding]
        
        yRange = [np.min(inLP[0])-padding, np.max(inLP[0])+padding]
        xRange = [np.min(inLP[1])-padding, np.max(inLP[1])+padding]
        zRange = [np.min(inLP[2])-padding, np.max(inLP[2])+padding]
        
        LPmask = inLPbinary[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]].astype(np.float)
        LPmask[LPmask==0] = np.nan
        
        CCFCoords = np.stack((self.data.index.get_level_values(c) for c in ('ccfX','ccfY','ccfZ')),axis=1)
        
        counts = np.zeros([np.diff(yRange), np.diff(xRange), np.diff(zRange)])
        elev = np.zeros_like(counts)
        azi = np.zeros_like(counts)
        data = self.data.laserOff.allTrials.sparseNoise
        for fitType, sub in zip(['onFit', 'offFit'], ['on', 'off']):
            for uindex, coords in enumerate(CCFCoords):
                if any(np.isnan(coords)):
                    continue
                else:
                    label = self.data.index.get_level_values('unitLabel')[uindex]
                    if sub in label:
                        ccf = coords/25
                        ccf = ccf.astype(int)
                        ccf -= np.array([xRange[0], yRange[0], zRange[0]])
                        
                        counts[ccf[1], ccf[0], ccf[2]]+=1
                        x,y = data[fitType][uindex][data.boxSize[uindex]==10][0][:2]
                        elev[ccf[1], ccf[0], ccf[2]] += y
                        azi[ccf[1], ccf[0], ccf[2]] += x
                        
        # plot recording positions
        for a in range(3):
            anyCounts = 255-counts.any(axis=a).astype(np.uint8)*255
            anyCounts = np.stack([anyCounts]*3,axis=-1)
            _,contours,_ = cv2.findContours(LPmask.astype(np.uint8).max(axis=a).copy(order='C'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(anyCounts,contours,-1,(127,)*3)
            if a==0:
                anyCounts = anyCounts.transpose(1,0,2)
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            ax.imshow(anyCounts)
                
        if not weighted:
            elev /= counts
            azi /= counts
        
        elev_s = probeData.gaussianConvolve3D(elev,sigma)
        azi_s = probeData.gaussianConvolve3D(azi, sigma)
        
        elev_s *= LPmask
        azi_s *= LPmask
        
        if weighted:
            counts_s = probeData.gaussianConvolve3D(counts, sigma)
            elev_s /= counts_s
            azi_s /= counts_s
     
        mat = np.zeros((sigma*3,)*3)
        i = (sigma*3)//2
        mat[i,i,i] = 1
        maskThresh = probeData.gaussianConvolve3D((mat>0).astype(float),sigma=sigma).max()*0.5
        mask = probeData.gaussianConvolve3D((counts>0).astype(float),sigma=sigma)
        elev_s[mask<maskThresh] = np.nan
        azi_s[mask<maskThresh] = np.nan
        
        minVal = [np.nanmin(elev_s), np.nanmin(azi_s)]
        maxVal = [np.nanmax(elev_s- minVal[0]) , np.nanmax(azi_s- minVal[1])]
        
        colorMaps = [np.full(elev_s.shape+(3,),np.nan) for _ in (0,1)]
        for im, m in enumerate([elev_s, azi_s]):
            for y in xrange(m.shape[0]):
                for x in xrange(m.shape[1]):
                    for z in xrange(m.shape[2]):
                        thisVox = (m[y,x,z] - minVal[im])/maxVal[im]
                        if not np.isnan(thisVox):
                            RGB = cm.jet(thisVox)
                            for i in (0,1,2):
                                colorMaps[im][y, x, z, i] = RGB[i]
                            
        fullShape = annotationData.shape+(3,)
        del(annotationData,inLP,inLPbinary,LPmask,elev,azi,elev_s,azi_s,counts,mask)
                            
        fullMap = [np.full(fullShape,np.nan) for _ in (0,1)]
        for i in (0,1):
            fullMap[i][yRange[0]:yRange[1],xRange[0]:xRange[1],zRange[0]:zRange[1]] = colorMaps[i]

        return fullMap, elev_s, azi_s
    
    
    def makeVolume(self, vals, units=None, padding=10, sigma=1, regions=[218], annotationDataFile=None, rgbVolume=True, weighted=False):
        if units is not None:        
            ind = self.data.index.get_level_values('unitID').isin(units)       
            data = self.data[ind]        
        else:
            data = self.data
        data = self.data.ix[uindex]   
        CCFCoords = np.stack((data.index.get_level_values(c) for c in ('ccfX','ccfY','ccfZ')),axis=1)
        
        if annotationDataFile is None:
            annotationDataFile = fileIO.getFile() 
        
        annotationData,_ = nrrd.read(annotationDataFile)
        annotationData = annotationData.transpose((1,2,0))
        
        inRegionBinary = np.zeros_like(annotationData)         
        for r in regions:
            thisRegion = annotationData == r
            inRegionBinary[thisRegion] = 1
        
        inRegion = np.where(inRegionBinary)
        
        #find left hemisphere region for xRange
        maxProj = np.max(inRegionBinary, axis=2).astype(int)                
        cnts,_ = cv2.findContours(maxProj.copy(order='C').astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        leftSide = np.argmin([np.min(u[:, :, 0]) for u in cnts])
        
        xRange = [np.min(cnts[leftSide][:, :, 0]) - padding, np.max(cnts[leftSide][:, :, 0]) + padding]
        yRange = [np.min(inRegion[0])-padding, np.max(inRegion[0])+padding]
        zRange = [np.min(inRegion[2])-padding, np.max(inRegion[2])+padding]
        
        mask = inRegionBinary[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]].astype(np.float)
        mask[mask==0] = np.nan
                
        counts = np.zeros([np.diff(yRange), np.diff(xRange), np.diff(zRange)])
        vol = np.zeros_like(counts)
        for uindex, coords in enumerate(CCFCoords):
            if any(np.isnan(coords)) or vals[uindex] is None or np.isnan(vals[uindex]):
                continue
            else:
                ccf = coords/25
                ccf = ccf.astype(int)
                ccf -= np.array([xRange[0], yRange[0], zRange[0]])
                if vals[uindex] < 1800:
                    counts[ccf[1], ccf[0], ccf[2]]+=1
                    vol[ccf[1], ccf[0], ccf[2]] += vals[uindex]
        
        if weighted is False:
            vol /= counts
            
        vol_s = probeData.gaussianConvolve3D(vol,sigma)
        vol_s *= mask
        
        if weighted:
            counts_s = probeData.gaussianConvolve3D(counts, sigma)        
            counts_s *= mask
            vol_s /= counts_s
        
        minVal = np.nanmin(vol_s)
        maxVal = np.nanmax(vol_s - minVal)
        if rgbVolume:        
            colorMap = np.full(vol_s.shape+(3,),np.nan)
            for y in xrange(vol_s.shape[0]):
                for x in xrange(vol_s.shape[1]):
                    for z in xrange(vol_s.shape[2]):
                        thisVox = (vol_s[y,x,z] - minVal)/maxVal
                        if not np.isnan(thisVox):
                            RGB = cm.jet(thisVox)
                            for i in (0,1,2):
                                colorMap[y, x, z, i] = RGB[i]
                                
            fullMap = np.full(annotationData.shape+(3,),np.nan)
            fullMap[yRange[0]:yRange[1],xRange[0]:xRange[1],zRange[0]:zRange[1]] = colorMap
                
        if rgbVolume:
            return fullMap, vol_s
        else:
            return vol_s
    
    
    def analyzeSTF(self):
        
        inSCAxons = self.getSCAxons()
        
        sf = np.array([0.01,0.02,0.04,0.08,0.16,0.32])
        tf = np.array([0.5,1,2,4,8])
        
        data = self.data.laserOff.run.gratings
        
        hasGratings = data.respMat.notnull()
        
        stfFit = np.stack(data.stfFitParams[hasGratings])
        
        # plot mean resp and f1/f0 matrices
        numUnits = np.count_nonzero(hasGratings)
        respMat = np.full((numUnits,tf.size,sf.size),np.nan)
        f1f0Mat = respMat.copy()
        for uind,u in enumerate(np.where(hasGratings)[0]):
            n = np.zeros(numUnits,dtype=bool)
            n[uind] = True
            i = np.in1d(tf,np.round(data.tf[u],2))
            j = np.in1d(sf,np.round(data.sf[u],2))
            ind = np.ix_(n,i,j)
            resp = data.respMat[u]
            f1f0 = data.f1f0Mat[u]
            bestOriInd = np.unravel_index(np.argmax(resp),resp.shape)[2]
            respMat[ind] = resp[:,:,bestOriInd]
            f1f0Mat[ind] = f1f0[:,:,bestOriInd]
        
        for scInd in (inSCAxons,~inSCAxons):
            ind = scInd[np.where(hasGratings)[0]]
            meanNormRespMat = np.nanmean(respMat[ind]/np.nanmax(np.nanmax(respMat[ind],axis=2),axis=1)[:,None,None],axis=0)
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            plt.imshow(meanNormRespMat,cmap='gray',interpolation='none',origin='lower')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xticklabels(sf)
            ax.set_yticklabels(tf)
            ax.set_xticks(np.arange(sf.size))
            ax.set_yticks(np.arange(tf.size))
            ax.set_xlabel('Cycles/deg',fontsize=20)
            ax.set_ylabel('Cycles/s',fontsize=20)
            plt.tight_layout()
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        plt.imshow(np.nanmean(f1f0Mat,axis=0),cmap='gray',interpolation='none',origin='lower')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xticklabels(sf)
        ax.set_yticklabels(tf)
        ax.set_xticks(np.arange(sf.size))
        ax.set_yticks(np.arange(tf.size))
        ax.set_xlabel('Cycles/deg',fontsize=20)
        ax.set_ylabel('Cycles/s',fontsize=20)
        plt.tight_layout()
        
        # plot center SF and TF and speed tuning index
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.log2(stfFit[:,0]),np.log2(stfFit[:,1]),'ko',markerfacecolor='none',markersize=10,markeredgewidth=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim(np.log2([sf[0]*0.5,sf[-1]*1.5]))
        ax.set_ylim(np.log2([tf[0]*0.5,tf[-1]*1.5]))
        ax.set_xticks(np.log2([sf[0],sf[-1]]))
        ax.set_yticks(np.log2([tf[0],tf[-1]]))
        ax.set_xticklabels([sf[0],sf[-1]])
        ax.set_yticklabels([tf[0],tf[-1]])
        ax.set_xlabel('Cycles/deg',fontsize=20)
        ax.set_ylabel('Cycles/s',fontsize=20)
        plt.tight_layout()
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        tfBins = np.concatenate(([0],tf*1.5))
        tfHist,_ = np.histogram(stfFit[~np.isnan(stfFit[:,1]),1],tfBins)
        ax.bar(np.arange(tf.size)+0.6,tfHist,color='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xticks(np.arange(tf.size)+1)
        ax.set_xticklabels(tf)
        ax.set_xlabel('Center Temporal Frequency',fontsize=20)
        ax.set_ylabel('Number of Units',fontsize=20)
        plt.tight_layout()
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        sfBins = np.concatenate(([0],sf*1.5))
        sfHist,_ = np.histogram(stfFit[~np.isnan(stfFit[:,0]),0],sfBins)
        ax.bar(np.arange(sf.size)+0.6,sfHist,color='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xticks(np.arange(sf.size)+1)
        ax.set_xticklabels(sf)
        ax.set_xlabel('Center Spatial Frequency',fontsize=20)
        ax.set_ylabel('Number of Units',fontsize=20)
        plt.tight_layout()
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        bins = np.arange(-0.3,1.4,0.1)
        ax.hist(stfFit[~np.isnan(stfFit[:,4]),4],bins=bins,color='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xticks([0,0.5,1])
        ax.set_xlabel('Speed Tuning Index',fontsize=20)
        ax.set_ylabel('Number of Units',fontsize=20)
        plt.tight_layout()
        
        # plot preferred speed
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        binInd = np.arange(8)
        bins = 10*2**binInd
        count,_ = np.histogram((stfFit[:,1]/stfFit[:,0])[stfFit[:,4]>0.75],bins)
        ax.bar(binInd[:-1]-0.5,count,width=1,color='k')
        ax.set_xticks(binInd[:-1])
        ax.set_xticklabels(bins[:-1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlabel('Preferred Speed (degrees/s)',fontsize=20)
        ax.set_ylabel('Number of Units',fontsize=20)
        plt.tight_layout()
    
    
    def analyzeOri(self):
        
        ori = np.arange(0,360,45)
        
        data = self.data.laserOff.run.gratings_ori
        
        hasGratings = data.respMat.notnull()
        
        dsi = np.array(data.dsi[hasGratings])
        prefDir = np.array(data.prefDir[hasGratings])
        osi = np.array(data.osi[hasGratings])
        prefOri = np.array(data.prefOri[hasGratings])
        
        # plot dsi and osi
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[1,1],color='0.6')
        ax.plot(osi,dsi,'ko',markerfacecolor='none')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='medium')
        amax = 1.1*max(np.nanmax(dsi),np.nanmax(osi))
        ax.set_xlim([0,amax])
        ax.set_ylim([0,amax])
        ax.set_xlabel('OSI',fontsize='large')
        ax.set_ylabel('DSI',fontsize='large')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        # plot preferred direction
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.hist(prefDir[dsi>0.125],np.arange(0,360,30),color='k')
        ax.set_xlim([-15,345])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='x-large')
        ax.set_xlabel('Preferred Direction',fontsize='xx-large')
        ax.set_ylabel('Number of Units',fontsize='xx-large')
        plt.tight_layout()
    
    
    def analyzeCheckerboard(self):
        
        inSCAxons = self.getSCAxons()
        
        data = self.data.laserOff.allTrials.checkerboard
        
        patchSpeed = bckgndSpeed = np.array([-90,-30,-10,0,10,30,90])
        
        # get data from units with spikes during checkerboard protocol
        # ignore days with laser trials
        hasCheckerboard = (data.respMat.notnull()) & (self.data.laserOn.run.checkerboard.respMat.isnull()) 
        respMat = np.stack(data.respMat[hasCheckerboard])
        hasSpikes = respMat.any(axis=2).any(axis=1)
        respMat = respMat[hasSpikes]
        uindex = np.where(hasCheckerboard)[0][hasSpikes]
        
        # get z score and determine significant responses
        spontRateMean = data.spontRateMean[uindex]
        spontRateStd = data.spontRateStd[uindex]
        respZ = (respMat-spontRateMean[:,None,None])/spontRateStd[:,None,None]
        hasResp = (respZ>10).any(axis=2).any(axis=1)
        respMat = respMat[hasResp]
        uindex = uindex[hasResp]
        
        # fill in NaNs where no running trials
        statRespMat = np.stack(self.data.laserOff.stat.checkerboard.respMat[uindex])
        y,x,z = np.where(np.isnan(respMat))
        for i,j,k in zip(y,x,z):
            respMat[i,j,k] = statRespMat[i,j,k]
       
# find distance between RF and patch
#        onVsOff = np.array(self.data.sparseNoise.onVsOff[uindex])
#        onFit = np.stack(self.data.sparseNoise.onFit[uindex])
#        offFit = np.stack(self.data.sparseNoise.offFit[uindex])
#        rfElev = offFit[:,1].copy()
#        rfElev[onVsOff>0] = onFit[onVsOff>0,1]
#        patchDist = np.full(uindex.size,np.nan)
#        for i,u in enumerate(uindex):
#            patchDist[i] = np.min(np.absolute(self.data.checkerboard.patchElevation[u]-rfElev[i]))
#        fig = plt.figure(facecolor='w')
#        ax = fig.add_subplot(1,1,1)
#        ax.plot(patchDist,respMat[:,patchSpeed!=0,bckgndSpeed==0].max(axis=1),'ko')
#        ax.set_xlabel('Patch RF Distance')
#        ax.set_ylabel('Max Patch Response')
        
        # compare patch and bckgnd responses and contribution to variance
        maxPatchResp = respMat[:,patchSpeed!=0,bckgndSpeed==0].max(axis=1)
        maxBckgndResp = respMat[:,patchSpeed==0,bckgndSpeed!=0].max(axis=1)
        patchIndex = (maxPatchResp-maxBckgndResp)/(maxPatchResp+maxBckgndResp)
        
        patchVar = respMat.mean(axis=2).std(axis=1)
        bckgndVar = respMat.mean(axis=1).std(axis=1)
        varIndex = (patchVar-bckgndVar)/(patchVar+bckgndVar)
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot([-1,1],[0,0],color='0.6')
        ax.plot([0,0],[-1,1],color='0.6')
        ax.plot(patchIndex,varIndex,'ko')
        ax.set_xlim((-0.75,0.75))
        ax.set_ylim((-0.75,0.75))
        ax.set_xlabel('Patch vs Background Response')
        ax.set_ylabel('Patch vs Background Speed Variance')
        
        patchIndexSC = patchIndex[inSCAxons[uindex]]
        patchIndexNonSC = patchIndex[~inSCAxons[uindex]]
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        cumProbPatchIndexSC = [np.count_nonzero(patchIndexSC<=i)/patchIndexSC.size for i in np.sort(patchIndexSC)]
        cumProbPatchIndexNonSC = [np.count_nonzero(patchIndexNonSC<=i)/patchIndexNonSC.size for i in np.sort(patchIndexNonSC)]
        ax.plot([0,0],[0,1],'k--')
        ax.plot(np.sort(patchIndexNonSC),cumProbPatchIndexNonSC,'0.6',linewidth=3)
        ax.plot(np.sort(patchIndexSC),cumProbPatchIndexSC,'k',linewidth=3)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([-0.8,0.8])
        ax.set_ylim([0,1.01])
        ax.set_xticks([-0.5,0,0.5])
        ax.set_yticks([0,0.5,1])
        ax.set_xlabel('Patch-Background Index',fontsize=20)
        ax.set_ylabel('Cumulative Probability',fontsize=20)
        plt.tight_layout()
        
        # correlate responses with response-type templates
#        patchTemplate = np.zeros((patchSpeed.size,bckgndSpeed.size))
#        patchTemplate[patchSpeed!=0,bckgndSpeed==0] = 1
#        bckgndRightTemplate = np.zeros((patchSpeed.size,bckgndSpeed.size))
#        bckgndRightTemplate[:,bckgndSpeed>0] = 1
#        bckgndLeftTemplate = bckgndRightTemplate[:,::-1]
#        frameTemplate = np.zeros((patchSpeed.size,bckgndSpeed.size))
#        frameTemplate[:,[0,-1]] = 1
#        frameTemplate[[0,-1],:] = 1
#        
#        templates = (patchTemplate,frameTemplate,bckgndRightTemplate,bckgndLeftTemplate)
#        cc = np.zeros((respMat.shape[0],len(templates)))
#        for i,resp in enumerate(respMat):
#            for j,template in enumerate(templates):
#                cc[i,j] = np.corrcoef(resp.ravel(),template.ravel())[0,1]
#        cc[:,2] = cc[:,2:].max(axis=1)
#        cc = cc[:,:-1]
#        
#        for i in (0,1,2):
#            plt.figure()
#            plt.hist(cc[:,i],np.arange(-1,1,0.1))
#        
#        patchSortInd = np.argsort(cc[:,0])[::-1]
#        plt.imshow(cc[patchSortInd],clim=(-1,1),cmap='bwr',interpolation='none')
#        
#        bestMatch = np.argmax(cc,axis=1)
#        aboveThresh = cc.max(axis=1)>0.4
#        typeCount = [np.count_nonzero(bestMatch[aboveThresh]==i) for i in range(3)]
#        print(typeCount)
#        
#        for n in range(10):
#            c = 2
#            if n<typeCount[c]:
#                ind = np.logical_and(bestMatch==c,aboveThresh)
#                fig = plt.figure()
#                ax = plt.subplot(1,1,1)
#                ax.imshow(respMat[ind][n],origin='lower',cmap='gray',interpolation='none')
#                ax.set_title(str(round(cc[ind,c][n],2)))
        
        # normalize responses various ways
        maxResp = respMat.max(axis=2).max(axis=1)
        minResp = respMat.min(axis=2).min(axis=1) 
        respMatNorm = respMat/maxResp[:,None,None]
        respMatBaseSub = respMat-respMat[:,patchSpeed==0,bckgndSpeed==0][:,:,None]
        respMatBaseSubNorm = respMatBaseSub/(maxResp-minResp)[:,None,None]
        
        fig = plt.figure(facecolor='w')
        gs = gridspec.GridSpec(3,4)
        for j,(r,title) in enumerate(zip((respMat,respMatNorm,respMatBaseSub,respMatBaseSubNorm),('resp','norm','base sub','base sub norm'))):
            ax = fig.add_subplot(gs[0,j])
            rmean = r.mean(axis=0)
            ax.imshow(rmean,cmap='gray',interpolation='none',origin='lower')
            ax.set_title(title)
            
            ax = fig.add_subplot(gs[1,j])
            rstd = r.std(axis=0)
            ax.imshow(rstd,cmap='gray',interpolation='none',origin='lower')
            ax.set_title(title)
            
            ax = fig.add_subplot(gs[2,j])
            plt.hist(r.ravel())
        
        # plot pc weightings
        for r in (respMat, respMatNorm, respMatBaseSub, respMatBaseSubNorm):
            r = r.reshape((r.shape[0],r[0].size))
            pcaData,eigVal,eigVec = clust.pca(r)
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            ax.plot(np.cumsum(eigVal/eigVal.sum()))
            fig = plt.figure()
            for i in range(9):
                ax = fig.add_subplot(3,3,i+1)
                cLim = max(abs(eigVec[:,i]))
                ax.imshow(eigVec[:,i].reshape(respMat.shape[1:]),clim=(-cLim,cLim),cmap='bwr',interpolation='none',origin='lower')
         
        for i in range(6):
            print(eigVal[i]/eigVal.sum())
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            cLim = max(abs(eigVec[:,i]))
            ax.imshow(eigVec[:,i].reshape(respMat.shape[1:]),clim=(-cLim,cLim),cmap='bwr',interpolation='none',origin='lower')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xticks(np.arange(bckgndSpeed.size))
            ax.set_xticklabels(bckgndSpeed,fontsize=18)
            ax.set_yticks(np.arange(patchSpeed.size))
            ax.set_yticklabels(patchSpeed,fontsize=18)
            ax.set_xlabel('Background Speed (deg/s)',fontsize=20)
            ax.set_ylabel('Patch Speed (deg/s)',fontsize=20)
            plt.tight_layout()
                
        # clusters using all dimensions
        for r in (respMat, respMatNorm, respMatBaseSub, respMatBaseSubNorm):
            r = r.reshape((r.shape[0],r[0].size))
            k = 3
            clustID,_ = clust.ward(r,nClusters=k,plotDendrogram=True)
            fig = plt.figure(facecolor='w')
            for i in np.unique(clustID):
                ax = fig.add_subplot(round(k**0.5),math.ceil(k**0.5),i)
                ax.imshow(respMat[clustID==i].mean(axis=0),cmap='bwr',interpolation='none',origin='lower')
                
        # cluster using principal components
        for r in (respMat, respMatNorm, respMatBaseSub, respMatBaseSubNorm):
            r = r.reshape((r.shape[0],r[0].size))
            pcaData,_,_ = clust.pca(r)
            k = 6
            clustID,_ = clust.ward(r[:,:6],nClusters=k,plotDendrogram=True)
            fig = plt.figure(facecolor='w')
            for i in np.unique(clustID):
                ax = fig.add_subplot(round(k**0.5),math.ceil(k**0.5),i)
                ax.imshow(respMat[clustID==i].mean(axis=0),cmap='bwr',interpolation='none',origin='lower')
            
        # cluster using nested PCA
        nSplit = 3
        for r in (respMat, respMatNorm, respMatBaseSub, respMatBaseSubNorm):
            r = r.reshape((r.shape[0],r[0].size))
            clustIDHier,linkageMat = clust.nestedPCAClust(r,nSplit=nSplit,minClustSize=2,varExplained=0.75,clustID=[],linkageMat=[])
            clustID = clust.getClustersFromHierarchy(clustIDHier)
            k = len(set(clustID))
            fig = plt.figure(facecolor='w')
            for ind,i in enumerate(np.unique(clustID)):
                ax = fig.add_subplot(round(k**0.5),math.ceil(k**0.5),ind+1)
                ax.imshow(respMat[clustID==i].mean(axis=0),cmap='bwr',interpolation='none',origin='lower')
        
        idNum = np.unique(clustID)
        
        inCluster = np.in1d(clustID,idNum[:4])
        print(np.count_nonzero(np.logical_and(inSCAxons,inCluster)))
        print(np.count_nonzero(np.logical_and(~inSCAxons,inCluster)))
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.imshow(respMat[np.in1d(clustID,idNum[:4])].mean(axis=0),cmap='bwr',interpolation='none',origin='lower')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xticks(np.arange(bckgndSpeed.size))
        ax.set_xticklabels(bckgndSpeed,fontsize=18)
        ax.set_yticks(np.arange(patchSpeed.size))
        ax.set_yticklabels(patchSpeed,fontsize=18)
        ax.set_xlabel('Background Speed (deg/s)',fontsize=20)
        ax.set_ylabel('Patch Speed (deg/s)',fontsize=20)
        plt.tight_layout()
                
            
    def plotSaccadeRate(self):
        protocolLabels = ('spontaneous','sparseNoise','gratings','checkerboard')
        saccadeRate = np.full((len(self.experimentFiles),len(protocolLabels)),np.nan)
        for i,exp in enumerate(self.experimentFiles):
            print('analyzing experiment '+str(i+1)+' of '+str(len(self.experimentFiles)))
            p = self.getProbeDataObj(exp)
            if hasattr(p,'behaviorData'):
                for j,label in enumerate(protocolLabels):
                    protocol = p.getProtocolIndex(label)
                    if protocol is not None and 'eyeTracking' in p.behaviorData[str(protocol)]:
                        eyeTrackSamples = p.behaviorData[str(protocol)]['eyeTracking']['samples']
                        pupilX = p.behaviorData[str(protocol)]['eyeTracking']['pupilX']
                        negSaccades = p.behaviorData[str(protocol)]['eyeTracking']['negSaccades']
                        negSaccades = negSaccades[negSaccades<eyeTrackSamples.size]
                        negSaccades = negSaccades[~np.isnan(pupilX[negSaccades])]
                        posSaccades = p.behaviorData[str(protocol)]['eyeTracking']['posSaccades']
                        posSaccades = posSaccades[posSaccades<eyeTrackSamples.size]
                        posSaccades = posSaccades[~np.isnan(pupilX[posSaccades])]
                        saccadeRate[i,j] = (negSaccades.size+posSaccades.size)/(eyeTrackSamples[-1]-eyeTrackSamples[0])*p.sampleRate
        mean = np.nanmean(saccadeRate,axis=0)
        sem = np.nanstd(saccadeRate,axis=0)/np.sqrt(np.nansum(~np.isnan(saccadeRate),axis=0))
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.bar(np.arange(mean.size)+0.1,mean,width=0.8,color='k',yerr=sem,error_kw={'ecolor':'k'})
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=16)
        ax.set_xticks(np.arange(mean.size)+0.5)
        ax.set_xticklabels(('Dark','Sparse Noise','Gratings','Checkerboard'),fontsize=18)
        ax.set_ylim((0,0.3))
        ax.set_yticks(np.arange(0,0.4,0.1))
        ax.set_ylabel('Saccades/s',fontsize=18)
        plt.tight_layout()
    
           
    def analyzeSaccades(self,protocol='sparseNoise'):
        # get data from all experiments
        protocol = str(protocol)
        pupilX = []
        saccadeAmp = []
        preSaccadeSpikeCount = []
        postSaccadeSpikeCount = []
        hasResp = []
        respPolarity = []
        latency = []
        analysisWindow = [0,0.2]
        winDur = analysisWindow[1]-analysisWindow[0]
        for i,exp in enumerate(self.experimentFiles):
            print('analyzing experiment '+str(i+1)+' of '+str(len(self.experimentFiles)))
            p = self.getProbeDataObj(exp)
            units = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            units, _ = p.getOrderedUnits(units)
            pIndex = p.getProtocolIndex(protocol)
            saccadeData = p.analyzeSaccades(units,pIndex,analysisWindow=analysisWindow,plot=False)
    
            if saccadeData is not None:
                pupilX.append(saccadeData['pupilX'])
                saccadeAmp.append(np.concatenate((saccadeData['negAmp'],saccadeData['posAmp'])))
                preSaccadeSpikeCount.append(saccadeData['preSaccadeSpikeCount'])
                postSaccadeSpikeCount.append(saccadeData['postSaccadeSpikeCount'])
                hasResp.append(saccadeData['hasResp'])
                respPolarity.append(saccadeData['respPolarity'])
                latency.append(saccadeData['latency'])
            else:
                for param in [pupilX,saccadeAmp,preSaccadeSpikeCount, postSaccadeSpikeCount, hasResp, respPolarity, latency]:
                    param.append(None)
            
        # pupil position histogram
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        for x in pupilX:
            if x is not None:
                x = x-np.nanmedian(x)
                count,bins = np.histogram(x[~np.isnan(x)],bins=np.arange(np.nanmin(x)-1,np.nanmax(x)+2))
                ax.plot(bins[1:]-0.5,count/count.sum(),'k')
        ax.plot([-5,-5],[0,1],'k--')
        ax.plot([5,5],[0,1],'k--')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([-25,25])
        ax.set_ylim([0,0.3])
        ax.set_xticks(np.arange(-20,30,10))
        ax.set_yticks(np.arange(0,0.4,0.1))
        ax.set_xlabel('Relative Pupil Position (degrees)',fontsize=20)
        ax.set_ylabel('Probability',fontsize=20)
        plt.tight_layout()
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        for x in pupilX:
            if x is not None:
                x = np.absolute(x-np.nanmedian(x))
                count,bins = np.histogram(x[~np.isnan(x)],bins=np.arange(np.nanmin(x)-1,np.nanmax(x)+2))
                ax.plot(bins[1:]-0.5,np.cumsum(count)/count.sum(),'k')
        ax.plot([5,5],[0,1],'k--')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([0,25])
        ax.set_ylim([0,1])
        ax.set_xticks(np.arange(0,30,10))
        ax.set_yticks([0,0.5,1])
        ax.set_xlabel('abs(Relative Pupil Position) (degrees)',fontsize=20)
        ax.set_ylabel('Cumulative Probability',fontsize=20)
        plt.tight_layout()
        
        # saccade amplitude histogram
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        for amp in saccadeAmp:
            if amp is not None and amp.size>0:
                count,bins = np.histogram(amp[~np.isnan(amp)],bins=np.arange(np.nanmin(amp)-1,np.nanmax(amp)+2))
                ax.plot(bins[1:]-0.5,count/count.sum(),'k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlabel('Saccade Amplitude (degrees)',fontsize=20)
        ax.set_ylabel('Probability',fontsize=20)
        plt.tight_layout()
        
        # pre and post saccade firing rate
        totalCount = excitCount = inhibCount = bothCount = 0
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        maxRate = 0
        for preSpikes,postSpikes,isResp,pol in zip(preSaccadeSpikeCount,postSaccadeSpikeCount,hasResp,respPolarity):
            if preSpikes is not None:
                for u,_ in enumerate(preSpikes):
                    for saccadeDirInd,clr in zip((0,1),('r','b')):
                        preRate = np.mean(preSpikes[u][saccadeDirInd])/winDur
                        postRate = np.mean(postSpikes[u][saccadeDirInd])/winDur
                        mfc = clr if isResp[u,saccadeDirInd] else 'none'
                        ax.plot(preRate,postRate,'o',mec=clr,mfc=mfc,alpha=0.5)
                        maxRate = max(maxRate,postRate)
                totalCount += isResp.shape[0]
                isExcit = np.logical_and(isResp[:,:2],pol[:,:2]>0).any(axis=1)
                isInhib = np.logical_and(isResp[:,:2],pol[:,:2]<0).any(axis=1)
                isBoth = np.logical_and(isExcit,isInhib)
                excitCount += np.logical_xor(isExcit,isBoth).sum()
                inhibCount += np.logical_xor(isInhib,isBoth).sum()
                bothCount += isBoth.sum()
        maxRate *= 1.05
        ax.plot([0,maxRate],[0,maxRate],'k--')
        ax.set_xlim([0,75])
        ax.set_ylim([0,75])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlabel('Pre-saccade spikes/s',fontsize=20)
        ax.set_ylabel('Post-saccade spikes/s',fontsize=20)
        ax.set_aspect('equal')
        plt.tight_layout()
        
        print('total = '+str(totalCount))
        print('excited only = '+str(excitCount))
        print('inhibited only = '+str(inhibCount))
        print('both = '+str(bothCount))
        
        # latency
        excitLat = []
        inhibLat = []
        for lat,isResp,pol in zip(latency,hasResp,respPolarity):
            if lat is not None:
                excitLat = np.concatenate((excitLat,lat[pol[:,-1]>0,-1]*1000))
                inhibLat = np.concatenate((inhibLat,lat[pol[:,-1]<0,-1]*1000))
        for lat,label in zip((excitLat,inhibLat),('Excited','Inhibited')):
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            binSize = 10
            ax.hist(lat[~np.isnan(lat)],bins=np.arange(np.nanmin(lat)-binSize,np.nanmax(lat)+2*binSize,binSize),color='k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xlabel('Latency (ms)',fontsize=20)
            ax.set_ylabel('# Units',fontsize=20)
            ax.set_title(label,fontsize=20)
            plt.tight_layout()
        
        # saccade modulation index
        postSac = []
        [postSac.extend(ppss) for ppss in postSaccadeSpikeCount]
        
        preSac = []
        [preSac.extend(ppss) for ppss in preSaccadeSpikeCount]
        
        meanPre = []
        meanPost = []
        for prs,pts in zip(preSac, postSac):
            if prs is not None:    
                meanPre.append(np.mean(prs[2]))
                meanPost.append(np.mean(pts[2]))
            else:
                meanPre.append(np.nan)
                meanPost.append(np.nan)
        
        meanPre = np.array(meanPre)
        meanPost = np.array(meanPost)
        
        sacModIndex = (meanPost - meanPre)/(meanPre + meanPost)        
        
        
    def analyzeOKR(self,protocolName='gratings'):
        # get data from all experiments
        if protocolName=='gratings':
            xparam = np.array([0.01,0.02,0.04,0.08,0.16,0.32])
            yparam = np.array([0.5,1,2,4,8])
            xparamName = 'Cycles/deg'
            yparamName = 'Cycles/s'
        else:
            xparam = yparam = np.array([-90,-30,-10,0,10,30,90]) 
            xparamName = 'Background Speed (deg/s)'
            yparamName = 'Patch Speed (deg/s)'
        meanPupilVel = np.full((len(self.experimentFiles),yparam.size,xparam.size),np.nan)
        okrGain = meanPupilVel.copy()
        for expInd,exp in enumerate(self.experimentFiles):
            print('analyzing experiment '+str(expInd+1)+' of '+str(len(self.experimentFiles)))
            p = self.getProbeDataObj(exp)
            okrData = p.analyzeOKR(protocolName,plot=False)
            if okrData is not None:
                n = np.zeros(len(self.experimentFiles),dtype=bool)
                n[expInd] = True
                i = np.in1d(yparam,np.round(okrData['yparam'],2))
                j = np.in1d(xparam,np.round(okrData['xparam'],2))
                ind = np.ix_(n,i,j)
                meanPupilVel[ind] = okrData['meanPupilVel']
                okrGain[ind] = okrData['okrGain']
        meanPupilVel = np.nanmean(meanPupilVel,axis=0)
        okrGain = np.nanmean(okrGain,axis=0)
        if protocolName=='gratings':
            stimSpeed = yparam[:,None]/xparam
            stimSpeedLabel = 'Grating Speed (deg/s)'
        else:
            stimSpeed = np.tile(xparam,(xparam.size,1))
            stimSpeedLabel = 'Background Speed (deg/s)'
                
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        maxVel = np.absolute(meanPupilVel).max()
        if protocolName=='gratings':
            clim = (0,maxVel)
            cmap = 'gray'
        else:
            clim = (-maxVel,maxVel)
            cmap = 'bwr'
        im = ax.imshow(meanPupilVel,clim=clim,cmap=cmap,origin='lower',interpolation='none')
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        if protocolName=='gratings':
            ax.set_xticks([0,xparam.size-1])
            ax.set_yticks([0,yparam.size-1])
            ax.set_xticklabels(xparam[[0,-1]])
            ax.set_yticklabels(yparam[[0,-1]])
        else:
            ax.set_xticks(np.arange(xparam.size))
            ax.set_yticks(np.arange(yparam.size))
            ax.set_xticklabels(xparam)
            ax.set_yticklabels(yparam)
        ax.set_xlabel(xparamName,fontsize=20)
        ax.set_ylabel(yparamName,fontsize=20)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(length=0,labelsize=18)
        cb.set_ticks(np.round(clim))
        plt.tight_layout()
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(okrGain,clim=[0,okrGain.max()],cmap='gray',origin='lower',interpolation='none')
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        if protocolName=='gratings':
            ax.set_xticks([0,xparam.size-1])
            ax.set_yticks([0,yparam.size-1])
            ax.set_xticklabels(xparam[[0,-1]])
            ax.set_yticklabels(yparam[[0,-1]])
        else:
            ax.set_xticks(np.arange(xparam.size))
            ax.set_yticks(np.arange(yparam.size))
            ax.set_xticklabels(xparam)
            ax.set_yticklabels(yparam)
        ax.set_xlabel(xparamName,fontsize=20)
        ax.set_ylabel(yparamName,fontsize=20)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(length=0,labelsize=18)
        cb.set_ticks([0,math.floor(round(okrGain.max(),2)*100)/100])
        plt.tight_layout()
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        if protocolName=='gratings':
            ax.semilogx(stimSpeed.ravel(),okrGain.ravel(),'ko',markersize=10)
            ax.set_xlim([0.9,1000])
        else:
            ax.plot(stimSpeed.ravel(),okrGain.ravel(),'ko',markersize=10)
            ax.set_xlim([-100,100])
        for side in ('top','right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(which='both',direction='out',top=False,right=False,labelsize=18)
        ax.set_ylim([0,1.05])
        ax.set_yticks([0,0.5,1])
        ax.set_xlabel(stimSpeedLabel,fontsize=20)
        ax.set_ylabel('OKR Gain',fontsize=20)
        plt.tight_layout()
        
        
    def analyzeRunningGratings(self):
        sR = []
        rR = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for exp in self.experimentFiles:
            p = self.getProbeDataObj(exp)
            units = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            units, _ = p.getOrderedUnits(units)
            pind = p.getProtocolIndex('gratings')
            if 'gratings_stf_laserOff_run' in p.units[units[0]] and 'gratings_stf_laserOff_stat' in p.units[units[0]]:
                v = p.visstimData[str(pind)]
                
                rTrials = p.units[units[0]]['gratings_stf_laserOff_run']['trials']
                r_sf = v['stimulusHistory_sf'][rTrials]
                r_tf = v['stimulusHistory_tf'][rTrials]
                r_ori = v['stimulusHistory_ori'][rTrials]
                
                sTrials = p.units[units[0]]['gratings_stf_laserOff_stat']['trials']
                s_sf = v['stimulusHistory_sf'][sTrials]
                s_tf = v['stimulusHistory_tf'][sTrials]
                s_ori = v['stimulusHistory_ori'][sTrials]
                
                #match trial number and identity
                if np.min([sTrials.size, rTrials.size]) > 5:
                    sTrialMatch = []
                    rTrialMatch = []
                    for sf in np.unique(v['stimulusHistory_sf']):
                        for tf in np.unique(v['stimulusHistory_tf']):
                            for ori in np.unique(v['stimulusHistory_ori']):
                                sMatchedTrialInds = np.where(np.logical_and(np.logical_and(s_sf==sf, s_tf==tf), s_ori==ori))[0]
                                rMatchedTrialInds = np.where(np.logical_and(np.logical_and(r_sf==sf, r_tf==tf), r_ori==ori))[0]
                                
                                trialCount = np.min([sMatchedTrialInds.size, rMatchedTrialInds.size])
                                
                                if trialCount > 0:
                                    sMatchedTrials = sTrials[sMatchedTrialInds]
                                    rMatchedTrials = rTrials[rMatchedTrialInds]
                                    
                                    sTrialMatch.extend(np.random.choice(sMatchedTrials, trialCount, replace=False))
                                    rTrialMatch.extend(np.random.choice(rMatchedTrials, trialCount, replace=False))
                                                                           
                    sTrialMatch = np.array(sTrialMatch)                   
                    rTrialMatch = np.array(rTrialMatch)
                             
                        
                    trialStarts, trialEnds = p.getTrialStartsEnds('gratings')
                    for u in units:
                        spikes = p.units[u]['times'][str(pind)]
                        sResponse = p.findSpikesPerTrial(trialStarts[sTrialMatch], trialEnds[sTrialMatch], spikes)
                        rResponse = p.findSpikesPerTrial(trialStarts[rTrialMatch], trialEnds[rTrialMatch], spikes)
                        sR.append(np.mean(sResponse))
                        rR.append(np.mean(rResponse))
                else:
                    sR.extend([np.nan]*len(units))
                    rR.extend([np.nan]*len(units))
            
            
            else:
                sR.extend([np.nan]*len(units))
                rR.extend([np.nan]*len(units))
                    
                    
        ax.plot(sR, rR, 'ko', alpha=0.5)
        maxR = np.nanmax([np.nanmax(sR), np.nanmax(rR)])
        ax.plot([0, maxR], [0, maxR], 'r--')            
        
        #Find running modulation index
        sR = np.array(sR)
        rR = np.array(rR)
        runMod = (rR - sR)/(rR+sR)
    
    def trialMatch(self, pObj, protocol, paramsToMatch, cond1Trials, cond2Trials):
        
        pind = pObj.getProtocolIndex(protocol)
        v = pObj.visstimData[str(pind)]
        paramList = [np.unique(v[pm]) for pm in paramsToMatch]
        paramCombos = list(itertools.product(*paramList))
        
        c1ParamList = np.array([[v[pm][t] for pm in paramsToMatch] for t in cond1Trials])
        c2ParamList = np.array([[v[pm][t] for pm in paramsToMatch] for t in cond2Trials])
        
        c1TrialMatch = []
        c2TrialMatch = []
        for combo in paramCombos:
            c1MatchedTrialInds = [it for it, t in enumerate(c1ParamList) if all(t==combo)]
            c2MatchedTrialInds = [it for it, t in enumerate(c2ParamList) if all(t==combo)]
            
            trialCount = np.min([len(c1MatchedTrialInds), len(c2MatchedTrialInds)])
            if trialCount > 0:
                c1MatchedTrials = cond1Trials[c1MatchedTrialInds]
                c2MatchedTrials = cond2Trials[c2MatchedTrialInds]
                
                c1TrialMatch.extend(np.random.choice(c1MatchedTrials, trialCount, replace=False))
                c2TrialMatch.extend(np.random.choice(c2MatchedTrials, trialCount, replace=False))
        
        return c1TrialMatch, c2TrialMatch
        
    def analyzeRunningSpontaneous(self):
        sR = []
        rR = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for exp in self.experimentFiles:
            p = self.getProbeDataObj(exp)
            units = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            units, _ = p.getOrderedUnits(units)
            pind = p.getProtocolIndex('spontaneous')
            
            if pind is not None:
                nsamps = p.behaviorData[str(pind)]['running'].size * 500
                trialStarts = np.arange(0, nsamps, 2*p.sampleRate)
                trialEnds = trialStarts + 2*p.sampleRate
                
                sTrials, rTrials, _ = p.parseRunning(pind, trialStarts=trialStarts, trialEnds=trialEnds)
                trialNum = np.min([len(sTrials), len(rTrials)])
                if trialNum > 3:
                    sTrialMatch = np.random.choice(sTrials, trialNum, replace=False)
                    rTrialMatch = np.random.choice(rTrials, trialNum, replace=False)
                
                    for u in units:
                        spikes = p.units[u]['times'][str(pind)]
                        sResponse = p.findSpikesPerTrial(trialStarts[sTrialMatch], trialEnds[sTrialMatch], spikes)
                        rResponse = p.findSpikesPerTrial(trialStarts[rTrialMatch], trialEnds[rTrialMatch], spikes)
                        sR.append(np.mean(sResponse))
                        rR.append(np.mean(rResponse))
                else:
                     sR.extend([np.nan]*len(units))
                     rR.extend([np.nan]*len(units))  
             
            else:
                 sR.extend([np.nan]*len(units))
                 rR.extend([np.nan]*len(units))   
        
        ax.plot(sR, rR, 'ko', alpha=0.5)
        maxR = np.nanmax([np.nanmax(sR), np.nanmax(rR)])
        ax.plot([0, maxR], [0, maxR], 'r--')           

    def runningHistograms(self):
         for exp in self.experimentFiles:
            p = self.getProbeDataObj(exp)
            expD, expID = p.getExperimentInfo()
            fig = plt.figure('Running Distribution for '+expD+'_'+expID, facecolor='w', figsize=[ 13.725 ,   8.8625])
            gs = gridspec.GridSpec(2,int(np.ceil(len(p.kwdFileList)/2)))
            for pro, _ in enumerate(p.kwdFileList):  
                if 'running' in p.behaviorData[str(pro)]:
                    wheelData = -p.behaviorData[str(pro)]['running']
                    h, b = np.histogram(wheelData, np.arange(100), normed=True)                    
                    
                    ax = fig.add_subplot(gs[int(pro%2), int(np.floor(pro/2))])
                    ax.bar(b[:-1], h, color='k')
                    ax.set_title(p.getProtocolLabel(pro))
            
            fig.set_tight_layout(True)
            
    def runningTuningCurves(self, protocol = 'checkerboard'):
        runningTuningCurves = []        
        for expInd, exp in enumerate(self.experimentFiles):
            print('Analyzing ' + str(expInd) + ' of ' + str(len(self.experimentFiles)) + ' experiments')
            p = self.getProbeDataObj(exp)
            units = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            units, _ = p.getOrderedUnits(units)
            expD, expID = p.getExperimentInfo()   
            pIndex = p.getProtocolIndex(protocol)
            if str(pIndex) in p.behaviorData and 'running' in p.behaviorData[str(pIndex)]:
                tuningCurves = p.analyzeRunning(pIndex, units=units, plot=False)
                runningTuningCurves.extend(tuningCurves)
            else:
                runningTuningCurves.extend([np.nan]*len(units))
            
        runModIndex = []
        peakSpeed = []
        for c in runningTuningCurves:
            if type(c) is float or any(np.isnan(c)):
                runModIndex.append(np.nan)
                peakSpeed.append(np.nan)
            else:
                peakInd = np.argmax(c)
                peakSpeed.append(peakInd)
                
                maxresp = np.max(c)
                minresp = np.min(c)
                
                runModIndex.append((maxresp - minresp)/(maxresp+minresp))
                
                
    def analyzeWaveforms(self, plot=True):
        
        templates = []
        for i,exp in enumerate(self.experimentFiles):
            print('analyzing experiment '+str(i+1)+' of '+str(len(self.experimentFiles)))
            p = self.getProbeDataObj(exp)
            units = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            units, _ = p.getOrderedUnits(units)
            temps = [p.units[u]['template'] for u in units]
            templates.extend(temps)
        
        tempArray = np.array(templates)
        peakToTrough = findPeakToTrough(tempArray, plot=plot)
        
        return peakToTrough

def findPeakToTrough(waveformArray, sampleRate=30000, plot=True):
    #waveform array should be units x samples x channels
    maxChan = [np.unravel_index(np.argmin(t), t.shape)[1] for t in waveformArray]
    tempArray = [waveformArray[i, :, maxChan[i]] for i in xrange(waveformArray.shape[0])]
    
    peakToTrough = np.zeros(len(tempArray))       
    for i,t in enumerate(tempArray):
        peakInd = np.argmax(np.absolute(t))
        peakToTrough[i] = (np.argmin(t[peakInd:]) if t[peakInd]>0 else np.argmax(t[peakInd:]))/(sampleRate/1000.0)
     
    if plot:
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.hist(peakToTrough,np.arange(0,1.2,0.05),color='k')
        ax.plot([0.35]*2,[0,180],'k--')
        for side in ('top','right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(which='both',direction='out',top=False,right=False,labelsize=18)
        ax.set_xlabel('Spike peak-to-trough (ms)',fontsize=20)
        ax.set_ylabel('# Units',fontsize=20)
        plt.tight_layout()
     
    return peakToTrough
     
if __name__=="__main__":
    pass