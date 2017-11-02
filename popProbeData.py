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
from xml.dom import minidom
import scipy.stats
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
from matplotlib import cm
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


class popProbeData():
    
    def __init__(self):
        self.experimentFiles = None
        self.excelFile = None
        self.data = None
        self.annotationData = None
        self.annotationStructures = None
        self.inSCAxonsVol = None
    
    
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
        
    
    def getExcelFile(self):
        filePath = fileIO.getFile(caption='Choose excel file with unit labels',fileType='*.xlsx')
        if filePath=='':
            return
        self.excelFile = filePath
    
    
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
            p.runAllAnalyses(protocolsToRun=protocols,splitRunning=True,plot=False)
            if save:
                p.saveHDF5(exp)
        
            
    def getProbeDataObj(self,experimentFilePath):
        p = probeData.probeData()
        p.loadHDF5(experimentFilePath)
        return p
    
        
    def getUnitLabels(self,probeDataObj=None,save=False):
        if self.excelFile is None:
            self.getExcelFile()
        if probeDataObj is None:
            for exp in self.experimentFiles:
                p = self.getProbeDataObj(exp)
                p.readExcelFile(fileName=self.excelFile)
                if save:
                    p.saveHDF5(exp)
        else:
            probeDataObj.readExcelFile(fileName=self.excelFile)
    
    
    def makeDataFrame(self,analyzeExperiments=False,findRegion=True):
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
        rowNames = ('experimentDate','animalID','genotype','unitID','unitLabel','ccfX','ccfY','ccfZ', 'region','photoTag')
        experimentDate = []
        animalID = []
        genotype = []
        unitID = []
        unitLabel = []
        ccfX = []
        ccfY = []
        ccfZ = []
        region = []
        photoTag = []
        
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
        data['laserOff']['allTrials']['waveform']= {'waveform':[]}
        for exp in exps:
            p = self.getProbeDataObj(exp)
            expDate,anmID = p.getExperimentInfo()
            units,ypos = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            for u in units:
                experimentDate.append(expDate)
                animalID.append(anmID)
                if hasattr(p,'genotype'):
                    genotype.append(p.genotype)
                else:
                    genotype.append('')
                unitID.append(u)
                unitLabel.append(p.units[u]['label'])
                if 'laserResp' in p.units[u].keys():
                    photoTag.append(p.units[u]['laserResp'])
                else:
                    photoTag.append(False)
                for i,c in enumerate((ccfX,ccfY,ccfZ)):
                    c.append(p.units[u]['CCFCoords'][i])
                
                if findRegion:
                    region.append(self.findRegions(p.units[u]['CCFCoords']))
                else:
                    region.append(np.nan)
                
                if 'waveform' in p.units[u]:
                    w = p.units[u]['waveform']
                    maxChan = np.unravel_index(np.argmin(w), w.shape)[1]
                    w = w[:, maxChan]
                else:
                    w = np.full(82, np.nan)
                data['laserOff']['allTrials']['waveform']['waveform'].append(w)
                
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
        rows = pd.MultiIndex.from_arrays([experimentDate,animalID,genotype,unitID,unitLabel,ccfX,ccfY,ccfZ,region,photoTag],names=rowNames)
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
        
    
    def getCCFCoords(self, cells=None):
        if cells is None:
            y,x,z = [np.array(self.data.index.get_level_values('ccf'+ax)) for ax in ('Y','X','Z')]    
        else:
            data = self.data[cells]
            y,x,z = [np.array(data.index.get_level_values('ccf'+ax)) for ax in ('Y','X','Z')]
        return y,x,z
    
    
    def getSCAxons(self):
        y,x,z = self.getCCFCoords()
        inSCAxons = (x<=170*25) & (z>=300*25)
        if self.inSCAxonsVol is None:
            filePath = fileIO.getFile('Select SC axons file','*.npy')
            self.inSCAxonsVol = np.load(filePath)
        inSCAxons = inSCAxons | np.array([self.inSCAxonsVol[int(i//25),int(j//25),int(k//25)] for i,j,k in zip(y,x,z)])
        return inSCAxons
        
        
    def getCellsInRegion(self,region=None,inSCAxons=None):
        if region is None:
            cellsInRegion = np.ones(len(self.data)).astype('bool')
        else:
            cellRegions = self.data.index.get_level_values('region')
            cellsInRegion = cellRegions==region
        if inSCAxons is not None:
            if inSCAxons:
                cellsInRegion = cellsInRegion & self.getSCAxons()
            else:
                cellsInRegion = cellsInRegion & ~self.getSCAxons()
        return cellsInRegion
        
        
    def getRegionID(self,regionLabel):
        if self.annotationStructures is None:
            f = fileIO.getFile('Choose annotation structures file','*.xml')
            self.annotationStructures = minidom.parse(f)
        for structure in self.annotationStructures.getElementsByTagName('structure'):
            if structure.childNodes[7].childNodes[0].nodeValue[1:-1]==regionLabel:
                return [int(sub.childNodes[0].nodeValue) for sub in structure.getElementsByTagName('id')]
        return []
        
        
    def getInRegion(self,regionLabel,padding=None):
        if self.annotationData is None:
            self.getAnnotationData()
        regionID = self.getRegionID(regionLabel)
        inRegion = np.in1d(self.annotationData,regionID).reshape(self.annotationData.shape)
        inRegion[:,inRegion.shape[1]//2:,:] = False
        if padding is not None:
            rng = [[a.min()-padding,a.max()+padding] for a in np.where(inRegion)]
            inRegion = inRegion[[slice(r[0],r[1]) for r in rng]]
        else:
            rng = [[0,s] for s in inRegion.shape]
        return inRegion,rng
        
        
    def getAnnotationData(self):
        f = fileIO.getFile('Choose annotation data file','*.nrrd')
        self.annotationData = nrrd.read(f)[0].transpose((1,2,0))
        
        
    def getIsPhotoTagged(self):
        i = np.array(self.data.index.get_level_values('photoTag').tolist())
        i[np.isnan(i)] = 0
        isPhotoTagged = i.astype(bool)
        notPhotoTagged = self.data.index.get_level_values('genotype')=='Ntsr1 Cre x Ai32'
        notPhotoTagged[isPhotoTagged] = False
        return isPhotoTagged,notPhotoTagged
    
    
    def plotLaserRaster(self):
        # plots spontaneous laser response
        # fig number is the dataframe index for each unit plotted
        if self.experimentFiles is None:
            self.getExperimentFiles()
        if self.excelFile is None:
            self.getExcelFile()
        figNumStart = 0
        for exp in self.experimentFiles:
            p = self.getProbeDataObj(exp)
            expDate,animalID = p.getExperimentInfo()
            table = pd.read_excel(self.excelFile,sheetname=expDate+'_'+animalID)
            if 'Genotype' not in table.keys() or table.Genotype[0]=='Ntsr1 Cre x Ai32':
                p.plotLaserRaster(figNum=figNumStart,nonMU=True)
            figNumStart += np.sum((self.data.index.get_level_values('experimentDate')==expDate) & (self.data.index.get_level_values('animalID')==animalID))
            
            
    def plotOMI(self,region=None,inSCAxons=None):
        cellsInRegion = self.getCellsInRegion(region,inSCAxons)
        
        rate = []
        omi = []
        
        # spont
        spontRate = []
        spontOmi = []
        windowDur = 15000
        uIndex = 0
        for exp in self.experimentFiles:
            p = self.getProbeDataObj(exp)
            protocol = p.getProtocolIndex('laser2')
            if protocol is None:
                protocol = p.getProtocolIndex('laser')
            try:
                pulseStarts,pulseEnds,pulseAmps = p.behaviorData[str(protocol)]['137_pulses']
            except:
                print(exp)
                uIndex += len(p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))[0])
                continue
            for u in p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))[0]:
                if cellsInRegion[uIndex]:                
                    spikes = p.units[u]['times'][str(protocol)]
                    controlResp = np.mean(p.findSpikesPerTrial(pulseStarts-windowDur,pulseStarts,spikes))
                    laserResp = np.mean(p.findSpikesPerTrial(pulseEnds-windowDur,pulseEnds,spikes))
                    spontRate.append(controlResp)
                    spontOmi.append((laserResp-controlResp)/(laserResp+controlResp))
                uIndex += 1
        rate.append(np.array(spontRate))
        omi.append(np.array(spontOmi))
        
        # sparseNoise
        controlMax,laserMax = [np.maximum(*[np.stack(self.data[laser].allTrials.sparseNoise[onOff][cellsInRegion]).max(axis=(1,2,3)) for onOff in ('onResp','offResp')]) for laser in ('laserOff','laserOn')]
        rate.append(controlMax)
        omi.append((laserMax-controlMax)/(laserMax+controlMax))
        
        # gratings
        controlMax,laserMax = [np.stack(self.data[laser].allTrials.gratings.respMat[cellsInRegion]).max(axis=(1,2,3)) for laser in ('laserOff','laserOn')]
        rate.append(controlMax)        
        omi.append((laserMax-controlMax)/(laserMax+controlMax))
        
        # checkerboard
        controlMax,laserMax = [np.stack(self.data[laser].allTrials.checkerboard.respMat[cellsInRegion]).max(axis=(1,2)) for laser in ('laserOff','laserOn')]
        rate.append(controlMax)        
        omi.append((laserMax-controlMax)/(laserMax+controlMax))
        
        # loom
        controlMax,laserMax = [np.stack(self.data[laser].allTrials.loom.peakResp[cellsInRegion]).max(axis=1) for laser in ('laserOff','laserOn')]
        rate.append(controlMax)        
        omi.append((laserMax-controlMax)/(laserMax+controlMax))
        
        # plot omi distributions
        protLabel = ('spont','sparseNoise','gratings','checkerboard','loom')
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1.1],'k--')
        for d,label,clr in zip(omi,protLabel,('k','r','g','b','0.5')):
            cumProb = [np.count_nonzero(d<=i)/d.size for i in np.sort(d)]
            ax.plot(np.sort(d),cumProb,clr,linewidth=2,label=label)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([-1,0.55])
        ax.set_ylim([0,1.01])
        ax.set_yticks([0,0.5,1])
        ax.set_xlabel('OMI',fontsize=20)
        ax.set_ylabel('Cumulative Probability',fontsize=20)
        plt.legend(loc='upper left',numpoints=1,frameon=False,fontsize=18)
        plt.tight_layout()
        
        # plot omi vs firing rate
        for r,i,label in zip(rate,omi,protLabel):
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            rmax = 1.05*r.max()
            ax.plot([0,rmax],[0,0],'k--')
            ax.plot(r,i,'ko')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xlim([-0.05,rmax])
            ax.set_ylim([-1.05,0.55])
            ax.set_yticks([-1,-0.5,0,0.5])
            ax.set_xlabel('Spikes/s',fontsize=20)
            ax.set_ylabel('OMI',fontsize=20)
            ax.set_title(label,fontsize=20)
            plt.tight_layout()
        
            
    def analyzeRF(self):
        
        inSCAxons = self.getSCAxons()
        
        region = 'LP'
       
        if region is None:
            cellsInRegion = np.ones(len(self.data)).astype('bool')
        else:
            cellRegions = self.data.index.get_level_values('region')
            cellsInRegion = cellRegions==region
        
        inSCAxons = inSCAxons[cellsInRegion]
        ccfY,ccfX,ccfZ = self.getCCFCoords(cellsInRegion)
        data = self.data.laserOff.allTrials.sparseNoise[cellsInRegion]        
        
        isOnOff = data.index.get_level_values('unitLabel')=='on off'
        isOn = data.index.get_level_values('unitLabel')=='on'
        isOff = data.index.get_level_values('unitLabel')=='off'  
        noRF = np.logical_not(isOnOff | isOn | isOff) 
        
        onVsOff = data.onVsOff
        onVsOff[noRF] = np.nan
        
        respLatency = np.stack(data.respLatency)*1000
        respLatency[isOff | noRF,0] = np.nan
        respLatency[isOn | noRF,1] = np.nan
        latencyCombined = np.nanmean(respLatency,axis=1)
        
        onFit = np.full((data.shape[0],7),np.nan)
        offFit = onFit.copy()
        sizeUsed = np.full(data.shape[0],np.nan)
        for u in range(data.shape[0]):
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
        
        # plot RF area and aspect ratio in and out of SC axons
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
        for rL in [respLatency[inSCAxons], respLatency[~inSCAxons]]:
            for i,clr in zip((0,1),('r','b')):
                plt.figure(facecolor='w')
                ax = plt.subplot(1,1,1)
                ax.hist(rL[~np.isnan(rL[:,i]),i],bins=np.arange(0,160,10),color=clr)
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
            sizeTuningOn = np.full((data.shape[0],len(sizeTuningSize)),np.nan)
            sizeTuningOff = sizeTuningOn.copy()
            ind = allSizesInd & scInd & (isOn | isOnOff)
            sizeTuningOn[ind] = np.stack(data.sizeTuningOn[ind])
            ind = allSizesInd & scInd & (isOff | isOnOff)
            sizeTuningOff[ind] = np.stack(data.sizeTuningOff[ind])
            
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
        goodUnits = []
        runData = self.data.laserOff.run.sparseNoise[cellsInRegion]
        statData = self.data.laserOff.stat.sparseNoise[cellsInRegion]
        for sub,tuning in zip(['on', 'off'], ['sizeTuningOn', 'sizeTuningOff']):
            good = []
            for u in xrange(data.shape[0]):
                st_r = np.array(runData[tuning][u])
                st_s = np.array(statData[tuning][u])
                
                if st_r.size > 3 and st_s.size > 3:
                    if ~any(np.isnan(st_r)) and ~any(np.isnan(st_s)):
                        label = data.index.get_level_values('unitLabel')[u]
                        if sub in label:                
                            good.append(u)
            goodUnits.append(good)
            
        for stateData,state in zip([statData, runData], ['stat', 'run']):
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
        
        # rf center colored by omi
        controlMax,laserMax = [np.maximum(*[np.stack(self.data[laser].allTrials.sparseNoise[onOff][cellsInRegion]).max(axis=(1,2,3)) for onOff in ('onResp','offResp')]) for laser in ('laserOff','laserOn')]
        omi = (laserMax-controlMax)/(laserMax+controlMax)
        expDate = data.index.get_level_values('experimentDate')
        centeredRFs = []
        centeredRFOMIs = []
        for exp in expDate.unique():
            expInd = expDate==exp
            for fit,label in zip((onFit,offFit),('On','Off')):
                ind = expInd & (fit[:,0]>-30) & (fit[:,0]<130) & (fit[:,1]>-40) & (fit[:,1]<95)
                weightedPopCenter = np.sum(fit[ind,:2]*-omi[ind][:,None],axis=0)/np.sum(-omi[ind])
                popCenter = fit[ind,:2].mean(axis=0)
                centeredRFs.append(fit[ind,:2]-popCenter)
                centeredRFOMIs.append(omi[ind])
                plt.figure(facecolor='w')
                ax = plt.subplot(1,1,1)
                ax.scatter(fit[ind,0],fit[ind,1],c=plt.cm.bwr(omi[ind]*0.5+0.5),s=100)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=18)
                ax.set_xlabel('Azimuth',fontsize=20)
                ax.set_ylabel('Elevation',fontsize=20)
                ax.set_title(exp+', '+label,fontsize=20)
                ax.set_aspect('equal')
                plt.tight_layout()
        centeredRFs = np.concatenate(centeredRFs)
        centeredRFOMIs = np.concatenate(centeredRFOMIs)
        
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.plot([-100,100],[0,0],'k--')
        ax.plot([0,0],[-100,100],'k--')
        ax.scatter(centeredRFs[:,0],centeredRFs[:,1],c=plt.cm.bwr(centeredRFOMIs*0.5+0.5),s=100)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([-100,100])
        ax.set_ylim([-100,100])
        ax.set_xlabel('Relative Azimuth',fontsize=20)
        ax.set_ylabel('Relative Elevation',fontsize=20)
        ax.set_title('OMI',fontsize=20)
        ax.set_aspect('equal')
        plt.tight_layout()
        
        radii = np.arange(10,105,10)
        inner = np.zeros(centeredRFs.shape[0],dtype=bool)
        ringOmiMean = np.zeros(radii.size)
        ringOmiSem = ringOmiMean.copy()
        for i,r in enumerate(radii):
            outer = ((centeredRFs[:,0]**2)+(centeredRFs[:,1]**2)<r**2)
            inRing = ~inner & outer
            inner = outer
            ringOmiMean[i] = centeredRFOMIs[inRing].mean()
            ringOmiSem[i] = centeredRFOMIs[inRing].std()/(inRing.sum()**0.5)
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.plot(radii,ringOmiMean,'k',linewidth=2)
        ax.fill_between(radii,ringOmiMean-ringOmiSem,ringOmiMean+ringOmiSem,color='k',alpha=0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([0,105])
        ax.set_xlabel('Bin Outer Radius',fontsize=20)
        ax.set_ylabel('Mean OMI',fontsize=20)
        plt.tight_layout()
        
        x = np.meshgrid(np.arange(-100,100.1,0.1))[0]
        inner = np.zeros((x.size,)*2,dtype=bool)
        ringOmiImg = np.zeros(inner.shape)
        for r,rOmi in zip(radii,ringOmiMean):
            outer = (x**2+x[:,None]**2)<r**2
            inRing = ~inner & outer
            inner = outer
            ringOmiImg[inRing] = -rOmi
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        im = ax.imshow(ringOmiImg,cmap='gray')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xticks([0,1000,2000])
        ax.set_yticks([0,1000,2000])
        ax.set_xticklabels([-100,0,100])
        ax.set_yticklabels([100,0,100])
        ax.set_xlabel('Azimuth',fontsize=20)
        ax.set_ylabel('Elevation',fontsize=20)
        ax.set_title('Mean OMI',fontsize=20)
        plt.colorbar(im)
        plt.tight_layout()
        
        # plot rf center vs probe position
        expDate = data.index.get_level_values('experimentDate')
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
                    e = patches.Ellipse(xy=[sub[0], sub[1]], width=sub[2], height=sub[3], angle=sub[4])
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
                
    
    def makeRFVolume(self, padding=10, sigma=3, region=None, weighted=False):
        
        cellsInRegion = self.getCellsInRegion(region)
        
        inLP,rng = self.getInRegion(region,padding=padding)
        LPmask = inLP.astype(float)
        LPmask[LPmask==0] = np.nan
        yRange,xRange,zRange = rng
        rangeSlice = tuple(slice(r[0],r[1]) for r in rng)
        
        CCFCoords = np.stack((self.data.index.get_level_values(c) for c in ('ccfX','ccfY','ccfZ')),axis=1)[cellsInRegion]
        
        counts = np.zeros([r[1]-r[0] for r in rng])
        elev = np.zeros_like(counts)
        azi = np.zeros_like(counts)
        data = self.data.laserOff.allTrials.sparseNoise[cellsInRegion]
        isOnOff = data.index.get_level_values('unitLabel')=='on off'
        isOn = data.index.get_level_values('unitLabel')=='on'
        isOff = data.index.get_level_values('unitLabel')=='off'  
        noRF = np.logical_not(isOnOff | isOn | isOff) 
        onFit = np.full((data.shape[0],7),np.nan)
        offFit = onFit.copy()
        sizeUsed = np.full(data.shape[0],np.nan)
        for u in range(data.shape[0]):
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
        
        for fitType, sub in zip([onFit, offFit], ['on', 'off']):
            for uindex, coords in enumerate(CCFCoords):
                if any(np.isnan(coords)):
                    continue
                else:
                    label = data.index.get_level_values('unitLabel')[uindex]
                    if sub in label:
                        ccf = coords/25
                        ccf = ccf.astype(int)
                        ccf -= np.array([xRange[0], yRange[0], zRange[0]])
                        
                        counts[ccf[1], ccf[0], ccf[2]]+=1
                        x,y = fitType[uindex][:2]
                        elev[ccf[1], ccf[0], ccf[2]] += y
                        azi[ccf[1], ccf[0], ccf[2]] += x
                        
        # plot recording positions
        ccfCoords = self.getCCFCoords(cellsInRegion)
        for a in range(3):
#            anyCounts = counts.any(axis=a).astype(np.uint8)
#            y,x = np.where(anyCounts)
            ind = [0,1,2]
            ind.remove(a)
            y,x = [ccfCoords[i]/25-rng[i][0] for i in ind]
            contours = cv2.findContours(LPmask.astype(np.uint8).max(axis=a).copy(order='C'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours)<3 else contours[1]
            cx,cy = np.squeeze(contours).T
            
            if self.inSCAxonsVol is None:
                _ = self.getSCAxons()
            scAxons = cv2.findContours(self.inSCAxonsVol[rangeSlice].astype(np.uint8).max(axis=a).copy(order='C'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            scAxons = scAxons[0] if len(scAxons)<3 else scAxons[1]
            s = [s.shape[0] for s in scAxons]
            scAxons = scAxons[s.index(max(s))]
            scx,scy = np.squeeze(scAxons).T
            
            if a==0:
                x,y = y,x
                cx,cy = cy,cx
                scx,scy = scy,scx
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            ax.add_patch(patches.Polygon(np.stack((scx,scy)).T,color='0.5',alpha=0.25))
            ax.plot(np.append(cx,cx[0]),np.append(cy,cy[0]),'k',linewidth=2)
            ax.plot(x,y,'ko',markersize=5)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_xlim([cx.min()-padding,cx.max()+padding])
            ax.set_ylim([cy.max()+padding,cy.min()-padding])
            plt.tight_layout()
                
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
              
        fullShape = self.getInRegion('LP')[0].shape+(4,)
        fullMap = [np.zeros(fullShape,dtype=np.uint8) for _ in (0,1)]
        for i in (0,1):
            fullMap[i][yRange[0]:yRange[1],xRange[0]:xRange[1],zRange[0]:zRange[1],:3] = colorMaps[i]*255
            fullMap[i][yRange[0]:yRange[1],xRange[0]:xRange[1],zRange[0]:zRange[1],3][~np.isnan(colorMaps[i][:,:,:,0])] = 255

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
        region = 'LP'
        if region is None:
            cellsInRegion = np.ones(len(self.data)).astype('bool')
        else:
            cellRegions = self.data.index.get_level_values('region')
            cellsInRegion = cellRegions==region
              
        sf = np.array([0.01,0.02,0.04,0.08,0.16,0.32])
        tf = np.array([0.5,1,2,4,8])
        
        laser = 'off'
        
        inSCAxons = self.getSCAxons()[cellsInRegion]
        
        laserOffData = self.data.laserOff.allTrials.gratings[cellsInRegion]
        hasGratings = laserOffData.respMat.notnull()
        
        laserOffRespMat = np.full((hasGratings.sum(),tf.size,sf.size),np.nan)
        if laser=='on':
            laserOnRespMat = laserOffRespMat.copy()
        for ind,u in enumerate(np.where(hasGratings)[0]):
            n = np.zeros(laserOffRespMat.shape[0],dtype=bool)
            n[ind] = True
            i = np.in1d(tf,np.round(laserOffData.tf[u],2))
            j = np.in1d(sf,np.round(laserOffData.sf[u],2))
            respInd = np.ix_(n,i,j)
            laserOffResp = laserOffData.respMat[u]
            bestOriInd = np.unravel_index(np.argmax(laserOffResp),laserOffResp.shape)[2]
            laserOffRespMat[respInd] = laserOffResp[:,:,bestOriInd]
            if laser=='on':
                laserOnResp = self.data.laserOn.allTrials.gratings[cellsInRegion].respMat[u]
                laserOnRespMat[respInd] = laserOnResp[:,:,bestOriInd]
                
        zthresh = 0
        spontRateMean = laserOffData.spontRateMean[hasGratings]
        spontRateStd = laserOffData.spontRateStd[hasGratings]
        respZ = (laserOffRespMat-spontRateMean[:,None,None])/spontRateStd[:,None,None]
        hasResp = (respZ>zthresh).any(axis=2).any(axis=1)
        
        stfFit = np.stack(laserOffData.stfFitParams)[hasGratings][hasResp]
        
        respMat = laserOffRespMat[hasResp]
        if laser=='on':
            stfFit = np.stack(self.data.laserOn.allTrials.gratings[cellsInRegion].stfFitParams[hasGratings][hasResp])
            respMat = laserOnRespMat[hasResp]
            
        inSCAxons = inSCAxons[hasGratings][hasResp]
        
        # plot mean resp
        for ind,title in zip((np.ones(inSCAxons.size,dtype=bool),inSCAxons,~inSCAxons),('all','SC','not SC')):
#            normRespMat = respMat[ind]/np.nanmax(respMat[ind],axis=(1,2))[:,None,None]
#            meanRespMat = np.nanmean(normRespMat,axis=0)
            meanRespMat = np.nanmean(respMat[ind],axis=0)
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            im = plt.imshow(meanRespMat,cmap='gray',interpolation='none',origin='lower')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xticklabels(sf)
            ax.set_yticklabels(tf)
            ax.set_xticks(np.arange(sf.size))
            ax.set_yticks(np.arange(tf.size))
            ax.set_xlabel('Cycles/deg',fontsize=20)
            ax.set_ylabel('Cycles/s',fontsize=20)
            ax.set_title(title+', laser '+laser,fontsize=20)
            plt.colorbar(im)
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
        ax.set_title('laser '+laser,fontsize=20)
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
        ax.set_title('laser '+laser,fontsize=20)
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
        ax.set_title('laser '+laser,fontsize=20)
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
        ax.set_title('laser '+laser,fontsize=20)
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
        ax.set_title('laser '+laser,fontsize=20)
        plt.tight_layout()
        
        
    def plotGratingsSDF(self,region=None,inSCAxons=None):
        sdfControl = []
        sdfLaser = []
        for expInd,exp in enumerate(self.experimentFiles):
            print('analyzing experiment '+str(expInd+1)+' of '+str(len(self.experimentFiles)))
            p = self.getProbeDataObj(exp)
            protocol = p.getProtocolIndex('gratings')
            laserTrials,controlTrials = p.findLaserTrials(protocol)
            p.analyzeGratings(trials=controlTrials,fit=False,plot=False,showLaserPreFrames=True,saveTag='_control')
            p.analyzeGratings(trials=laserTrials,fit=False,plot=False,showLaserPreFrames=True,saveTag='_laser')
            for u in p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))[0]:
                respMat = p.units[str(u)]['gratings_stf_control']['respMat']
                maxRespInd = np.unravel_index(np.argmax(respMat),respMat.shape)
                sdfControl.append(p.units[str(u)]['gratings_stf_control']['_sdf'][maxRespInd])
                sdfLaser.append(p.units[str(u)]['gratings_stf_laser']['_sdf'][maxRespInd])
        sdfTime = p.units[str(u)]['gratings_stf_control']['_sdfTime']
        frameRate = p.visstimData[str(protocol)]['frameRate']
        preTime = p.visstimData[str(protocol)]['preTime']/frameRate
        stimTime = p.visstimData[str(protocol)]['stimTime']/frameRate
        postTime = p.visstimData[str(protocol)]['postTime']/frameRate
        laserPreTime = p.visstimData[str(protocol)]['laserPreFrames']/frameRate
        laserPostTime = p.visstimData[str(protocol)]['laserPostFrames']/frameRate
        laserRampTime = p.visstimData[str(protocol)]['laserRampFrames']/frameRate
        stimStartTime = preTime+laserPreTime
        laserStart = preTime-0.25
        laserEnd = preTime+laserPreTime+stimTime+laserPostTime
        
        if inSCAxons is None:
            cellsInRegion = [self.getCellsInRegion(region)]
            label = ('',)
        else:
            cellsInRegion = [self.getCellsInRegion(region,inSCAxons=b) for b in (True,False)]
            label = ('in SC axons','not SC axons')
        
        for ind,lbl in zip(cellsInRegion,label):
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            h = 0.5 # bar height
            ax.add_patch(patches.Rectangle([stimStartTime,-h],width=stimTime,height=h,color='0.5'))
            ax.add_patch(patches.Polygon(np.array([[laserStart,-3*h],[laserStart+laserRampTime,-2*h],[laserEnd,-2*h],[laserEnd+laserRampTime,-3*h]]),color='b'))
            ymax = 0
            for s,clr in zip((sdfControl,sdfLaser),('k','b')):
                sdf = np.stack(s)[ind]
                sdfMean = sdf.mean(axis=0)
                sdfSem = sdf.std(axis=0)/(ind.sum()**0.5)
                ax.plot(sdfTime,sdfMean,clr,linewidth=2)
                ax.fill_between(sdfTime,sdfMean+sdfSem,sdfMean-sdfSem,color=clr,alpha=0.5)
                ymax = max(ymax,np.max(sdfMean+sdfSem))
            ax.set_xlim([0,5])
            ax.set_ylim([-2,1.05*ymax])
            ax.set_xticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,bottom=False,labelsize=18)
            ax.set_ylabel('Spikes/s',fontsize=20)
            ax.set_title(lbl,fontsize=20)
            plt.tight_layout()
            
        for ind,lbl in zip(cellsInRegion,label):
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            h = 0.5 # bar height
            ax.add_patch(patches.Rectangle([stimStartTime,-h],width=stimTime,height=h,color='0.5'))
            ax.add_patch(patches.Polygon(np.array([[laserStart,-3*h],[laserStart+laserRampTime,-2*h],[laserEnd,-2*h],[laserEnd+laserRampTime,-3*h]]),color='b'))
            ymax = 0
            for s,clr in zip((sdfControl,sdfLaser),('k','b')):
                sdf = np.stack(s)[ind]
                sdfMean = sdf.mean(axis=0)
                baselineInd = (sdfTime>stimStartTime-0.5) & (sdfTime<stimStartTime)
                if clr=='k':
                    baseline = sdfMean[baselineInd].mean()
                else:
                    sdfMean += baseline-sdfMean[baselineInd].mean()
                sdfSem = sdf.std(axis=0)/(ind.sum()**0.5)
                ax.plot(sdfTime,sdfMean,clr,linewidth=2)
                ax.fill_between(sdfTime,sdfMean+sdfSem,sdfMean-sdfSem,color=clr,alpha=0.5)
                ymax = max(ymax,np.max(sdfMean+sdfSem))
            ax.set_xlim([0,5])
            ax.set_xticks([])
            ax.set_ylim([-2,1.05*ymax])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,bottom=False,labelsize=18)
            ax.set_ylabel('Spikes/s (laser baseline shifted)',fontsize=20)
            ax.set_title(lbl,fontsize=20)
            plt.tight_layout()
        
        for ind,lbl in zip(cellsInRegion,label):
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            h = 0.05 # bar height
            ax.add_patch(patches.Rectangle([stimStartTime,-h],width=stimTime,height=h,color='0.5'))
            ax.add_patch(patches.Polygon(np.array([[laserStart,-3*h],[laserStart+laserRampTime,-2*h],[laserEnd,-2*h],[laserEnd+laserRampTime,-3*h]]),color='b'))
            ax.plot(sdfTime,np.stack(sdfLaser)[ind].mean(axis=0)/np.stack(sdfControl)[ind].mean(axis=0),'k',linewidth=2)
            ax.set_xlim([0,5])
            ax.set_xticks([])
            ax.set_ylim([-0.2,1.6])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,bottom=False,labelsize=18)
            ax.set_ylabel('Laser/Control',fontsize=20)
            ax.set_title(lbl,fontsize=20)
            plt.tight_layout()
    
    
    def analyzeOri(self):
        inSCAxons = self.getSCAxons()
        
        region = 'LP'
        if region is None:
            cellsInRegion = np.ones(len(self.data)).astype('bool')
        else:
            cellRegions = self.data.index.get_level_values('region')
            cellsInRegion = cellRegions==region
        
        inSCAxons = inSCAxons[cellsInRegion]
        ccfY,ccfX,ccfZ = self.getCCFCoords(cellsInRegion)
        data = self.data.laserOff.allTrials.gratings_ori[cellsInRegion]    
        
        ori = np.arange(0,360,45)
        
        
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
        
        
    def compareLaserCheckerboards(self,region=None):
        if region=='LP':
            inSCAxons = (True,False)
            label = ('in SC Axons','not SC Axons')
        else:
            inSCAxons = (None,)
            label = (str(region),)
        cellsInRegion = [self.getCellsInRegion(region,insc) for insc in inSCAxons]    
        
        patchSpeed = bckgndSpeed = np.array([-80,-20,0,20,80])
        
        controlData = self.data.laserOff.allTrials.checkerboard
        controlRespMat = np.stack(controlData.respMat)
        
        laserData = self.data.laserOn.allTrials.checkerboard
        laserRespMat = np.stack(laserData.respMat)
        
        controlMax = controlRespMat.max(axis=(1,2))
        laserMax = laserRespMat.max(axis=(1,2))
        omi = (laserMax-controlMax)/(laserMax+controlMax)
        
        zthresh = 6
        spontRateMean = np.array(controlData.spontRateMean)
        spontRateStd = np.array(controlData.spontRateStd)
        respZ = (controlRespMat-spontRateMean[:,None,None])/spontRateStd[:,None,None]
        hasResp = (respZ>zthresh).any(axis=2).any(axis=1)
        
        for cellInd,lbl in zip(cellsInRegion,label):
            ind = cellInd & hasResp & (omi<np.inf)
            meanRespControl = controlRespMat[ind].mean(axis=0)
            meanRespLaser = laserRespMat[ind].mean(axis=0)
            for resp,respType in zip((meanRespControl,meanRespLaser),('control','laser')):
                fig = plt.figure(facecolor='w')
                ax = fig.add_subplot(1,1,1)
                im = ax.imshow(resp,clim=(meanRespLaser.min(),meanRespControl.max()),cmap='gray',interpolation='none',origin='lower')
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=18)
                ax.set_xticks(np.arange(bckgndSpeed.size))
                ax.set_xticklabels(bckgndSpeed,fontsize=18)
                ax.set_yticks(np.arange(patchSpeed.size))
                ax.set_yticklabels(patchSpeed,fontsize=18)
                ax.set_xlabel('Background Speed (deg/s)',fontsize=20)
                ax.set_ylabel('Patch Speed (deg/s)',fontsize=20)
                ax.set_title(lbl+' '+respType,fontsize=20)
                plt.colorbar(im)
                plt.tight_layout()
                
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            controlMaxRows = controlRespMat[ind].max(axis=1)
            laserMaxRows = laserRespMat[ind].max(axis=1)
            meanControl = controlMaxRows.mean(axis=0)
            meanLaser = laserMaxRows.mean(axis=0)
            semControl = controlMaxRows.std(axis=0)/(ind.sum()**0.5)
            semLaser = laserMaxRows.std(axis=0)/(ind.sum()**0.5)
            ax.plot(bckgndSpeed,meanControl,'k',linewidth=2)
            ax.fill_between(bckgndSpeed,meanControl+semControl,meanControl-semControl,color='k',alpha=0.5)
            ax.plot(bckgndSpeed,meanLaser,'b',linewidth=2)
            ax.fill_between(bckgndSpeed,meanLaser+semLaser,meanLaser-semLaser,color='b',alpha=0.5)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_ylim([0,1.05*np.max(meanControl+semControl)])
            ax.set_xticks(bckgndSpeed)
            ax.set_xlabel('Background Speed (deg/s)',fontsize=20)
            ax.set_ylabel('Max Response (spikes/s)',fontsize=20)
            ax.set_title(lbl,fontsize=20)
            plt.tight_layout()
            
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            sup = 100*(1-laserMaxRows/controlMaxRows)
            meanSup = sup.mean(axis=0)
            semSup = sup.std(axis=0)/(ind.sum()**0.5)
            ax.plot(bckgndSpeed,meanSup,'k',linewidth=2)
            ax.fill_between(bckgndSpeed,meanSup+semSup,meanSup-semSup,color='k',alpha=0.5)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_ylim([0,1.05*np.max(meanSup+semSup)])
            ax.set_xticks(bckgndSpeed)
            ax.set_xlabel('Background Speed (deg/s)',fontsize=20)
            ax.set_ylabel('% Suppression',fontsize=20)
            ax.set_title(lbl,fontsize=20)
            plt.tight_layout()
    
    
    def analyzeCheckerboard(self, hasLaser=False):
        inSCAxons = self.getSCAxons()
        
        region = 'LP'
        if region is None:
            cellsInRegion = np.ones(len(self.data)).astype('bool')
        else:
            cellRegions = self.data.index.get_level_values('region')
            cellsInRegion = cellRegions==region
        
        inSCAxons = inSCAxons[cellsInRegion]
        ccfY,ccfX,ccfZ = self.getCCFCoords(cellsInRegion)
        
        data = self.data.laserOff.allTrials.checkerboard[cellsInRegion]    
        
        if hasLaser:
            patchSpeed = bckgndSpeed = np.array([-80,-20,0,20,80])
        else:
            patchSpeed = bckgndSpeed = np.array([-90,-30,-10,0,10,30,90])
        
        
        # get data from units with spikes during checkerboard protocol
        # ignore days with laser trials
        if not hasLaser:
            hasCheckerboard = (data.respMat.notnull()) & (self.data.laserOn.allTrials.checkerboard.respMat.isnull()[cellsInRegion]) 
        else:
            hasCheckerboard = (data.respMat.notnull())
        
        #get z score and determine significant responses
        zthresh = 10
        
        #if laser, take uindex from control trials
        if hasLaser:
            controlData = self.data.laserOff.allTrials.checkerboard[cellsInRegion]
            respMat = np.stack(controlData.respMat[hasCheckerboard])
            hasSpikes = respMat.any(axis=2).any(axis=1)
            respMat = respMat[hasSpikes]
            uindex = np.where(hasCheckerboard)[0][hasSpikes]
            spontRateMean = controlData.spontRateMean[uindex]
            spontRateStd = controlData.spontRateStd[uindex]
            respZ = (respMat-spontRateMean[:,None,None])/spontRateStd[:,None,None]
            hasResp = (respZ>zthresh).any(axis=2).any(axis=1)
            uindex = uindex[hasResp]
            
        else:
            respMat = np.stack(data.respMat[hasCheckerboard])
            hasSpikes = respMat.any(axis=2).any(axis=1)
            respMat = respMat[hasSpikes]
            uindex = np.where(hasCheckerboard)[0][hasSpikes]
            spontRateMean = data.spontRateMean[uindex]
            spontRateStd = data.spontRateStd[uindex]
            respZ = (respMat-spontRateMean[:,None,None])/spontRateStd[:,None,None]
            hasResp = (respZ>zthresh).any(axis=2).any(axis=1)
            uindex = uindex[hasResp]
        
        respMat = np.stack(data.respMat[uindex])
        inSCAxonsIndex = inSCAxons[uindex]        
        
#        # fill in NaNs where no running trials
#        statRespMat = np.stack(self.data.laserOff.stat.checkerboard.respMat[uindex])
#        y,x,z = np.where(np.isnan(respMat))
#        for i,j,k in zip(y,x,z):
#            respMat[i,j,k] = statRespMat[i,j,k]
       
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
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
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
        
        # plot mean response in and out of SC axons        
        for resp in [respMat[inSCAxonsIndex], respMat[~inSCAxonsIndex]]:
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            r = np.nanmean(resp, axis=0)
            cLim = np.max(r)
            im = ax.imshow(r,clim=(0,cLim),cmap='bwr',interpolation='none',origin='lower')
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
            plt.colorbar(im)
        
        for ind, label in zip([inSCAxonsIndex, ~inSCAxonsIndex], ['In SC', 'Out of SC']):
            fig = plt.figure(label, facecolor='w')
            gs = gridspec.GridSpec(3,4)
            for j,(r,title) in enumerate(zip((respMat,respMatNorm,respMatBaseSub,respMatBaseSubNorm),('resp','norm','base sub','base sub norm'))):
                ax = fig.add_subplot(gs[0,j])
                rmean = np.nanmean(r[ind], axis=0)
                ax.imshow(rmean,cmap='gray',interpolation='none',origin='lower')
                ax.set_title(title)
                
                ax = fig.add_subplot(gs[1,j])
                rstd = np.nanstd(r[ind], axis=0)
                ax.imshow(rstd,cmap='gray',interpolation='none',origin='lower')
                ax.set_title(title)
                
#                ax = fig.add_subplot(gs[2,j])
#                plt.hist(r[ind].ravel())
            
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
            clustID,_ = clust.ward(r,nClusters=k,plot=True)
            fig = plt.figure(facecolor='w')
            for i in np.unique(clustID):
                ax = fig.add_subplot(round(k**0.5),math.ceil(k**0.5),i)
                ax.imshow(respMat[clustID==i].mean(axis=0),cmap='bwr',interpolation='none',origin='lower')
                
        # cluster using principal components
        for r in (respMat, respMatNorm, respMatBaseSub, respMatBaseSubNorm):
            r = r.reshape((r.shape[0],r[0].size))
            pcaData,_,_ = clust.pca(r)
            k = 6
            clustID,_ = clust.ward(r[:,:6],nClusters=k,plot=True)
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
        
        
    def analyzeLoom(self):
        
        inSCAxons = self.getSCAxons()
        
        region = 'LP'
        if region is None:
            cellsInRegion = np.ones(len(self.data)).astype('bool')
        else:
            cellRegions = self.data.index.get_level_values('region')
            cellsInRegion = cellRegions==region
        
        inSCAxons = inSCAxons[cellsInRegion]
        ccfY,ccfX,ccfZ = self.getCCFCoords(cellsInRegion)
        data = self.data.laserOff.allTrials.loom[cellsInRegion]            
        
        lvRatio = np.array([10,20,40,80])
        
        hasLoom = np.array(data.peakResp.notnull())
        
        
        peakResp = np.stack(data.peakResp[hasLoom])
        spontRateMean = np.array(data.spontRateMean[hasLoom])
        spontRateStd = np.array(data.spontRateStd[hasLoom])
        respZ = (peakResp-spontRateMean[:,None])/spontRateStd[:,None]
        hasResp = (respZ>6).any(axis=1)
        hasRespInd = np.where(hasLoom)[0][hasResp]
        hasNoRespInd = np.where(hasLoom)[0][~hasResp]
        
        maxLoomResp = np.max(peakResp,axis=1)
        maxLoomZ = np.max(respZ,axis=1)
        
        # compare max loom peak resp and z score in/out SC axons
        inSCInd = inSCAxons[np.where(hasLoom)[0]]
        for resp,binSize,xlab in zip((maxLoomResp-spontRateMean,maxLoomZ),(5,2),('spikes/s','z score')):
            axmax = 1.05*resp.max()
            for ind,title in zip((inSCInd,~inSCInd),('SC Axons','Not SC Axons')):
                plt.figure(facecolor='w')
                ax = plt.subplot(1,1,1)
                ax.hist(resp[ind],bins=np.arange(0,axmax,binSize),range=[0,axmax],color='k')
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=18)
                ax.set_xlabel('Max Loom Response ('+xlab+')',fontsize=20)
                ax.set_ylabel('Number of Cells',fontsize=20)
                ax.set_title(title,fontsize=20)
                plt.tight_layout()
        
        # plot cumulative distribution for inSC and outSC max responses        
        fig, ax = plt.subplots()
        for ind, color in zip((inSCInd, ~inSCInd), ('k', 'g')):
            axmax = 1.05*maxLoomZ.max()
            h, b = np.histogram(maxLoomZ[ind], bins=np.arange(0, 30, 2))
            h = h/np.sum(h)
            ax.plot(b, np.append(np.cumsum(h), 1), color)
            ax.set_xlim([0, 31])
            ax.set_ylim([0, 1.1])
        probeData.formatFigure(fig, ax, 'Max Loom response', xLabel='z-score', yLabel='Cumulative fraction')
        ax.text(22, 0.2, 'Out of SC axons', {'color':'g'})
        ax.text(22, 0.1, 'In SC axons', {'color':'k'})    
        
        
        # histogram of r-value of linear fit
        # linFit = (slope, intercept, r-value, p-value, stderror)
        peakTimes = np.stack(data.bestConditionPeakTimes[hasRespInd])
        linFitLV = np.zeros((hasRespInd.size,5))
        linFitSqrtLV = linFitLV.copy()
        for ind,pt in enumerate(peakTimes):
            linFitLV[ind] = scipy.stats.linregress(lvRatio,pt)
            linFitSqrtLV[ind] = scipy.stats.linregress(np.sqrt(lvRatio),pt)
        
        binWidth = 0.05
        for r,label in zip((linFitLV[:,2],linFitSqrtLV[:,2]),('Size/Speed','sqrt(Size/Speed)')):
            for ind,title in zip((inSCAxons[hasRespInd],~inSCAxons[hasRespInd]),('SC Axons','Not SC Axons')):
                plt.figure(facecolor='w')
                ax = plt.subplot(1,1,1)
                h,bins = np.histogram(r[ind],bins=np.arange(-1,1+binWidth/2,binWidth))
                ax.bar(bins[:-1],h/ind.sum(),width=binWidth,color='k')
                ax.set_xlim([-1,1])
                ax.set_ylim([0,0.55])
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=18)
                ax.set_xlabel('r (Peak Time vs '+label+')',fontsize=20)
                ax.set_ylabel('Fraction of Cells',fontsize=20)
                ax.set_title(title,fontsize=20)
                plt.tight_layout()
                
        # plot r values vs location in LP
        padding = 10
        jitter = 3
        inLP,rng = self.getInLP(padding=padding)
        ccfCoords = self.getCCFCoords(cellsInRegion)
        isGoodFit = linFitLV[:,2]>=0.95
        goodFitInd = hasRespInd[isGoodFit]
        notGoodFitInd = np.union1d(hasNoRespInd,hasRespInd[~isGoodFit])
        for a in range(3):
            ind = [0,1,2]
            ind.remove(a)
            y,x = [ccfCoords[i]/25-rng[i][0] for i in ind]
            contours,_ = cv2.findContours(inLP.astype(np.uint8).max(axis=a).copy(order='C'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cx,cy = np.squeeze(contours).T
            if a==0:
                x,y = y,x
                cx,cy = cy,cx
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(1,1,1)
            ax.plot(np.append(cx,cx[0]),np.append(cy,cy[0]),'k',linewidth=2)
            jit = np.random.uniform(-jitter,jitter+1,(2,notGoodFitInd.size))
            ax.plot(x[notGoodFitInd]+jit[0],y[notGoodFitInd]+jit[1],'o',mfc='0.5',mec='none')
            jit = np.random.uniform(-jitter,jitter+1,(2,goodFitInd.size))
            ax.plot(x[goodFitInd]+jit[0],y[goodFitInd]+jit[1],'o',mfc='r',mec='none')
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_xlim([cx.min()-padding,cx.max()+padding])
            ax.set_ylim([cy.max()+padding,cy.min()-padding])
            plt.tight_layout()
            
        # compare max loom and checkerboard resp
        patchSpeed = bckgndSpeed = np.array([-90,-30,-10,0,10,30,90])
        checkerData = self.data.laserOff.allTrials.checkerboard[cellsInRegion]
        hasCheckerboard = (checkerData.respMat.notnull()) & (self.data.laserOn.allTrials.checkerboard.respMat.isnull()[cellsInRegion])
        uindex = np.where(hasCheckerboard & hasLoom)[0]
        respMat = np.stack(checkerData.respMat[uindex])
        spontRateMean = checkerData.spontRateMean[uindex]
        spontRateStd = checkerData.spontRateStd[uindex]
        respZ = (respMat-spontRateMean[:,None,None])/spontRateStd[:,None,None]
        maxPatchResp = respMat[:,patchSpeed!=0,bckgndSpeed==0].max(axis=1)
        maxBckgndResp = respMat[:,patchSpeed==0,bckgndSpeed!=0].max(axis=1)
        patchIndex = (maxPatchResp-maxBckgndResp)/(maxPatchResp+maxBckgndResp)
        
        axmax = 1.05*max(maxBckgndResp.max(),maxPatchResp.max(),maxLoomResp.max())
        for c,xlab in zip((maxBckgndResp,maxPatchResp),('Background','Patch')):
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            ax.plot([0,axmax],[0,axmax],'k--')
            i = np.in1d(uindex,notGoodFitInd)
            ax.plot(c[i],maxLoomResp[i],'o',mfc='0.5',mec='none')
            i = np.in1d(uindex,goodFitInd)
            ax.plot(c[i],maxLoomResp[i],'o',mfc='r',mec='none')
            ax.set_xlim([0,axmax])
            ax.set_ylim([0,axmax])
            ax.set_aspect('equal')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xlabel('Max '+xlab+' Response (spikes/s)',fontsize=20)
            ax.set_ylabel('Max Loom Response (spikes/s)',fontsize=20)
            plt.tight_layout()
            
        patchIndexGoodFit = patchIndex[np.in1d(uindex,goodFitInd) & inSCInd]
        patchIndexNotGoodFit = patchIndex[np.in1d(uindex,notGoodFitInd) & inSCInd]
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        cumProbPatchIndexGoodFit = [np.count_nonzero(patchIndexGoodFit<=i)/patchIndexGoodFit.size for i in np.sort(patchIndexGoodFit)]
        cumProbPatchIndexNotGoodFit = [np.count_nonzero(patchIndexNotGoodFit<=i)/patchIndexNotGoodFit.size for i in np.sort(patchIndexNotGoodFit)]
        ax.plot([0,0],[0,1],'k--')
        ax.plot(np.sort(patchIndexNotGoodFit),cumProbPatchIndexNotGoodFit,'0.5',linewidth=3)
        ax.plot(np.sort(patchIndexGoodFit),cumProbPatchIndexGoodFit,'r',linewidth=3)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlim([-0.85,0.85])
        ax.set_ylim([0,1.01])
        ax.set_xticks([-0.5,0,0.5])
        ax.set_yticks([0,0.5,1])
        ax.set_xlabel('Patch-Background Index',fontsize=20)
        ax.set_ylabel('Cumulative Probability',fontsize=20)
        plt.tight_layout()
                
        # r vs slope
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.plot(linFitLV[:,0],linFitLV[:,2],'ko')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xlabel('Slope',fontsize=20)
        ax.set_ylabel('r',fontsize=20)
        plt.tight_layout()
        
        # plot fit for each cell        
        for pt,fit,inSC in zip(peakTimes,linFitLV,inSCAxons[uindex]):
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            ax.plot(lvRatio,lvRatio*fit[0]+fit[1],'k-')
            ax.plot(lvRatio,pt,'ko')
            ax.set_xlim([0,90])
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xlabel('Size/Speed',fontsize=20)
            ax.set_ylabel('Peak Time (ms)',fontsize=20)
            title = 'SC' if inSC else 'not SC'
            ax.set_title('r = '+str(fit[2])+', '+title,fontsize=20)
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
        
    def analyzeRunningCheckerboard(self, laser=False):
        sR = []
        rR = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        laserCond = 'laserOn' if laser else 'laserOff'
        for exp in self.experimentFiles:
            p = self.getProbeDataObj(exp)
            units, _ = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            pind = p.getProtocolIndex('checkerboard')
            if 'checkerboard_laserOff_run' in p.units[units[0]] and 'checkerboard_laserOff_stat' in p.units[units[0]]:
                
                rTrials = p.units[units[0]]['checkerboard_laserOn_run']['trials'] if laser else p.units[units[0]]['checkerboard_laserOff_run']['trials']
                sTrials = p.units[units[0]]['checkerboard_laserOn_stat']['trials'] if laser else p.units[units[0]]['checkerboard_laserOff_stat']['trials']
                
                paramsToMatch = ['trialPatchPos', 'trialBckgndDir', 'trialBckgndSpeed', 'trialPatchDir', 'trialPatchSpeed']
                
                sMatchedTrials, rMatchedTrials = self.trialMatch(p, 'checkerboard', paramsToMatch, sTrials, rTrials)
                
                sMatchedTrials = np.array(sMatchedTrials)
                rMatchedTrials = np.array(rMatchedTrials)

                p.analyzeCheckerboard(trials=sMatchedTrials, saveTag='_runningAnalysis_stat_' + laserCond, laser=laser,plot=False)
                p.analyzeCheckerboard(trials=rMatchedTrials, saveTag='_runningAnalysis_run_' + laserCond, laser=laser, plot=False)
                for u in units:
#                    sResponse = np.nanmax(p.units[u]['checkerboard_runningAnalysis_stat_' + laserCond]['respMat'])
#                    rResponse = np.nanmax(p.units[u]['checkerboard_runningAnalysis_run_' + laserCond]['respMat'])
                    sResponse = p.units[u]['checkerboard_runningAnalysis_stat_' + laserCond]['respMat']
                    rResponse = p.units[u]['checkerboard_runningAnalysis_run_' + laserCond]['respMat']
                    sR.append(sResponse)
                    rR.append(rResponse)
        
        ax.plot(sR, rR, 'ko', alpha=0.5)
        maxR = np.nanmax([np.nanmax(sR), np.nanmax(rR)])
        ax.plot([0, maxR], [0, maxR], 'r--')     
        
        
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
                
                
    def analyzeWaveforms(self):
        
        templates = []
        for i,exp in enumerate(self.experimentFiles):
            print('analyzing experiment '+str(i+1)+' of '+str(len(self.experimentFiles)))
            p = self.getProbeDataObj(exp)
            units = p.getUnitsByLabel('label',('on','off','on off','supp','noRF'))
            units, _ = p.getOrderedUnits(units)
            temps = [p.getSpikeTemplate(u, plot=False) for u in units]
            templates.extend(temps)
        
        tempArray = np.array(templates)
        maxs = np.max(np.abs(tempArray), axis=1)
        tempArray /= maxs[:, None]
        peakPosition = np.argmax(np.abs(tempArray),axis=1)
        tempShifted = [tempArray[i, ppo - peakPosition.min() : ppo + tempArray.shape[1] - peakPosition.max()] for i, ppo in enumerate(peakPosition)]
        tempShifted = np.array(tempShifted)
        
        peakToTrough = np.zeros(tempShifted.shape[0])       
        for i,t in enumerate(tempShifted):
            peakInd = np.argmax(np.absolute(t))
            peakToTrough[i] = (np.argmin(t[peakInd:]) if t[peakInd]>0 else np.argmax(t[peakInd:]))/30.0
            
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
        
        return tempShifted, peakToTrough
    
    def findRegions(self, ccfCoords, tolerance=100):
        mcc = MouseConnectivityCache(manifest_file='connectivity/mouse_connectivity_manifest.json')
        struct_df = mcc.get_structures()
        
        if self.annotationData is None:
            self.getAnnotationData()
        
        ccf = np.copy(ccfCoords)
        ccf /= 25
        ccf = np.round(ccf).astype(int)
        regionID = self.annotationData[ccf[1], ccf[0], ccf[2]]
        if regionID!=218:

            #check to see if unit is within 'tolerance' microns of LP        
            tolerance /= 25
            inLP, _ = self.getInRegion('LP')
            mask = np.zeros(inLP.shape)
            rng = [[int(a-tolerance-1),int(a+tolerance+1)] for a in [ccf[1], ccf[0], ccf[2]]]
            mask[[slice(r[0],r[1]) for r in rng]] = 1
            inLP *= mask.astype('bool')
#            LPregion = np.array(np.where(self.annotationData==218)).T
            if np.sum(inLP)>0:            
                LPregion = np.array(np.where(inLP)).T
                distances = np.sum((LPregion - [ccf[1], ccf[0], ccf[2]])**2, axis=1)**0.5
                if distances.min() <= tolerance:
                    regionID=218
                
#            for (y, x, z) in LPregion:
#                if euclidean([y,x,z],[ccf[1], ccf[0], ccf[2]])<=tolerance:
#                    regionID=218
#                    break
        
        return struct_df[struct_df['id']==regionID]['acronym'].tolist()[0]
        
#        def getUnitsByWaveform(self, cellType = 'FS'):
                
        
def findPeakToTrough(waveformArray, sampleRate=30000, plot=True):
    #waveform array should be units x samples
    if waveformArray.ndim==1:
        waveformArray=waveformArray[None,:]
    
    peakToTrough = np.zeros(len(waveformArray))       
    for iw, w in enumerate(waveformArray):
#        peakInd = np.argmax(np.absolute(w))
#        peakToTrough[iw] = (np.argmin(w[peakInd:]) if w[peakInd]>0 else np.argmax(w[peakInd:]))/(sampleRate/1000.0)
        if any(np.isnan(w)):
            peakToTrough[iw] = np.nan
        else:
            peakInd = np.argmin(w)
            peakToTrough[iw] = np.argmax(w[peakInd:])/(sampleRate/1000.0)       
    
    if plot:
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.hist(peakToTrough[~np.isnan(peakToTrough)],np.arange(0,1.2,0.05),color='k')
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