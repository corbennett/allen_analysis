# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:07:49 2016

@author: svc_ccg
"""

import clust, fileIO, probeData
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec


class popProbeData():
    
    def __init__(self):
        self.dataFiles = []
        self.probeDataObjs = []
        self.data = None
    
    
    def getDataFiles(self):
        filePaths = fileIO.getFiles(fileType='*.hdf5')
        if filePaths is None:
            return
        self.dataFiles = filePaths
        self.probeDataObjs = []
        for f in filePaths:
            self.probeDataObjs.append(probeData.probeData())
            self.probeDataObjs[-1].loadHDF5(f)
    
        
    def getUnitLabels(self):
        filePath = fileIO.getFile(fileType='*.xlsx')
        if filePath is None:
            return
        for p in self.probeDataObjs:
            p.readExcelFile(fileName=filePath)
        
        
    def checkUnitLabels(self):
        isLabeled = []
        for p in self.probeDataObjs:
            isLabeled.append(all([label in list(u.keys()) for u in p.units.values() for label in ('label','CCFCoords')]))
        return isLabeled
    
    
    def getData(self, unitLabel=['on','off','on off'], tag=''):
        
        if unitLabel is not None and not all(self.checkUnitLabels()):
            raise ValueError('Not all units are labeled')
            
        # dataFrame rows
        rowNames = ('experimentDate','animalID','unitID','unitLabel')
        experimentDate = []
        animalID = []
        unitID = []
        unitLbl = []
        
        # dataFrame columns
        columnNames = ('paramType','paramName')
        paramType = ['probe','probe']
        paramName = ['ypos','CCFCoords']
        
        # data: dictionary corresponding to columns with len(units) lists of values
        data = {'probe':{'ypos':[],'CCFCoords':[]}}
        protocolList = ('sparseNoise','gratings_stf','gratings_ori','checkerboard')
        for protocol in protocolList:
            data[protocol] = {}
        for p in self.probeDataObjs:
            expDate,anmID = p.getExperimentInfo()
            units = p.units.keys() if unitLabel is None else p.getUnitsByLabel('label',unitLabel)
            units,ypos = p.getOrderedUnits(units)
            for u,pos in zip(units,ypos):
                experimentDate.append(expDate)
                animalID.append(anmID)
                unitID.append(u)
                unitLbl.append(p.units[u]['label'])
                data['probe']['ypos'].append(pos)
                data['probe']['CCFCoords'].append(p.units[u]['CCFCoords'])
                for protocol in protocolList:
                    protocolName = 'gratings' if protocol=='gratings_stf' else protocol
                    protocolInd = p.getProtocolIndex(protocolName)
                    if protocolInd is None:
                        for pname in data[protocol]:
                            data[protocol][pname].append(np.nan)
                    else:
                        if protocol+tag not in p.units[u]:
                            if protocol=='sparseNoise':
                                p.findRF(u,saveTag=tag,plot=False)
                            elif protocol=='gratings_stf':
                                p.analyzeGratings(u,saveTag=tag,plot=False)
                            elif protocol=='gratings_ori':
                                p.analyzeGratings(u,protocolType='ori',saveTag=tag,plot=False)
                            elif protocol=='checkerboard':
                                p.analyzeCheckerboard(u,saveTag=tag,plot=False)
                        for param,val in p.units[u][protocol+tag].items():
                            if param not in data[protocol]:
                                paramType.append(protocol)
                                paramName.append(param)
                                data[protocol][param] = [np.nan for _ in range(len(unitID)-1)]
                            self.d = data
                            self.param = param
                            self.paramName = paramName
                            self.prot = protocol
                            data[protocol][param].append(val)
            
        rows = pd.MultiIndex.from_arrays([experimentDate,animalID,unitID,unitLbl],names=rowNames)
        cols = pd.MultiIndex.from_arrays([paramType,paramName],names=columnNames)
        self.data = pd.DataFrame(index=rows,columns=cols)
        for ptype in data:
            for pname in data[ptype]:
                self.data[ptype,pname] = data[ptype][pname]
    
    
    def loadData(self, filePath=None):
        filePath = fileIO.getFile(fileType='*.hdf5')
        if filePath is None:
            return
        self.data = pd.read_hdf(filePath,'table')
    
    
    def saveData(self, filePath=None):
        filePath = fileIO.saveFile(fileType='*.hdf5')
        if filePath is None:
            return
        self.data.to_hdf(filePath,'table')
    
                
    def analyzeRF(self):
        respLatency = np.reshape(self.data['rf']['respLatency'],(-1,2))
        plt.hist(respLatency,bins=np.arange(0,0.2,0.005))
    
    
    def analyzeSTF(self):
        pass
    
    
    def analyzeOri(self):
        pass
    
    
    def analyzeCheckerboard(self):
        
        patchSpeed = bckgndSpeed = np.array([-90,-30,-10,0,10,30,90]) 
        
        # get data from units with checkerboard protocol
        hasCheckerboard = self.data.checkerboard.respMat.notnull()
        respMat = np.stack(self.data.checkerboard.respMat[hasCheckerboard])
        hasSpikes = respMat.any(axis=2).any(axis=1)
        respMat = respMat[hasSpikes]
        uindex = np.where(hasCheckerboard)[0][hasSpikes]
        
        # find distance between RF and patch
        onVsOff = np.array(self.data.sparseNoise.onVsOff[uindex])
        onFit = np.stack(self.data.sparseNoise.onFit[uindex])
        offFit = np.stack(self.data.sparseNoise.offFit[uindex])
        rfElev = offFit[:,1].copy()
        rfElev[onVsOff>0] = onFit[onVsOff>0,1]
        patchDist = np.full(uindex.size,np.nan)
        for i,u in enumerate(uindex):
            patchDist[i] = np.min(np.absolute(self.data.checkerboard.patchElevation[u]-rfElev[i]))
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot(patchDist,respMat[:,patchSpeed!=0,bckgndSpeed==0].max(axis=1),'ko')
        ax.set_xlabel('Patch RF Distance')
        ax.set_ylabel('Max Patch Response')
        
        # get z score and determine significant responses
        spontRateMean = self.data.checkerboard.spontRateMean[uindex]
        spontRateStd = self.data.checkerboard.spontRateStd[uindex]
        respZ = (respMat-spontRateMean[:,None,None])/spontRateStd[:,None,None]
        hasResp = (respZ>5).any(axis=2).any(axis=1)
        
        respMat = respMat[hasResp]
        
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
            
        # compare patch and bckgnd responses and contribution to variance
        maxPatchResp = respMat[:,patchSpeed!=0,bckgndSpeed==0].max(axis=1)
        maxBckgndResp = respMat[:,patchSpeed==0,bckgndSpeed!=0].max(axis=1)
        patchBckgndIndex = (maxPatchResp-maxBckgndResp)/(maxPatchResp+maxBckgndResp)
        
        patchVar = respMat.mean(axis=2).std(axis=1)
        bckgndVar = respMat.mean(axis=1).std(axis=1)
        varIndex = (patchVar-bckgndVar)/(patchVar+bckgndVar)
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot([-1,1],[0,0],color='0.6')
        ax.plot([0,0],[-1,1],color='0.6')
        ax.plot(patchBckgndIndex,varIndex,'ko')
        ax.set_xlim((-0.75,0.75))
        ax.set_ylim((-0.75,0.75))
        ax.set_xlabel('Patch vs Background Response')
        ax.set_ylabel('Patch vs Background Speed Variance')
        
        # plot pc weightings
        for r in (respMat, respMatNorm, respMatBaseSub, respMatBaseSubNorm):
            r = r.reshape((r.shape[0],r[0].size))
            pcaData,eigVal,eigVec = clust.pca(r)
            fig = plt.figure()
            for i in range(9):
                ax = fig.add_subplot(3,3,i+1)
                cLim = max(abs(eigVec[:,i]))
                ax.imshow(eigVec[:,i].reshape(respMat.shape[1:]),clim=(-cLim,cLim),cmap='bwr',interpolation='none',origin='lower')
                
        # clusters using all dimensions
        for r in (respMat, respMatNorm, respMatBaseSub, respMatBaseSubNorm):
            r = r.reshape((r.shape[0],r[0].size))
            k = 4
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
        for r in (respMat, respMatNorm, respMatBaseSub, respMatBaseSubNorm):
            r = r.reshape((r.shape[0],r[0].size))
            clustIDHier,linkageMat = clust.nestedPCAClust(r,nSplit=3,minClustSize=2,varExplained=0.75,clustID=[],linkageMat=[])
            clustID = clust.getClustersFromHierarchy(clustIDHier)
            k = len(set(clustID))
            fig = plt.figure(facecolor='w')
            for ind,i in enumerate(np.unique(clustID)):
                ax = fig.add_subplot(round(k**0.5),math.ceil(k**0.5),ind+1)
                ax.imshow(respMat[clustID==i].mean(axis=0),cmap='bwr',interpolation='none',origin='lower')

        


if __name__=="__main__":
    pass