# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:07:49 2016

@author: svc_ccg
"""

import clust, fileIO, probeData
import cv2, math, nrrd
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse


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
    
    
    def getData(self, unitLabel=['on','off','on off','supp','noRF'], tag='', replace=False):
        
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
                        if replace or protocol+tag not in p.units[u]:
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
        
        isOn = self.data.index.get_level_values(3)=='on'
        isOff = self.data.index.get_level_values(3)=='off'  
        
        probePos = self.data.probe.ypos
        
        onFit = np.stack(self.data.sparseNoise.onFit)
        offFit = np.stack(self.data.sparseNoise.offFit)
        onFit[isOff,:] = np.nan
        offFit[isOn,:] = np.nan
        
        onArea = np.pi*np.prod(onFit[:,2:4],axis=1)
        onArea[onArea<100] = np.nan
        offArea = np.pi*np.prod(offFit[:,2:4],axis=1)
        
        respLatency = np.stack(self.data.sparseNoise.respLatency)*1000
        respLatency[isOff,0] = np.nan
        respLatency[isOn,1] = np.nan
        
        ind = self.data.index.get_level_values(1)=='258284'
        sizeTuningOn = np.stack(self.data.sparseNoise.sizeTuningOn[ind])
        sizeTuningOff = np.stack(self.data.sparseNoise.sizeTuningOff[ind])
        
        
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
            
        for rfA,clr in zip((onArea,offArea),('r','b')):
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            ax.hist(rfA[~np.isnan(rfA)],bins=np.arange(0,4000,200),color=clr)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='large')
            ax.set_xlabel('Receptive Field Area ($\mathregular{degrees^2}$)',fontsize='x-large')
            ax.set_ylabel('Number of Cells',fontsize='x-large')
            plt.tight_layout()
            
        sizeTuningSize = [5,10,20,50]
        sizeTuningLabel = [5,10,20,'full']
        for sizeResp,clr in zip((sizeTuningOn,sizeTuningOff),('r','b')):
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            sizeRespNorm = sizeResp/np.nanmax(sizeResp,axis=1)[:,None]
            sizeRespMean = np.nanmean(sizeRespNorm,axis=0)
            sizeRespStd = np.nanstd(sizeRespNorm,axis=0)
            plt.plot(sizeTuningSize,sizeRespMean,color=clr,linewidth=3)
            plt.fill_between(sizeTuningSize,sizeRespMean+sizeRespStd,sizeRespMean-sizeRespStd,color=clr,alpha=0.3)
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
            
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
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
        
        for i in [0]:
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            plt.plot(onFit[:,0],onFit[:,1],'ro',label='On')
            plt.plot(offFit[:,0],offFit[:,1],'bo',label='Off')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize='large')
            ax.set_xlabel('Azimuth',fontsize='x-large')
            ax.set_ylabel('Elevation',fontsize='x-large')
            plt.legend(frameon=False)
            plt.tight_layout()
        
        xlim = np.array((-1300,-600))
        for exp in ('07282016','07292016','09162016','10192016','10202016'):
            ind = self.data.index.get_level_values(0)==exp
            for rf,clr in zip((onFit,offFit),('r','b')):
                for i,azimOrElev in zip((0,1),('Azimuth','Elevation')):
                    plt.figure(facecolor='w')
                    ax = plt.subplot(1,1,1)
                    hasRF = np.logical_not(np.isnan(rf[ind,i]))
                    linFit = scipy.stats.linregress(probePos[ind][hasRF],rf[ind,i][hasRF])
                    ax.plot(xlim,linFit[0]*xlim+linFit[1],color='0.6',linewidth=2)
                    ax.plot(probePos[ind],rf[ind,i],'o',color=clr,markersize=10)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize='large')
                    ax.set_xlim(xlim)
                    ylim = (-20,120) if azimOrElev=='Azimuth' else (-40,80)
                    ax.set_ylim(ylim)
                    ax.set_xlabel('Probe Position (microns)',fontsize='x-large')
                    ax.set_ylabel(azimOrElev+' (degrees)',fontsize='x-large')
                    plt.tight_layout()
    
        colors = [(1, 0, 0), (0, 0, 1)]
        for i, fit in enumerate([onFit, offFit]):
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(111, aspect='equal')
            
            for sub in fit:
                x,y = probeData.getEllipseXY(*sub[:5])
                e = Ellipse(xy=[sub[0], sub[1]], width=sub[2], height=sub[3], angle=sub[4])
                e.set_edgecolor('none')
                e.set_alpha(0.2)
                e.set_facecolor(colors[i])
        
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize='large')
                ax.set_xlabel('Azimuth',fontsize='x-large')
                ax.set_ylabel('Elevation',fontsize='x-large')
                ax.add_artist(e)
                ax.set_xlim(-30, 110)
                ax.set_ylim(-45, 75)
                fig.tight_layout()
                
    
    def makeRFVolume(self, padding=10, sigma=1, annotationDataFile=None):
        if annotationDataFile is None:
            annotationDataFile = fileIO.getFile() 
        
        annotationData,_ = nrrd.read(annotationDataFile)
        annotationData = annotationData.transpose((1,2,0))
        inLP = np.where(annotationData == 218)
        inLPbinary = annotationData==218
        
        yRange = [np.min(inLP[0])-padding, np.max(inLP[0])+padding]
        xRange = [np.min(inLP[1])-padding, np.max(inLP[1])/2+padding]
        zRange = [np.min(inLP[2])-padding, np.max(inLP[2])+padding]
        
        LPmask = inLPbinary[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]].astype(np.float)
        LPmask[LPmask==0] = np.nan
        
        CCFCoords = self.data.probe.CCFCoords
        
        counts = np.zeros([np.diff(yRange), np.diff(xRange), np.diff(zRange)])
        elev = np.zeros_like(counts)
        azi = np.zeros_like(counts)
        for fitType in ['onFit', 'offFit']:
            for uindex, coords in enumerate(CCFCoords):
                
                if any(np.isnan(coords)):
                    continue
                else:
                    ccf = coords/25
                    ccf = ccf.astype(int)
                    ccf -= np.array([xRange[0], yRange[0], zRange[0]])
                    
                    counts[ccf[1], ccf[0], ccf[2]]+=1
                    elev[ccf[1], ccf[0], ccf[2]] += self.data.sparseNoise[fitType][uindex][1]
                    azi[ccf[1], ccf[0], ccf[2]] += self.data.sparseNoise[fitType][uindex][0]
                
        elev /= counts
        azi /= counts
        
        
        elev_s = probeData.gaussianConvolve3D(elev,sigma)
        azi_s = probeData.gaussianConvolve3D(azi, sigma)
        
        elev_s *= LPmask
        azi_s *= LPmask
                
        
        for z in xrange(elev_s.shape[2]):
            lpSlice = inLPbinary[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0] + z].astype(np.uint8)
            _,contour,_ = cv2.findContours(np.copy(lpSlice, order='C').astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            eslice = np.copy(elev_s[:, :, z], order='C')
            aslice = np.copy(azi_s[:, :, z], order='C')
            cv2.drawContours(eslice,contour,-1,(0,0,0),1)
            cv2.drawContours(aslice,contour,-1,(0,0,0),1)
            
            elev_s[:, :, z] = eslice
            azi_s[:, :, z] = aslice
#            xs = []
#            ys = []
#            for c in contour[1]:
#                xs.append(c[0][0] - xRange[0])
#                ys.append(c[0][1] - yRange[0])
##        
#
#        plt.figure()
#        plt.imshow(elev_s[:, :, goodSlice-zRange[0]], interpolation='none')
#        plt.plot(xs, ys, 'k')
#        plt.colorbar() 
#        
#        plt.figure()
#        plt.imshow(azi_s[:, :, goodSlice-zRange[0]], interpolation='none')
#        plt.plot(xs, ys, 'k')
#        plt.colorbar() 
        return elev_s, azi_s
    
    
    def analyzeSTF(self):
        
        sf = np.array([0.01,0.02,0.04,0.08,0.16,0.32])
        tf = np.array([0.5,1,2,4,8])
        
        hasGratings = self.data.gratings_stf.respMat.notnull()
        
        stfFit = np.stack(self.data.gratings_stf.stfFitParams[hasGratings])
        
        fig = plt.figure(facecolor='w')
        
        ax = fig.add_subplot(2,2,1)
        ax.plot(np.log2(stfFit[:,0]),np.log2(stfFit[:,1]),'ko',markerfacecolor='none')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='medium')
        ax.set_xlim(np.log2([sf[0]*0.5,sf[-1]*1.5]))
        ax.set_ylim(np.log2([tf[0]*0.5,tf[-1]*1.5]))
        ax.set_xticks(np.log2([sf[0],sf[-1]]))
        ax.set_yticks(np.log2([tf[0],tf[-1]]))
        ax.set_xticklabels([sf[0],sf[-1]])
        ax.set_yticklabels([tf[0],tf[-1]])
        ax.set_xlabel('Cycles/deg',fontsize='large')
        ax.set_ylabel('Cycles/s',fontsize='large')
        
        ax = fig.add_subplot(2,2,2)
        tfBins = np.concatenate(([0],tf*1.5))
        tfHist,_ = np.histogram(stfFit[~np.isnan(stfFit[:,1]),1],tfBins)
        ax.bar(np.arange(tf.size)+0.6,tfHist,color='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='medium')
        ax.set_xticks(np.arange(tf.size)+1)
        ax.set_xticklabels(tf)
        ax.set_xlabel('Center TF',fontsize='large')
        ax.set_ylabel('Number of Units')
        
        ax = fig.add_subplot(2,2,3)
        sfBins = np.concatenate(([0],sf*1.5))
        sfHist,_ = np.histogram(stfFit[~np.isnan(stfFit[:,0]),0],sfBins)
        ax.bar(np.arange(sf.size)+0.6,sfHist,color='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='medium')
        ax.set_xticks(np.arange(sf.size)+1)
        ax.set_xticklabels(sf)
        ax.set_xlabel('Center SF',fontsize='large')
        ax.set_ylabel('Number of Units')
        
        ax = fig.add_subplot(2,2,4)
        bins = np.arange(-0.3,1.4,0.1)
        ax.hist(stfFit[~np.isnan(stfFit[:,4]),4],bins=bins,color='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='medium')
        ax.set_xticks([0,0.5,1])
        ax.set_xlabel('Speed Tuning Index',fontsize='large')
        ax.set_ylabel('Number of Units')
        
        plt.tight_layout()
    
    
    def analyzeOri(self):
        
        ori = np.arange(0,360,45)
        
        hasGratings = self.data.gratings_ori.respMat.notnull()
        
        dsi = np.array(self.data.gratings_ori.dsi[hasGratings])
        prefDir = np.array(self.data.gratings_ori.prefDir[hasGratings])
        osi = np.array(self.data.gratings_ori.osi[hasGratings])
        prefOri = np.array(self.data.gratings_ori.prefOri[hasGratings])
            
        for i in [0]:
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