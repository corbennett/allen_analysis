# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 14:17:52 2017

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

#########################
#cluster params from RF
#########################
data = self.data.laserOff.allTrials.sparseNoise
numCells = len(data)            
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
rfAreaCombined = np.nanmean(np.stack((onArea,offArea)),axis=0)

sizeTuningCOM = []          #center of mass for size tuning curves
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
st_z = (sizeTuningCOM - np.nanmean(sizeTuningCOM))/np.nanstd(sizeTuningCOM)
rfa_z = (rfAreaCombined - np.nanmean(rfAreaCombined))/np.nanstd(rfAreaCombined)

########################
#cluster params from stf
########################
data = self.data.laserOff.allTrials.gratings
hasGratings = data.respMat.notnull()
stfFit = np.stack(data.stfFitParams[hasGratings])
sfs = stfFit[:, 0]
tfs = stfFit[:, 1]
sfs_all = np.full(numCells, np.nan)
sfs_all[np.array(hasGratings)] = sfs
tfs_all = np.full(numCells, np.nan)
tfs_all[np.array(hasGratings)] = tfs
sfLog = np.log2(sfs_all)
tfLog = np.log2(tfs_all)

sf_z = (sfLog - np.nanmean(sfLog))/np.nanstd(sfLog) 
tf_z = (tfLog - np.nanmean(tfLog))/np.nanstd(tfLog)

###########################
#cluster params from ori
###########################
data = self.data.laserOff.allTrials.gratings_ori
hasOri = data.respMat.notnull()
dsi = np.array(data.dsi[hasOri])
osi = np.array(data.osi[hasOri])
osi_all = np.full(numCells, np.nan)
osi_all[np.array(hasOri)] = osi
dsi_all = np.full(numCells, np.nan)
dsi_all[np.array(hasOri)] = dsi

osi_z = (osi_all - np.nanmean(osi_all))/np.nanstd(osi_all)
dsi_z = (dsi_all - np.nanmean(dsi_all))/np.nanstd(dsi_all)

##################################
#cluster params from checkerboard
##################################
data = self.data.laserOff.allTrials.checkerboard
patchSpeed = bckgndSpeed = np.array([-90,-30,-10,0,10,30,90])
hasCheckerboard = data.respMat.notnull()
respMat = np.stack(data.respMat[hasCheckerboard])
hasSpikes = respMat.any(axis=2).any(axis=1)
respMat = respMat[hasSpikes]
uindex = np.where(hasCheckerboard)[0][hasSpikes]

spontRateMean = data.spontRateMean[uindex]
spontRateStd = data.spontRateStd[uindex]
respZ = (respMat-spontRateMean[:,None,None])/spontRateStd[:,None,None]
hasResp = (respZ>0).any(axis=2).any(axis=1)
respMat = respMat[hasResp]
uindex = uindex[hasResp]

#statRespMat = np.stack(self.data.laserOff.stat.checkerboard.respMat[uindex])
#y,x,z = np.where(np.isnan(respMat))
#for i,j,k in zip(y,x,z):
#    respMat[i,j,k] = statRespMat[i,j,k]
#
#respMat = respMat
maxPatchResp = respMat[:,patchSpeed!=0,bckgndSpeed==0].max(axis=1)
maxBckgndResp = respMat[:,patchSpeed==0,bckgndSpeed!=0].max(axis=1)
patchIndex = (maxPatchResp-maxBckgndResp)/(maxPatchResp+maxBckgndResp)

pI = np.full(numCells, np.nan)
pI[uindex] = patchIndex
pI_z = (pI - np.nanmean(pI))/np.nanstd(pI)

nanmask = pI_z + osi_z + dsi_z + sf_z + tf_z + st_z + rfa_z
nonNanIndex = ~np.isnan(nanmask)    

####################
#cluster data
####################
cData = []
for param in [pI_z, osi_z, dsi_z, sf_z, tf_z, st_z, rfa_z]:
    cData.append(param[nonNanIndex])

cData = np.array(cData).T
pcData, evects, evals = clust.pca(cData, plot=True)

numClusters = 2
clustID, lm = clust.cluster(cData, numClusters, plot = True)
#clustID, lm = clust.kmeans(pcData[:, :3], numClusters)

#plot histograms of params for each cluster
cD = []
colors=['gray', 'cyan', 'b']
names = ['pI', 'osi', 'dsi', 'sfLog', 'tfLog', 'sizeTuningCOM', 'rfArea']
for param, name in zip([pI, osi_all, dsi_all, sfLog, tfLog, sizeTuningCOM, rfAreaCombined], names):
    paramNoNan = param[nonNanIndex]    
    cD.append(paramNoNan)
    
    plt.figure(name)
    minVal = np.nanmin(paramNoNan)
    maxVal = np.nanmax(paramNoNan)
    bins = np.arange(minVal, maxVal, 20)
    for c in np.unique(clustID):
        h, b = np.histogram(paramNoNan[clustID==c], bins=20, range=[minVal, maxVal])
        h = h/np.sum(h)
        plt.bar(b[:-1], h, np.diff(b)[0], facecolor = colors[c-1], edgecolor=colors[c-1], alpha=0.5)
#        plt.hist(paramNoNan[clustID==c], bins=20, range=[minVal, maxVal], color = colors[c-1], alpha=0.5, normed=True)    
        

##########################
#Plot clusters in LP
##########################

#Get CCF data for clustered cells
CCFCoords = np.stack((self.data.index.get_level_values(c) for c in ('ccfX','ccfY','ccfZ')),axis=1)
clustCCF = CCFCoords[nonNanIndex, :]

padding = 10

#Get Atlas data for LP contours
annotationData,_ = nrrd.read('annotation_25.nrrd')
annotationData = annotationData.transpose((1,2,0))
inLP = np.where(annotationData == 218)
inLPbinary = annotationData==218
maxProj = np.max(inLPbinary, axis=2).astype(int)                
cnts,_ = cv2.findContours(maxProj.copy(order='C').astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
leftSide = np.argmin([np.min(u[:, :, 0]) for u in cnts])

xRange = [np.min(cnts[leftSide][:, :, 0]) - padding, np.max(cnts[leftSide][:, :, 0]) + padding]
yRange = [np.min(inLP[0])-padding, np.max(inLP[0])+padding]
zRange = [np.min(inLP[2])-padding, np.max(inLP[2])+padding]

LPmask = inLPbinary[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]].astype(np.float)
LPmask[LPmask==0] = np.nan

x = np.random.randint(0, 1, clustCCF.shape[0]) + (clustCCF[:, 0]/25)-xRange[0]
y = np.random.randint(0, 1, clustCCF.shape[0]) + (clustCCF[:, 1]/25)-yRange[0]
z = np.random.randint(0, 1, clustCCF.shape[0]) + (clustCCF[:, 2]/25)-zRange[0]
plotXYZ = [[z, x], [z,y], [x,y]]

#Plot cells in LP
colors = ['gray', 'cyan', 'orange', 'green', 'red']
for a in [0, 1, 2]:
    counts = np.zeros([np.diff(yRange), np.diff(xRange), np.diff(zRange)])
    anyCounts = 255-counts.any(axis=a).astype(np.uint8)*255
    anyCounts = np.stack([anyCounts]*3,axis=-1)
    
    contours,_ = cv2.findContours(LPmask.astype(np.uint8).max(axis=a).copy(order='C'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(anyCounts,contours,-1,(127,)*3)
    plt.figure()
    plt.imshow(anyCounts)
    for cID in np.unique(clustID):
#        plt.plot(plotXYZ[a][0][clustID==cID], plotXYZ[a][1][clustID==cID], 'o', mfc='none', mec=colors[cID-1]) 
        plt.scatter(plotXYZ[a][0][clustID==cID], plotXYZ[a][1][clustID==cID], color=colors[cID-1], lw=0,alpha=0.4, s=60) 
        
        
        
##########################################################
#Plot mean gratings and checkerboard matrices for clusters  
##########################################################

#Plot Checkerboard responses for clusters
patchSpeed = bckgndSpeed = np.array([-90,-30,-10,0,10,30,90])
respMat = self.data.laserOff.allTrials.checkerboard.respMat[nonNanIndex]
for cID in np.unique(clustID):
    rm = respMat[clustID==cID]
    fig = plt.figure('Cluster ' + str(cID) + ' checkerboard', facecolor='w')
    ax = fig.add_subplot(111)    
    im=ax.imshow(np.mean(rm, axis=0), interpolation='none', cmap='bwr', origin='lower')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=18)
    ax.set_xticks(np.arange(bckgndSpeed.size))
    ax.set_xticklabels(bckgndSpeed,fontsize=18)
    ax.set_yticks(np.arange(patchSpeed.size))
    ax.set_yticklabels(patchSpeed,fontsize=18)
    ax.set_xlabel('Background Speed (deg/s)',fontsize=20)
    ax.set_ylabel('Patch Speed (deg/s)',fontsize=20)
    plt.colorbar(im)

        
#Plot mean gratings_stf matrix for clusters
sf = np.array([0.01,0.02,0.04,0.08,0.16,0.32])
tf = np.array([0.5,1,2,4,8])
data = self.data.laserOff.allTrials.gratings[nonNanIndex]
for cID in np.unique(clustID):
    clustData = data[clustID==cID]
    hasGratings = clustData.respMat.notnull()
    
    stfFit = np.stack(clustData.stfFitParams[hasGratings])
    
    # plot mean resp and f1/f0 matrices
    numUnits = np.count_nonzero(hasGratings)
    respMat = np.full((numUnits,tf.size,sf.size),np.nan)
    f1f0Mat = respMat.copy()
    for uind,u in enumerate(np.where(hasGratings)[0]):
        n = np.zeros(numUnits,dtype=bool)
        n[uind] = True
        i = np.in1d(tf,np.round(clustData.tf[u],2))
        j = np.in1d(sf,np.round(clustData.sf[u],2))
        ind = np.ix_(n,i,j)
        resp = clustData.respMat[u]
        f1f0 = clustData.f1f0Mat[u]
        bestOriInd = np.unravel_index(np.argmax(resp),resp.shape)[2]
        respMat[ind] = resp[:,:,bestOriInd]
        f1f0Mat[ind] = f1f0[:,:,bestOriInd]
    meanNormRespMat = np.nanmean(respMat/np.nanmax(np.nanmax(respMat,axis=2),axis=1)[:,None,None],axis=0)
    
    fig = plt.figure('Cluster ' + str(cID) +' gratings stf',facecolor='w')
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
            
        

#Plot size tuning curves for clusters
sizeTuningSize = [5,10,20,50]
sizeTuningLabel = [5,10,20,'full']
data = self.data.laserOff.allTrials.sparseNoise[nonNanIndex]
labels = data.index.get_level_values('unitLabel')        
for cID in np.unique(clustID):
    clustData = data[clustID==cID]
    clustLabels = labels[clustID==cID]
    
    hasOn = ['on' in l for l in clustLabels]
    hasOff = ['off' in l for l in clustLabels]
    
    sizeTuningOn = np.stack(clustData.sizeTuningOn[np.array(hasOn)])
    sizeTuningOff = np.stack(clustData.sizeTuningOff[np.array(hasOff)])
    
    fig = plt.figure('Cluster ' + str(cID) +' size tuning', facecolor='w')
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
        
        ax = fig.add_subplot(2,2,2*axind+2)
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
    
    
    
        
        
        
        

