# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:16:42 2016

@author: Gale
"""

import numpy as np
import scipy.cluster
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def standardizeData(data):
    # data is n samples x m parameters
    data = data.copy()
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    return data


def pca(data,plot=False):
    # data is n samples x m parameters
    eigVal,eigVec = np.linalg.eigh(np.cov(data,rowvar=False))
    order = np.argsort(eigVal)[::-1]
    eigVal = eigVal[order]
    eigVec = eigVec[:,order]
    pcaData = data.dot(eigVec)
    if plot:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(1,eigVal.size+1),eigVal.cumsum()/eigVal.sum(),'k')
        ax.set_xlim((0.5,eigVal.size+0.5))
        ax.set_ylim((0,1.02))
        ax.set_xlabel('PC')
        ax.set_ylabel('Cumulative Fraction of Variance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(eigVec,clim=(-1,1),cmap='bwr',interpolation='none',origin='lower')
        ax.set_xlabel('PC')
        ax.set_ylabel('Parameter')
        ax.set_title('PC Weightings')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(length=0)
        cb.set_ticks([-1,0,1])
    return pcaData,eigVal,eigVec
    
    
def plotClusters3d(data,clustID,colors=None):
    pcaData,_,_ = pca(data)
    clusters = np.unique(clustID)
    plt.figure()
    ax = plt.subplot(111,projection='3d')
    if colors is None:
        colors = plt.cm.Set1(1/(np.arange(clusters.size)+1))
    for clust,clr in zip(clusters,colors):
        i = clustID==clust
        ax.plot(pcaData[i,0],pcaData[i,1],pcaData[i,2],'o',mec=clr,mfc='none')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    
def kmeans(data,k,iterations=100,initCentroids='points'):
    # data is n samples x m parameters 
    centriods,clustID = scipy.cluster.vq.kmeans2(data,k,iter=iterations,minit=initCentroids)
    return clustID

    
def ward(data,nClusters=None,plotDendrogram=False):
    # data is n samples x m parameters
    linkageMat = scipy.cluster.hierarchy.linkage(data,'ward')
    if plotDendrogram:
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        colorThresh = 0 if nClusters<2 else linkageMat[::-1,2][nClusters-2]
        scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,color_threshold=colorThresh)
        ax.set_yticks([])
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
        
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.plot(np.arange(linkageMat.shape[0])+2,linkageMat[::-1,2],'k')
        ax.set_xlim([0,data.shape[0]])
        ax.set_xlabel('# Clusters')
        ax.set_ylabel('Minimum Linkage Distance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
    if nClusters is not None:
        clustID = scipy.cluster.hierarchy.fcluster(linkageMat,nClusters,'maxclust')
        return clustID,linkageMat

        
def nestedPCAClust(data,nSplit,minClustSize,varExplained=0.9,clustID=[],linkageMat=[]):
    pcaData,eigVal,_ = pca(data)
    pcToUse = np.where(eigVal.cumsum()/eigVal.sum()>varExplained)[0][0]+1
    c,link = ward(pcaData[:,:pcToUse],nClusters=2)
    clustID.append(c)
    linkageMat.append(link)
    if nSplit>1:
        for i in (1,2):
            clustID.append([])
            linkageMat.append([])
            if np.count_nonzero(c==i)>minClustSize:
                _,_ = nestedPCAClust(data[c==i],nSplit-1,minClustSize,varExplained,clustID[-1],linkageMat[-1])
    return clustID,linkageMat

    
def getClustersFromHierarchy(clustIDHier,clustID=None,index=None):
    if clustID is None:
        clustID = clustIDHier[0].copy()
    else:
        clustID[index] = clustIDHier[0]+clustID.max()
    if len(clustIDHier)>1:
        maxID = clustID.max()
        for i,c in enumerate(clustIDHier[1:]):
            if len(c)>0:
                _ = getClustersFromHierarchy(c,clustID,clustID==maxID-1+i)
    return clustID-clustID.min()+1 

        
if __name__=="__main__":
    pass