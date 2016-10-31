# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:16:42 2016

@author: Gale
"""

import numpy as np
import scipy.cluster
from matplotlib import pyplot as plt


def pca(data,plotEigVal=False):
    # data is n samples x m parameters
    eigVal,eigVec = np.linalg.eigh(np.cov(data,rowvar=False))
    order = np.argsort(eigVal)[::-1]
    eigVal = eigVal[order]
    eigVec = eigVec[:,order]
    pcaData = data.dot(eigVec)
    if plotEigVal:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(1,eigVal.size+1),eigVal.cumsum()/eigVal.sum(),'k')
        ax.set_xlim((0.5,eigVal.size+0.5))
        ax.set_ylim((0,1))
        ax.set_xlabel('PC')
        ax.set_ylabel('Cumulative Fraction of Variance')
    return pcaData,eigVal,eigVec

    
def ward(data,nClusters=None,plotDendrogram=False):
    # data is n samples x m parameters
    linkageMat = scipy.cluster.hierarchy.linkage(data,'ward')
    if plotDendrogram:
        plt.figure(facecolor='w')
        scipy.cluster.hierarchy.dendrogram(linkageMat)
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