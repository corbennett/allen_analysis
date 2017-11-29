# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:16:42 2016

@author: Gale
"""

from __future__ import division
import numpy as np
import scipy.cluster
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_samples, silhouette_score
from kmodes import kmodes as km
from probeData import formatFigure


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
    
    
def lda(data,clustID,plot=False):
    clusters = np.unique(clustID)
    withinClusterScatter = np.zeros((data.shape[1],)*2)
    betweenClusterScatter = np.zeros_like(withinClusterScatter)
    overallMean = data.mean(axis=0)
    for clust in clusters:
        clusterData = data[clustID==clust]
        clusterMean = clusterData.mean(axis=0)
        diff = clusterMean-overallMean
        betweenClusterScatter += clusterData.shape[0]*diff[:,None].dot(diff[None,:])
        for sample in clusterData:
            diff = sample-clusterMean
            withinClusterScatter +=  diff[:,None].dot(diff[None,:])
        
    eigVal,eigVec = np.linalg.eig(np.linalg.inv(withinClusterScatter+np.eye(data.shape[1])*1e-6).dot(betweenClusterScatter))
    eigVal = np.real(eigVal)
    order = np.argsort(eigVal)[::-1]
    eigVal = eigVal[order]
    eigVec = np.real(eigVec)[:,order]
    ldaData = data.dot(eigVec)
    
    if plot:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(1,clusters.size),eigVal[:clusters.size-1].cumsum()/eigVal[:clusters.size-1].sum(),'k')
        ax.set_xlim((0.5,clusters.size-0.5))
        ax.set_ylim((0,1.02))
        ax.set_xlabel('Discriminant')
        ax.set_ylabel('Cumulative Fraction of Variance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(eigVec,clim=(-1,1),cmap='bwr',interpolation='none',origin='lower')
        ax.set_xlabel('Discriminant')
        ax.set_ylabel('Parameter')
        ax.set_title('Dicriminant Weightings')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(length=0)
        cb.set_ticks([-1,0,1])
    return ldaData,eigVal,eigVec
    
    
def plotClusters3d(data,clustID,method='pca',colors=None):
    if method=='pca':
        plotData = pca(data)[0]
    elif method=='lda':
        plotData = lda(data,clustID)[0]
    clusters = np.unique(clustID)
    plt.figure()
    ax = plt.subplot(111,projection='3d')
    if colors is None:
        colors = plt.cm.Set1(1/(np.arange(clusters.size,dtype=float)+1))[:,:3]
    for clust,clr in zip(clusters,colors):
        i = clustID==clust
        if clusters.size>2:
            ax.plot(plotData[i,0],plotData[i,1],plotData[i,2],'o',mec=clr,mfc='none')
        else:
            ax.plot(plotData[i,0],plotData[i,1],'o',mec=clr,mfc='none')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if clusters.size>2:
        ax.set_zlabel('PC3')
    
    
def plotPseudoF(data,maxClusters=None,method='kmeans',iterations=100):
    if maxClusters is None:
        maxClusters = data.shape[0]//2
    Fratio = np.zeros(maxClusters)
    overallMean = data.mean(axis=0)
    for k in range(1,maxClusters+1):
        if method=='kmeans':
            clustID,_ = kmeans(data,k,iterations)
        elif method=='ward':
            clustID,_ = ward(data,k)
        betweenClusterSSE = 0
        withinClusterSSE = 0
        for clust in np.unique(clustID):
            clusterData = data[clustID==clust]
            clusterMean = clusterData.mean(axis=0)
            betweenClusterSSE += clusterData.shape[0]*np.sum(np.square(clusterMean-overallMean))
            withinClusterSSE += np.sum(np.square(clusterData-clusterMean))
        Fratio[k-1] = betweenClusterSSE/withinClusterSSE
        
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.arange(1,maxClusters+1),Fratio,'k')
    ax.set_xlim([0,maxClusters+0.5])
    ax.set_xlabel('# Clusters')
    ax.set_ylabel('Pseudo F Ratio (SSE Between/SSE Within)')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    
    
def plotMeanSilhouetteValue(data,maxClusters=None,method='kmeans',iterations=100):
    if maxClusters is None:
        maxClusters = data.shape[0]//2
    silValues = []
    for k in xrange(2,maxClusters+1):
        if method=='kmeans':
            clustID,_ = kmeans(data,k,iterations)
        elif method=='ward':
            clustID,_ = ward(data,k)
        
        silValues.append(silhouette_score(data, clustID))
        
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.arange(2,maxClusters+1),silValues,'k')
    ax.set_xlim([0,maxClusters+0.5])
    ax.set_xlabel('# Clusters')
    ax.set_ylabel('Mean Silhouette Value')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)

         
def silhouettePlot(data,clusterLabels,metric='euclidean'):
    numClusters = np.unique(clusterLabels).size
    silVals = silhouette_samples(data, clusterLabels, metric=metric)
    
    fig = plt.figure('Silhouette Values')
    ax = fig.add_subplot(111)
    
    ylower = 10
    for ic, clust in enumerate(np.unique(clusterLabels)):
        clustSilVals = silVals[clusterLabels==clust]
        clustSilVals = np.sort(clustSilVals)
        
        clustYs = np.arange(ylower, ylower + clustSilVals.size)
        color = plt.cm.spectral(float(ic) / numClusters)
        
        ax.fill_betweenx(clustYs, 0, clustSilVals, facecolor=color, edgecolor=color)
        ax.text(-0.05, ylower + 0.5*clustSilVals.size, str(ic))
        ax.text(0.95*ax.get_xlim()[1], ylower + 0.5*clustSilVals.size, str(np.mean(clustSilVals)))
        ylower += clustSilVals.size + 10
        
    formatFigure(fig, ax, xLabel='Silhouette Values', yLabel='Cluster')
            
    
def kmeans(data,k,iterations=100,initCentroids='points',plot=False):
    # data is n samples x m parameters 
    centroids,clustID = scipy.cluster.vq.kmeans2(data,k,iter=iterations,minit=initCentroids)
    clustID += 1
    if plot:
        plotPseudoF(data,maxClusters=min(k*2,data.shape[0]),method='kmeans',iterations=iterations)
        plotMeanSilhouetteValue(data,maxClusters=min(k*2,data.shape[0]),method='kmeans',iterations=iterations)
        plotClusters3d(data,clustID)
    return clustID,centroids
    
    
def kmodes(data,k,iterations=100,plot=False):
    m = km.KModes(n_clusters=k, init='Huang', n_init=iterations, verbose=False)
    clustID = m.fit_predict(data)+1
    centroids = m.cluster_centroids_
    if plot:
        pass
    return clustID,centroids

    
def cluster(data,nClusters=None,method='ward',metric='euclidean',plot=False,colors=None):
    # data is n samples x m parameters
    linkageMat = scipy.cluster.hierarchy.linkage(data,method=method,metric=metric)
    if nClusters is None:
        clustID = None
    else:
        clustID = scipy.cluster.hierarchy.fcluster(linkageMat,nClusters,'maxclust')
    if plot:
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        colorThresh = 0 if nClusters<2 else linkageMat[::-1,2][nClusters-2]
        if colors is not None:
            scipy.cluster.hierarchy.set_link_color_palette(list(colors))
        scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,color_threshold=colorThresh,above_threshold_color='k')
        scipy.cluster.hierarchy.set_link_color_palette(None)
        ax.set_yticks([])
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
            
        n = 1000
        randLinkage = np.zeros((n,linkageMat.shape[0]))
        rdata = data.copy()
        for i in range(n):
            for j in range(data.shape[1]):
                rdata[:,j] = data[np.random.permutation(data.shape[0]),j]
            _,m = cluster(rdata)
            randLinkage[i] = m[::-1,2]
        
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        k = np.arange(linkageMat.shape[0])+2
        ax.plot(k,np.percentile(randLinkage,1,axis=0),'k--')
        ax.plot(k,np.percentile(randLinkage,99,axis=0),'k--')
        ax.plot(k,linkageMat[::-1,2],'k')
        ax.set_xlim([0,data.shape[0]])
        ax.set_xlabel('# Clusters')
        ax.set_ylabel('Minimum Linkage Distance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        
#        plotPseudoF(data,method='ward')
#        plotMeanSilhouetteValue(data,method='ward')
#        if clustID is not None:
#            plotClusters3d(data,clustID)
    return clustID,linkageMat
    
    
def ward(data,nClusters=None,plot=False):
    clustID,linkageMat = cluster(data,nClusters=nClusters,method='ward',metric='euclidean',plot=plot)

        
def nestedPCAClust(data,method='kmeans',nSplit=2,minClustSize=2,varExplained=0.9,clustID=[],linkageMat=[]):
    pcaData,eigVal,_ = pca(data)
    pcToUse = np.where(eigVal.cumsum()/eigVal.sum()>varExplained)[0][0]+1
    if method=='kmeans':
        c,_ = kmeans(pcaData[:,:pcToUse],2)
        clustID.append(c)
        linkageMat.append(None)
    elif method=='ward':
        c,link = ward(pcaData[:,:pcToUse],nClusters=2)
        clustID.append(c)
        linkageMat.append(link)
    if nSplit>1:
        for i in (1,2):
            clustID.append([])
            linkageMat.append([])
            if np.count_nonzero(c==i)>minClustSize:
                _,_ = nestedPCAClust(data[c==i],method,nSplit-1,minClustSize,varExplained,clustID[-1],linkageMat[-1])
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