# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:43:18 2017

@author: joshs

Added bootstrap and signaltonoise by Xiaoxuan Jia 

"""

import numpy as np
import os, fileIO, csv
#from scipy.stats import signaltonoise

def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array.
    Sample with replacement.
    From analysis/sampling.py.
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample

    
def calculate_waveform(exptpath=None, clustersToAnalyze=None, numChannels=128, total_waveforms=100, returnWaveforms = False, save = True, saveFileDir = None, saveFileName='mean_waveforms.npy', channelMap = None):
    """Re-calculate waveforms for sorted clusters from raw data.
    Bootstrap for units with more than 100 spikes.
    n=100
    boots=100
    
    clustersToAnalyze should be None (default) or string (eg 'good') or a list of strings (['good', 'mua'])
    specifying which clusters to analyze based on the manual sort labels. If None, all 
    clusters are analyzed.
    """
    if exptpath is None:
        exptpath = fileIO.getDir()
    
    datFileName = [d for d in os.listdir(exptpath) if d.endswith('.dat')][0]     
    rawDataFile = os.path.join(exptpath, datFileName)
    clustersFile = os.path.join(exptpath,'spike_clusters.npy')
    spikeTimesFile = os.path.join(exptpath,'spike_times.npy')
    clusterAssignmentFile = os.path.join(exptpath, 'cluster_groups.csv')
    waveformsFile = os.path.join(exptpath, saveFileName)
    
    if channelMap is None:
        channelMap = np.arange(numChannels)
        
        
    samplesPerSpike = 82
    preSamples = 20

    rawData = np.memmap(rawDataFile, dtype='int16', mode='r')
    data = np.reshape(rawData, (rawData.size/numChannels, numChannels))

    clusters = np.load(clustersFile)
    spike_times = np.load(spikeTimesFile)
    cluster_nums = np.unique(clusters)
    
    if clusters.size > spike_times.size:
        print 'Cluster assignments outnumber spike times. Taking subset.'
        clusters = clusters[:spike_times.size]
    
    #make dictionary with cluster nums as keys and assigments (eg 'good' or 'mua') as values 
    cafReader = csv.reader(open(clusterAssignmentFile), delimiter='\t')
    clusterAssignments = {}
    for row in cafReader:
        clusterAssignments[row[0]] = row[1]
    
    mean_waveforms = np.zeros((np.max(clusters)+1,samplesPerSpike,numChannels))
    waveforms_dict = {}
    for cluster_idx, cluster_num in enumerate(cluster_nums):

        clusterAssignment = clusterAssignments[str(cluster_num)]        
        if clustersToAnalyze is None or clusterAssignment in clustersToAnalyze:
            in_cluster = np.where(clusters == cluster_num)[0]
            times_for_cluster = spike_times[in_cluster]
                
            if times_for_cluster.size > total_waveforms:
                TW = total_waveforms
            else:
                TW = times_for_cluster.size
                
            boots = 100
            if times_for_cluster.size > total_waveforms:
                print('Analyzing cluster: ' + str(cluster_num) + ', ' + str(cluster_idx+1) + ' of ' + str(cluster_nums.size))
                waveform_boots = np.zeros((boots,samplesPerSpike, numChannels))
                SNR_boots=np.zeros((boots,samplesPerSpike, numChannels))
                for i in range(boots):
                    times_boot = bootstrap_resample(times_for_cluster,n=total_waveforms)
                    waveforms = np.zeros((samplesPerSpike, numChannels, TW))
                    badSpikes = []
                    for wv_idx in range(0, TW):
                        peak_time = times_boot[wv_idx][0]
                        rawWaveform = data[int(peak_time-preSamples):int(peak_time+samplesPerSpike-preSamples),:]
                        if rawWaveform.shape[0]<samplesPerSpike:
                            badSpikes.append(wv_idx)
                            continue
                        else:
                            normWaveform = rawWaveform - np.tile(rawWaveform[0,:],(samplesPerSpike,1))
                            waveforms[:, :, wv_idx] = normWaveform
                    if len(badSpikes)>0:
                        waveforms = waveforms[:, :, np.setdiff1d(np.arange(TW), badSpikes)]
                    SNR_boots[i,:,:]=signaltonoise(waveforms, axis=2)
                    waveform_boots[i,:,:]=np.mean(waveforms,2)   
                mean_waveforms[cluster_num, :, :] = np.squeeze(np.mean(waveform_boots,0))[:, channelMap]
                SNR=np.squeeze(np.mean(SNR_boots,0))[:, channelMap]    
                waveforms_dict[str(cluster_num)] = {'waveform': np.squeeze(np.mean(waveform_boots,0))[:, channelMap], 'SNR': np.squeeze(np.mean(SNR_boots,0))[:, channelMap] }
            else:
                waveforms = np.zeros((samplesPerSpike, numChannels, TW))
                badSpikes = []
                for wv_idx in range(0, TW):
                    peak_time = times_for_cluster[wv_idx][0]
                    rawWaveform = data[int(peak_time-preSamples):int(peak_time+samplesPerSpike-preSamples),:]
                    if rawWaveform.shape[0]<samplesPerSpike:
                        badSpikes.append(wv_idx)
                        continue
                    else:
                        normWaveform = rawWaveform - np.tile(rawWaveform[0,:],(samplesPerSpike,1))
                        waveforms[:, :, wv_idx] = normWaveform
                if len(badSpikes)>0:
                    waveforms = waveforms[:, :, np.setdiff1d(np.arange(TW), badSpikes)]
                mean_waveforms[cluster_num, :, :] = np.mean(waveforms,2)[:, channelMap]
                SNR=signaltonoise(waveforms, axis=2)[:, channelMap] 
                waveforms_dict[str(cluster_num)] = {'waveform': np.mean(waveforms,2)[:, channelMap], 'SNR': signaltonoise(waveforms, axis=2)[:, channelMap]}

    mean_waveforms =  mean_waveforms[np.max(mean_waveforms, axis=(1,2))>0]
    if save:    
        if saveFileDir is None:
            np.save(waveformsFile, mean_waveforms, SNR)
        else:
            np.save(os.path.join(saveFileDir, saveFileName), mean_waveforms, SNR)
        
    if returnWaveforms:
        return waveforms_dict
        


