# -*- coding: utf-8 -*-
"""
Created on Mon Aug 01 15:42:27 2016

@author: SVC_CCG
"""
import os, csv
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
color50 = ["#67572e",
"#272d38",
"#e47689",
"#7499db",
"#64e251",
"#4553a8",
"#e55728",
"#325338",
"#da94d7",
"#591d42",
"#d3e571",
"#c7af44",
"#5e3ecc",
"#7f75df",
"#9d5258",
"#a3e841",
"#cb48b4",
"#4a6890",
"#c3e19a",
"#77606d",
"#50a874",
"#e1e53b",
"#68e0ca",
"#ac2b51",
"#cf9894",
"#829b44",
"#e54150",
"#da4687",
"#382f1c",
"#927933",
"#73c5dc",
"#dc865f",
"#925991",
"#e8b12e",
"#b22d25",
"#518c8e",
"#3d6e2a",
"#572282",
"#55ad3d",
"#cf832e",
"#8a9675",
"#dabd88",
"#62221e",
"#6fe594",
"#9ab92f",
"#312557",
"#b74cdf",
"#994923",
"#c1b4d1",
"#c5dac7"]




#units = p.units
#
#fig,ax = plt.subplots(1,1,figsize=(0.8,10))
##ax.imshow('imec_p2.png',alpha=0.2)
#for i,u in enumerate(units.keys()):
#    circle=plt.Circle((units[u]['xpos'],units[u]['ypos']*-1),10,color=color50[i%50],alpha=0.8)
#    ax.add_artist(circle)
#    ax.text(units[u]['xpos'], units[u]['ypos']*-1, u, ha='center')
#ax.set_ylim(0,1500)
#ax.set_xlim(0,70)
##ephys.cleanAxes(ax,total=True)

imec_p2_positions = np.zeros((128,2))
imec_p2_positions[:,0][::2] = 18
imec_p2_positions[:,0][1::2] = 48
imec_p2_positions[:,1] = np.floor(np.linspace(0,128,128)/2) * 20;imec_p2_positions[:,1][-1]=1260.
#badChannels = [0 , 18, 47, 63, 79, 96, 11, 127, 128]
badChannels = [0,18,63,96]
imec_p2_positions = imec_p2_positions[np.setdiff1d(np.arange(128), badChannels), :]
def load_phy_template(path,site_positions = imec_p2_positions,**kwargs):
    # load spike data that has been manually sorted with the phy-template GUI
    # the site_positions should contain coordinates of the channels in probe space. for example, in um on the face of the probe
    # returns a dictionary of 'good' units, each of which includes:
    #	times: spike times, in seconds
    #	template: template used for matching
    #	ypos: y position on the probe, calculated from the template. requires an accurate site_positions. averages template from 100 spikes.
    #	xpos: x position on the probe, calcualted from the template. requires an accurate site_positions. averages template from 100 spikes.
    clusters = np.load(os.path.join(path,'spike_clusters.npy'))
    spikes = np.load(os.path.join(path,'spike_times.npy'))
    spike_templates = np.load(os.path.join(path,'spike_templates.npy'))
    templates = np.load(os.path.join(path,'templates.npy'))
    amplitudes = np.load(os.path.join(path, 'amplitudes.npy'))
#    channel_positions = np.load(os.path.join(path, 'channel_positions.npy'))
    cluster_id = [];
    [cluster_id.append(row) for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))];
    if 'sampling_rate' in kwargs.keys():
        samplingrate = kwargs['sampling_rate']
    else:
        samplingrate =30000.
        print 'no sampling rate specified, using default of 30kHz'
    		
    units = {}
    for i in np.arange(1,np.shape(cluster_id)[0]):
        if cluster_id[i][0].split('\t')[1] == 'good' :#:or cluster_id[i][0].split('\t')[1] == 'unsorted' :#if it is a 'good' cluster by manual sort
    		unit = int(cluster_id[i][0].split('\t')[0])
    		units[str(unit)] = {}
    			
    		#get the unit spike times
    		units[str(unit)]['times'] = spikes[np.where(clusters==unit)]
    		units[str(unit)]['times'] = units[str(unit)]['times'].flatten()
         	units[str(unit)]['amplitudes'] = amplitudes[np.where(clusters==unit)[0]]	
    		#get the mean template used for this unit
    		all_templates = spike_templates[np.where(clusters==unit)].flatten()
    		n_templates_to_subsample = 100
    		random_subsample_of_templates = templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
    		mean_template = np.mean(random_subsample_of_templates,axis=0)
    		units[str(unit)]['template'] = mean_template
    #			
		#take a weighted average of the site_positions, where the weights is the absolute value of the template for that channel
		#this gets us the x and y positions of the unit on the probe.
		weights = np.zeros(site_positions.shape)
		for channel in range(site_positions.shape[0]):
			weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
		weights = weights/np.max(weights)
		(xpos,ypos)=np.average(site_positions,axis=0,weights=weights)
		units[str(unit)]['xpos'] = xpos
		units[str(unit)]['ypos'] = ypos - site_positions[-1][1]
			
    return units
