# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:45:56 2019

@author: svc_ccg
"""

import probeData as pd
import popProbeData as ppd
import numpy as np
import os

annotationDataFile = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\CCF\annotation_25.nrrd"
annotationStructures = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\CCF\annotationStructures.xml"

pp = ppd.popProbeData()
pp.getAnnotationData(annoFile=annotationDataFile, structFile=annotationStructures)

datadir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\LP dataframes\popAnalysis\LP"
goodFiles = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]


hdf5Files = goodFiles
saveDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\LP dataframes\popAnalysis\LP"

for h in hdf5Files:
    p = pd.probeData()
    p.loadHDF5(os.path.join(datadir, h))
    
    channelCoords = p.findChannelCCFCoords()
    channelRegions = [pp.findRegions(ccf) for ccf in channelCoords]
    
    np.savez(os.path.join(saveDir, h[:15]+'_channelCoords.npz'), ccfCoords=channelCoords, ccfRegion=channelRegions)