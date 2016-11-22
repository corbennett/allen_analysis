# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:47:32 2016

@author: Gale
"""

import h5py
import numpy as np
from PyQt4 import QtGui


def getFile(caption='Choose File',rootDir='',fileType=''):
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return str(QtGui.QFileDialog.getOpenFileName(None,caption,rootDir,fileType))


def getFiles(caption='Choose File',rootDir='',fileType=''):
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    filePaths = QtGui.QFileDialog.getOpenFileNames(None,caption,rootDir,fileType)
    return [str(f) for f in filePaths]

    
def getDir(caption='Choose Directory',rootDir=''):
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return str(QtGui.QFileDialog.getExistingDirectory(None,caption,rootDir))
    

def saveFile(caption='Save As',rootDir='',fileType=''):
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    return str(QtGui.QFileDialog.getSaveFileName(None,caption,rootDir,fileType))


def objToHDF5(obj, filePath=None, fileOut=None, saveDict=None, grp=None):
    if filePath is None and fileOut is None:
        filePath = saveFile(fileType='*.hdf5')
        if filePath=='':
            return
        fileOut = h5py.File(filePath, 'w')
    elif filePath is not None and fileOut is None:            
        fileOut = h5py.File(filePath,'w')

    if saveDict is None:
        saveDict = obj.__dict__
    if grp is None:    
        grp = fileOut['/']
    
    for key in saveDict:
        if key[0]=='_':
            continue
        elif type(saveDict[key]) is dict:
            objToHDF5(obj, fileOut=fileOut, saveDict=saveDict[key], grp=grp.create_group(key))
        else:
            try:
                grp.create_dataset(key,data=saveDict[key],compression='gzip',compression_opts=1)
            except:
                try:
                    grp[key] = saveDict[key]
                except:
                    try:
                        grp.create_dataset(key,data=np.array(saveDict[key],dtype=object),dtype=h5py.special_dtype(vlen=str))
                    except:
                        print('Could not save: ', key)
                
                
def hdf5ToObj(obj, filePath=None, grp=None, loadDict=None):
    if filePath is None and grp is None:        
        filePath = getFile(fileType='*.hdf5')
        if filePath=='':
            return
    if grp is None:
        grp = h5py.File(filePath)
    for key,val in grp.items():
        if isinstance(val,h5py._hl.dataset.Dataset):
            v = val.value
            if isinstance(v,np.ndarray) and v.dtype==np.object:
                v = v.astype('U')
            if loadDict is None:
                setattr(obj,key,v)
            else:
                loadDict[key] = v
        elif isinstance(val,h5py._hl.group.Group):
            if loadDict is None:
                setattr(obj,key,{})
                hdf5ToObj(obj,grp=val,loadDict=getattr(obj,key))
            else:
                loadDict[key] = {}
                hdf5ToObj(obj,grp=val,loadDict=loadDict[key])
                
                
if __name__=="__main__":
    pass 