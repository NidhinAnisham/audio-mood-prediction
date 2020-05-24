# -*- coding: utf-8 -*-
"""

Divya Machenahalli Lokesh, DXM190018
Nidhin Anisham, NXA190000
    
"""

import numpy as np
import pandas as pd
from pyAudioAnalysis import audioBasicIO #A
from pyAudioAnalysis import ShortTermFeatures #B
import matplotlib.pyplot as plt
from os import walk

def preProcess(fileName):
    
    [Fs, x] = audioBasicIO.read_audio_file(fileName) #A
    
    if( len( x.shape ) > 1 and  x.shape[1] == 2 ):
        x = np.mean( x, axis = 1, keepdims = True )
    else:
        x = x.reshape( x.shape[0], 1 )
    
    F, f_names = ShortTermFeatures.feature_extraction(x[ :, 0 ],Fs, 0.050*Fs,0.025*Fs)
    return (f_names, F)


def getChromagram(audioData):

    temp_data =  audioData[21].reshape(1,audioData[21].shape[0])
    chronograph = temp_data
    
    for i in range(22,33):
        temp_data = audioData[i].reshape(1,audioData[i].shape[0])
        chronograph = np.vstack([chronograph,temp_data])
    
    return chronograph


def getNoteFrequency(chromagram):
    
    numberOfWindows = chromagram.shape[1]  
    freqVal = chromagram.argmax(axis = 0)
    histogram, bin = np.histogram(freqVal,bins = 12)
    normalized_hist = histogram.reshape(1, 12).astype(float)/numberOfWindows
    return normalized_hist

def getDataset(files):
    X = pd.DataFrame()
    columns=[ "G#", "G", "F#", "F", "E", "D#", "D", "C#", "C", "B", "A#", "A" ]
    chromagrams = []
    noteFrequencies = []
    musicList = []
    for file in files:
        try:
            print("Processing file: "+file)
            feature_name,features = preProcess(file)
            chromagram = getChromagram(features)
            noteFrequency = getNoteFrequency(chromagram)
            x_new = pd.Series(noteFrequency[0,:])
            X = pd.concat([X,x_new],axis = 1)
            chromagrams.append(chromagram)
            noteFrequencies.append(noteFrequency)
            musicList.append(file)
        except:
            continue
        
    data = X.T.copy()
    data.columns = columns
    data.index = [i for i in range(0,data.shape[0])]
            
    return data,chromagrams,noteFrequencies,musicList


def plotHeatmap(chromagraph, smallSample = True):
    
    notesLabels = [ "G#", "G", "F#", "F", "E", "D#", "D", "C#", "C", "B", "A#", "A" ]
    
    fig, axis = plt.subplots()
    
    if smallSample:
        im = axis.imshow(chromagraph[:,0:300], cmap = "YlGn")
    else:
        im = axis.imshow(chromagraph)
        
    cbar = axis.figure.colorbar(im,ax = axis,cmap = "YlGn")
    cbar.ax.set_ylabel("Amplitude",rotation=-90, va="bottom")
    
    axis.set_yticks(np.arange(len(notesLabels)))
    axis.set_yticklabels(notesLabels)
    axis.set_title("chromagram")
    fig.tight_layout()
    _ = plt.show()
    

def noteFrequencyPlot( noteFrequency, smallSample = True ):
    
    fig, axis = plt.subplots(1, 1, sharey=True )
    axis.plot( np.arange( 1, 13 ), noteFrequency[0, :] )
    _ = plt.show()
    
if __name__ == '__main__':
    print("Extracting notes...")
    f = []
    d=[]
    mypath = "Music" #data folder
    for (dirpath, dirnames, filenames) in walk(mypath):
        d.extend(dirnames)
        break
    file_path=dirpath+'/'
    for directory in d:
        files=[]
        for (dirpath, dirnames, filenames) in walk(file_path+directory):
            files.extend(filenames)
            break
        
        f = np.hstack((f,[dirpath + '/' + s for s in files])).tolist()
        
        
    data,chromagram,noteFrequency,musicList = getDataset(f)
    data.to_csv('MusicData.csv',index=False)
    
    print("Data Extracted into 'MusicData.csv'")
    plotHeatmap(chromagram[0],False)
    noteFrequencyPlot(noteFrequency[0],False)
   
    with open('MusicList.txt', 'w', encoding='utf-8') as filehandle:
        filehandle.writelines("%s\n" % i for i in musicList)
