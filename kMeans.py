# -*- coding: utf-8 -*-
"""

Divya Machenahalli Lokesh, DXM190018
Nidhin Anisham, NXA190000
    
"""

import numpy as np
import pandas as pd
import random
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot
from sklearn.decomposition import PCA
import pickle

def plot2d(data,clusterLabels,k=4):
    if k>10:
        print("Number of clusters too high")
        return
    
    pca_2d = PCA(n_components=2)
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(data.drop(["Cluster"], axis=1)))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    plotX = pd.concat([data,PCs_2d], axis=1, join='inner')
    plotX["dummy"] = 0

    trace = []
    colors = ['red','blue','yellow','pink','green','purple','black','orange','brown','lime']
    for i in range(k):
        trace.append(   go.Scatter(
                        x = plotX[plotX["Cluster"] == i]["PC1_2d"],
                        y = plotX[plotX["Cluster"] == i]["PC2_2d"],
                        mode = "markers",
                        name = clusterLabels[i],
                        marker = dict(color = colors[i]),
                        text = None) )
    
    init_notebook_mode(connected=True)
    title = "Visualizing Clusters in Two Dimensions Using PCA"
    layout = dict(title = title,
                  xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                  yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
                 )
    
    fig = dict(data = trace, layout = layout)
    
    plot(fig)


def getClusterLabel(final_cluster_label):
    cluster_labels={}
    for i in range(len(final_cluster_label)):
        temp=np.array(final_cluster_label[i])
        labels,counts=np.unique(temp, return_counts=True)
        index=np.argmax(counts)
        cluster_labels[i]=labels[index]
        
    with open("Kmeans_labels.dat", "wb") as f:
        pickle.dump(cluster_labels, f)
        
    return cluster_labels
        
def kMeans(X, labels,k=4, iterations=300):
   
    music_data = X.to_numpy()
    no_of_clusters = k
    total_var=float('inf')
    centroid=[]
    
    
    for i in range(0,no_of_clusters):
        index = random.randint(0,len(music_data)-1)
        centroid.append(music_data[index])

    for a in range(iterations):
        clusters = [[] for _ in range(no_of_clusters)]
        distance_cluster = [[] for _ in range(no_of_clusters)]
        clusters_music = [[] for _ in range(no_of_clusters)]
        
        for i in range(len(music_data)):
            min_distance=float('inf')
            for l in range(len(centroid)):
                cluster = centroid[l]
                point = music_data[i]
                distance = 0
                for k in range(len(point)):
                    distance += (point[k]-cluster[k])**2

                if min_distance>distance:
                    music_cluster=l
                    min_distance=distance
            
            clusters[music_cluster].append(i)
            clusters_music[music_cluster].append(music_data[i])
            distance_cluster[music_cluster].append(min_distance)
        
        variance=0        
        
        for i in distance_cluster:
            variance += np.var(i)
            
        if total_var>variance:
            final_cluster=clusters
            final_clusters_music = clusters_music
            total_var=variance

        for i in range(len(clusters_music)):
            centroid[i]=np.sum(clusters_music[i],axis=0)/len(clusters_music[i])
            
    '''For creating labels'''       
    files = open('MusicList.txt', 'r',encoding = 'utf-8') 
    lists = files.readlines()  
    files.close()
    music_list = []    
    for music in lists: 
        music_list.append(music.strip()[6:])  
        
        
    final_list=[]
    for i in final_cluster:
        name=[]
        for j in i:
            music_name=music_list[j].split('/')
            if music_name[0].strip() in labels: 
                name.append(music_name[0])
        final_list.append(name)
    
    clusterLabels = getClusterLabel(final_list)
    print(clusterLabels) 
        
    '''end of getting labels'''     
    data = []
    for i in range(len(final_clusters_music)):
        for j in final_clusters_music[i]:
            record = np.append(j,i)
            data.append(record)
    
    data = pd.DataFrame(data)
    data.columns = ["G#","G","F#","F","E","D#","D","C#","C","B","A#","A","Cluster"]
    data = data.sample(frac=1).reset_index(drop=True)
    return data,clusterLabels
    
if __name__=='__main__':
    
    X = pd.read_csv("MusicData.csv")
    labels=['happy','sad','romantic','party']
    
    print("Training Model...")
    labelledData,clusterLabels = kMeans(X,labels)
    labelledData.to_csv("LabelledMusicData.csv",index=False)
    
    print("Model Created! Labelled Data stored in 'LabelledMusicData.csv'")       
    plot2d(labelledData,clusterLabels)         