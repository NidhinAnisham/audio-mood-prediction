# -*- coding: utf-8 -*-
"""

Divya Machenahalli Lokesh, DXM190018
Nidhin Anisham, NXA190000
    
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pyplot as plt

data_normalized = pd.read_csv("MusicData.csv")
sse = {}

# Fit KMeans and calculate sum of squared error (SSE) for each k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data_normalized)
    sse[k] = kmeans.inertia_

plt.title('The Elbow Method')
plt.xlabel('k'); plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()
  