"""
Land Development Model (LDM) code developed by Michael Batty and Fulvio D. Lopane
Centre for Advanced Spatial Analysis
University College London

This module generates the dendogram and the visualisation of the input layers correlation matrix

Started developing in May 2024
"""

########################################################################################################################
# Import phase
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from config import *

########################################################################################################################
df = pd.read_csv(inputs["correlation_matrix_csv"])

correlations = df.corr()
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.tick_params(left=False, bottom=False)

sns.heatmap(round(correlations, 1), cmap='gray', annot_kws={"size": 7}, vmin=0, vmax=1, linewidths=.5)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.tick_params(left=False, bottom=False)

dissimilarity = 1 - abs(correlations)

sns.heatmap(dissimilarity, cmap='gray', annot_kws={"size": 7}, vmin=0, vmax=1, linewidths=.5)
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.tick_params(left=False, bottom=False)

Z = linkage(squareform(dissimilarity), 'complete')

plt.yticks(visible=False)

dendrogram(Z, orientation='top', leaf_rotation=90)
plt.savefig(outputs["Dendrogram"])

threshold = 0.8
labels = fcluster(Z, threshold, criterion='distance')
labels_order = np.argsort(labels)

for idx, i in enumerate(df.columns[labels_order]):
    if idx == 0:
        clustered = pd.DataFrame(df[i])
    else:
        df_to_append = pd.DataFrame(df[i])
        clustered = pd.concat([clustered, df_to_append], axis=1)

plt.figure(figsize=(6,5))
dissimilarity = clustered.corr()
plt.yticks(visible=False)
sns.heatmap(dissimilarity, cmap='gray', annot=False, annot_kws={"size": 7}, vmin=0, vmax=1, linewidths=.5)
plt.savefig(outputs["Correlation_matrix"])

sns.clustermap(correlations, method="complete", cmap='RdBu', annot_kws={"size": 7}, vmin=0, vmax=1, figsize=(6,6), linewidths=.5)