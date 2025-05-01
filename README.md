
This note contains the code for the paper "**AdPrST:An adversarial graph deep learning pre-clustering framework for deciphering spatiotemporal structures in spatially resolved transcriptomics**".

## Introduction

AdPrST


## Installation  
AdPrST is implemented using Python 3.9.21 and Pytorch 1.13.0.    


### Requirements
torch==1.13.0+CUDA117  
scanpy=1.9.1
numpy=1.22.0
pandas=1.5.0
sklearn=1.1.1
scipy=1.9.1
squidpy=1.2.2
rpy2=3.5.11

## Datasets
All datasets used in this paper are publicly available. Users can download them from the links below.

### DLPFC
http://research.libd.org/spatialLIBD/

### MOSTA
https://db.cngb.org/stomics/mosta/

### STARmap
https://github.com/SunXQlab/PearlST

### Mouse Olfactory Bulb(Slide-SeqV2)
https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary

### Mouse Olfactory Bulb(Stereo-Seq)
https://github.com/JinmiaoChenLab/SEDR_analyses/

### Human breast cancer  
https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1  

### Coronal mouse brain
https://squidpy.readthedocs.io

## Tutorial

Here, we present two examples to illustrate the application of AdPrST for spatial domain identification. We employed two datasets for this demonstration: **the Human Breast Cancer dataset** and **the Mouse Olfactory Bulb dataset**. 

### Human breast cancer  
```python
# Importing necessary libraries
import os
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["R_HOME"] = "D:/R-4.4.1" # Your R 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
from Model import FusionAdversarialFramework
from Train import train_Multi_view_Adversarial
from SpatialAdj import Transfer_pytorch_Data, mclust_R,SpatialNetConstruction

# Loading spatial transcriptomics data using Scanpy
adata = sc.read("Data/10X/HBC.h5ad")

# Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# SpatialNet
SpatialNetConstruction(adata,k_cutoff=6, rad_cutoff=150) 

# Train
adata = train_Multi_view_Adversarial(adata,n_epochs=800,lr=0.008,alpha=0.9, beta=0.1,gamma=0.1,delta=0.1,n_clusters=70)

# Clustering and Evaluation
adata = mclust_R(adata, num_cluster=20,used_obsm='AdPrST' )
obs_df = adata.obs.dropna()

ARI = adjusted_rand_score(obs_df['mclust'], obs_df['ground_truth'])
print('ARI = %.2f' %ARI)

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(12,4),constrained_layout=True)
sc.pl.embedding(adata, basis="spatial",  color='ground_truth', size=50, ax=axs[0],show=False)
axs[0].spines['right'].set_visible(False) 
axs[0].spines['top'].set_visible(False)   
axs[0].spines['left'].set_visible(False) 
axs[0].spines['bottom'].set_visible(False)   
axs[0].get_yaxis().set_visible(False)
axs[0].get_xaxis().set_visible(False)
axs[0].set_title('Mannal annotation', fontsize=12)


sc.pl.embedding(adata, basis="spatial",  color='mclust', size=50, ax=axs[1],show=False)
axs[1].spines['right'].set_visible(False) 
axs[1].spines['top'].set_visible(False)   
axs[1].spines['left'].set_visible(False) 
axs[1].spines['bottom'].set_visible(False)   
axs[1].get_yaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)
axs[1].set_title('AdPrST', fontsize=12)

```



### Mouse Olfactory Bulb(Slide-SeqV2)
First, load the required packages and data.

```Python
# Importing necessary libraries
import os
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["R_HOME"] = "D:/R-4.4.1"
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
from Model import FusionAdversarialFramework
from Train import train_Multi_view_Adversarial
from SpatialAdj import Transfer_pytorch_Data, mclust_R,SpatialNetConstruction

# Loading spatial transcriptomics data using Scanpy
adata = sc.read("Data/SlideSeqV2/MOB.h5ad")
sc.pp.filter_genes(adata, min_cells=50)

# Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# SpatialNet
SpatialNetConstruction(adata,k_cutoff=3, rad_cutoff=50) 

# Train
adata = train_Multi_view_Adversarial(adata,n_epochs=400,lr=0.008,alpha=0.1, beta=0.1,gamma=0.1,delta=0.1,n_clusters=40)

sc.pp.neighbors(adata, use_rep='AdPrST')
sc.tl.umap(adata)

# Clustering
adata = mclust_R(adata, num_cluster=11,used_obsm='AdPrST' )

# Visualization
plot_colors = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#79706e"
]
plt.rcParams["figure.figsize"] = (6, 6)
sc.pl.embedding(adata, basis="spatial", color="mclust",size=50,palette=plot_colors, show=False, title='AdPrST')
plt.axis('off')

# Evaluation
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(adata.obsm['AdPrST'], adata.obs['mclust'])
print("SC-Score:", silhouette_avg)

from sklearn.metrics import davies_bouldin_score

davies_bouldin_score = davies_bouldin_score(adata.obsm['AdPrST'], adata.obs['mclust'])
print("Davies-Bouldin:", davies_bouldin_score)

# pSM inference
from SpatialAdj import pseudo_Spatiotemporal_Map
pseudo_Spatiotemporal_Map(adata, emb_name='AdPrST', n_neighbors=30, resolution=1.0)
plt.rcParams["figure.figsize"] = (6, 6)
sc.pl.embedding(adata_AdPrST, basis="spatial", color="pSM_values",size=60,show=False,cmap = 'summer', title='AdPrST pSM')
plt.axis('off')

# PAGA inference
plot_colors = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#79706e"
]
cluster_anno={'1':'GCL_2','2':'GL_1','3':'AOBgr','4':'GL_2','5':'ONL','6':'IPL','7':'GCL_1','8':'AOB','9':'EPL','10':'RMS','11':'MCL'}
adata.obs['domain_type']=adata.obs['mclust'].astype('str').map(cluster_anno)
sc.tl.paga(adata,groups='domain_type')
plt.rcParams["figure.figsize"] = (6,6)
sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,
                   title='AdPrST', legend_fontoutline=2, palette=plot_colors,show=False)

```


## Citation

Coming Soon....
