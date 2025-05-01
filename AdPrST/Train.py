
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from Model import FusionAdversarialFramework
from SpatialAdj import Transfer_pytorch_Data

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from torch import nn
from sklearn.decomposition import PCA
import ot
import scanpy as sc
import random

def get_initial_label(adata, n_clusters, refine=True, method="mclust"):
    features = adata.X
    if type(features) == np.ndarray:
        features = features
    else:
        features = features.todense()
    pca_input = dopca(features, dim = 20)
    if method == "mclust":
        pred = mclust_RR(embedding=pca_input, num_cluster=n_clusters)
    if method == "louvain":
        adata.obsm["pca"] = pca_input
        sc.pp.neighbors(adata, n_neighbors=50, use_rep="pca")
        sc.tl.louvain(adata, resolution=n_clusters, random_state=0)
        pred=adata.obs['louvain'].astype(int).to_numpy()
    if refine:
        pred = refine_label(pred, adata.obsm["spatial"], radius=60)
    pred = list(map(int, pred))
    return np.array(pred)

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim, random_state=42)
    X = np.asarray(X)  # 转换为 numpy ndarray
    X_10 = pcaten.fit_transform(X)
    return X_10

def mclust_RR(embedding, num_cluster, modelNames='EEE', random_seed=0):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(
        embedding), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    mclust_res = mclust_res.astype('int')
    # mclust_res = mclust_res.astype('category')
    return mclust_res



def refine_label(label, position, 
                 radius=50):
    new_type = []

    # calculate distance
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, radius+1):
            neigh_type.append(label[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


def train_Multi_view_Adversarial(
    adata, 
    hidden_dims=[512, 30], 
    n_epochs=1000, 
    lr=0.008, 
    key_added='AdPrST',
    gradient_clipping=5., 
    weight_decay=0.0001, 
    verbose=True, 
    random_seed=0, 
    save_loss=False, 
    save_reconstruction=False, 
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    dim_input=3000, 
    dim_output=64, 
    alpha=0.9, 
    beta=0.1,
    gamma=0.1,
    delta=0.1, 
    deconvolution=False,
    n_clusters=40,
    cluster_method="mclust"
):  

    
    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
   
    loss_func = nn.CrossEntropyLoss()
    adata.X = sp.csr_matrix(adata.X)
    
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Construct two kinds of spatial neighborhood network first!")

    print('Size of Input: ', adata_Vars.shape)
    add_contrastive_label(adata)
    get_feature(adata) 
    
    
    ss_labels = get_initial_label(adata, method=cluster_method, n_clusters=n_clusters)
    if np.min(ss_labels) == 1:
        ss_labels = ss_labels - 1
    ss_labels = torch.tensor(ss_labels, dtype=torch.int64).to(device)

    features = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)
    features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(device)
    label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)
    adj1 = adata.obsm['adj1']
    adj2 = adata.obsm['adj2']
    graph_neigh1 = torch.FloatTensor(adata.obsm['graph_neigh1'].copy() + np.eye(adj1.shape[0])).to(device)
    graph_neigh2 = torch.FloatTensor(adata.obsm['graph_neigh2'].copy() + np.eye(adj2.shape[0])).to(device)
    
    adj1 = preprocess_adj(adj1) 
    adj1 = torch.FloatTensor(adj1).to(device)
    adj2 = preprocess_adj(adj2) 
    adj2 = torch.FloatTensor(adj2).to(device)
     
    classifier = nn.Sequential(
    nn.Linear(dim_input, 128, bias=False),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.1),  # 增加 Dropout 率
    nn.Linear(128, 64, bias=False),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, n_clusters, bias=False),
    nn.Sigmoid()
    ).to(device)


    model = FusionAdversarialFramework(dim_input, dim_output).to(device)
    
    loss_CSL = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr, weight_decay=weight_decay)
    
    print('Begin to train...')

    # 初始化损失记录
    loss_feat_list = []
    loss_sl_1_list = []
    loss_sl_2_list = []
    reg_loss_list = []
    total_loss_list = []
    loss_cls_list = []

    for epoch in tqdm(range(n_epochs)): 
        model.train()
        classifier.train()
        
        features_a = permutation(features) 
        hiden_feat, emb, ret, ret_a, reg_loss = model(features, features_a, adj1, adj2, graph_neigh1, graph_neigh2)
        
        #print(emb.shape)  # 应输出 torch.Size([3635, 64])
        # 计算分类器的输出
        output_cls = classifier(emb)
        loss_cls = loss_func(output_cls, ss_labels)

        # 计算其他损失
        loss_sl_1 = loss_CSL(ret, label_CSL)
        loss_sl_2 = loss_CSL(ret_a, label_CSL)
        loss_feat = F.mse_loss(features, emb)
        
        # 综合损失
        loss = alpha * loss_feat + beta * (loss_sl_1 + loss_sl_2) + gamma * reg_loss + delta*loss_cls  # 综合损失
        
        optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        classifier_optimizer.step()
        
        loss_feat_list.append(loss_feat.item())
        loss_sl_1_list.append(loss_sl_1.item())
        loss_sl_2_list.append(loss_sl_2.item())
        reg_loss_list.append(reg_loss.item())
        loss_cls_list.append(loss_cls.item())
        total_loss_list.append(loss.item())
        
    with torch.no_grad():
        model.eval()
        classifier.eval()
        if deconvolution:
            emb_rec = model(features, features_a, adj1, adj2)[1]
        else:
            emb_rec = model(features, features_a, adj1, adj2, graph_neigh1, graph_neigh2)[1].detach().cpu().numpy()
            
        adata.obsm[key_added] = emb_rec  
        if save_loss:
            adata.uns['SpaMAG_loss_feat'] = loss_feat_list
            adata.uns['SpaMAG_loss_sl_1'] = loss_sl_1_list
            adata.uns['SpaMAG_loss_sl_2'] = loss_sl_2_list
            adata.uns['SpaMAG_reg_loss'] = reg_loss_list
            adata.uns['SpaMAG_loss_cls'] = loss_cls_list
            adata.uns['SpaMAG_total_loss'] = total_loss_list
        if save_reconstruction:
            ReX = emb.to('cpu').detach().numpy()
            ReX[ReX < 0] = 0
            adata.layers['SpaMAG_ReX'] = ReX  

    return adata

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()
    
def preprocess_adj(adj):
    #Preprocessing of adjacency matrix of simple GCN model and conversion of tuple representation
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def permutation(feature):
    #Feature random arrangement
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids) 
    feature_permutated = feature[ids]
    return feature_permutated 
    
def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL 
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']] 
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    # data augmentation
    feat_a = permutation(feat) 
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    