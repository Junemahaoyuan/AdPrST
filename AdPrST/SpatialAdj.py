
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import ot
from torch_geometric.data import Data
import scanpy as sc
import anndata



def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data

 
def SpatialNetConstruction(adata, rad_cutoff=None, k_cutoff=None):
    
    coor = pd.DataFrame(adata.obsm['spatial']) 
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    n_spot = coor.shape[0]
    
    nbrs1 = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
    distances1, indices1 = nbrs1.radius_neighbors(coor, return_distance=True)
    interaction1 = np.zeros([n_spot, n_spot]) 
    for i in range(n_spot):
        interaction1[i,indices1[i]] = 1
    adj1 = interaction1

    KNN_list1 = []
    for it in range(indices1.shape[0]):
        KNN_list1.append(pd.DataFrame(zip([it]*indices1[it].shape[0], indices1[it], distances1[it])))
    
    nbrs2 = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1,metric='chebyshev').fit(coor)#,metric='cosine'
    distances2, indices2 = nbrs2.kneighbors(coor)

    x = indices2[:, 0].repeat(k_cutoff)
    y = indices2[:, 1:].flatten()
    interaction2 = np.zeros([n_spot, n_spot])
    interaction2[x, y] = 1
    interaction2[y, x] = 1
    
    adj2 = interaction2
    adj2 = adj2 + adj2.T
    adj2 = np.where(adj2>1, 1, adj2)
   
    KNN_list2 = []
    for it in range(indices2.shape[0]):
        KNN_list2.append(pd.DataFrame(zip([it]*indices2.shape[1],indices2[it,:], distances2[it,:])))

    adata.obsm['graph_neigh1'] = interaction1
    adata.obsm['graph_neigh2'] = interaction2
    adata.obsm['adj1'] = adj1
    adata.obsm['adj2'] = adj2    
    
    KNN_df1 = pd.concat(KNN_list1)
    KNN_df2 = pd.concat(KNN_list2)
    df = pd.concat([KNN_df1,KNN_df2],ignore_index=True)
    df.columns = ['Cell1', 'Cell2', 'Distance']
    KNN_df = df.drop_duplicates(subset=["Cell1", "Cell2"],ignore_index=True)
    

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,] #It removes its own distance
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index) ))#Establish correspondence between indexes and cells
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    adata.uns['Spatial_Net'] = Spatial_Net


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='AdPrST', random_seed=2023):

    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=20)
    embedding = kpca.fit_transform(adata.obsm[used_obsm].copy())
    adata.obsm['emb_pca'] = embedding

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm['emb_pca']), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    
    return adata
    


import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import os
import scipy
import numpy as np
import pandas as pd




from scipy.spatial import distance_matrix
import numpy as np
import anndata
import scanpy as sc
import pandas as pd

def pseudo_Spatiotemporal_Map(adata_all, emb_name, n_neighbors=50, resolution=1.0):
    """
    Perform pseudo-Spatiotemporal Map for ST data
    :param adata_all: AnnData object containing all data
    :param emb_name: Name of the embedding in adata_all.obsm
    :param n_neighbors: The size of local neighborhood (default: 50)
    :param resolution: Resolution parameter for Leiden clustering (default: 1.0)
    """
    error_message = "No embedding found, please ensure you have run train() method before calculating pseudo-Spatiotemporal Map!"
    max_cell_for_subsampling = 5000
    try:
        print("Performing pseudo-Spatiotemporal Map")
        embedding = adata_all.obsm[emb_name]
        print(f"Embedding '{emb_name}' shape: {embedding.shape}")
        
        
        adata = anndata.AnnData(embedding)
        print("Created new AnnData object from reduced embedding.")
        
    
        print("Checking data quality...")
        if np.isnan(adata.X).any() or np.isinf(adata.X).any():
            raise ValueError("数据中存在 NaN 或无限值，请清洗数据。")
        
        zero_variance = np.var(adata.X, axis=0) == 0
        if zero_variance.any():
            adata = adata[:, ~zero_variance]
            print(f"Removed {zero_variance.sum()} zero-variance features.")
        else:
            print("No zero-variance features found.")
        
        sc.pp.pca(adata, n_comps=15, svd_solver='arpack')#n=50
       
        if 'X_pca' in adata.obsm:
            print(f"'X_pca' exists with shape {adata.obsm['X_pca'].shape}")
            print("First few rows of 'X_pca':")
            print(adata.obsm['X_pca'][:2])
        else:
            raise KeyError("'X_pca' not found in adata.obsm after PCA.")
        
        print("Checking 'X_pca' for NaN or infinite values...")
        if np.isnan(adata.obsm['X_pca']).any() or np.isinf(adata.obsm['X_pca']).any():
            raise ValueError("'X_pca' contains NaN or infinite values. Please clean the data.")
        else:
            print("'X_pca' is clean.")
        
        print("Computing neighbors...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
        print("Neighbors computed.")
        
     
        if 'neighbors' in adata.uns:
            neighbors_keys = adata.uns['neighbors'].keys()
            print(f"'neighbors' contains the following keys: {list(neighbors_keys)}")
            if 'connectivities_key' in adata.uns['neighbors']:
                connectivities_key = adata.uns['neighbors']['connectivities_key']
                if connectivities_key in adata.obsp:
                    connectivities = adata.obsp[connectivities_key]
                    print(f"Number of non-zero connections in the neighborhood graph: {connectivities.nnz}")
                    if connectivities.nnz == 0:
                        print("Warning: The neighborhood graph has no connections. Consider increasing 'n_neighbors'.")
                else:
                    raise KeyError(f"'{connectivities_key}' not found in adata.obsp.")
            else:
                raise KeyError("'connectivities_key' not found in adata.uns['neighbors'].")
        else:
            raise KeyError("'neighbors' not found in adata.uns after computing neighbors.")
        
      
        print("Computing UMAP...")
        sc.tl.umap(adata)
        print("UMAP computed.")
        
    
        print("Performing Leiden clustering...")
        sc.tl.leiden(adata, resolution=resolution)
        print("Leiden clustering completed.")
        print(f"Leiden clusters: {adata.obs['leiden'].unique()}")
        print(f"Number of clusters: {len(adata.obs['leiden'].unique())}")
        print(f"Categories ({len(adata.obs['leiden'].unique())}, object): {adata.obs['leiden'].cat.categories.tolist()}")
        
        adata.obs['leiden'] = adata.obs['leiden'].astype('category')

    
        print("Leiden clusters size:")
        print(adata.obs['leiden'].value_counts())
     
        missing_leiden = adata.obs['leiden'].isnull().sum()
        print(f"Number of cells with missing 'leiden' labels: {missing_leiden}")
        if missing_leiden > 0:
            raise ValueError(f"{missing_leiden} cells have missing 'leiden' labels.")
        
    
        print("Computing PAGA...")
        sc.tl.paga(adata, groups='leiden')
        print("PAGA computed.")
        
   
        if 'paga' not in adata.uns:
            raise ValueError("PAGA computation failed, 'paga' not found in adata.uns.")
        else:
            print("PAGA computation successful.")
        
        print("Subsampling data for pseudotime calculation...")
        if adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = adata.X
            print(f"Using all {adata.shape[0]} cells for subsampling.")
        else:
            indices = np.arange(adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = adata.X[selected_ind, :]
            print(f"Subsampled {max_cell_for_subsampling} cells for pseudotime calculation.")
   
        print("Computing distance matrix and identifying root node...")
        sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
        root_idx = np.argmax(sum_dists)
        adata.uns['iroot'] = root_idx
        print(f"Root index for DPT: {root_idx}")
        
        print("Computing Diffmap...")
        sc.tl.diffmap(adata)
        print("Diffmap computed.")
        
        print("Computing DPT...")
        sc.tl.dpt(adata)
        print("DPT computed.")
    
        pSM_values = adata.obs['dpt_pseudotime'].to_numpy()
        print("Extracted pseudotime values.")
        
        adata_all.obs['pSM_values'] = pSM_values
        print("Assigned pseudotime values to adata_all.obs.")
        print("pseudo-Spatiotemporal Map(pSM) calculation complete!")
        
    except (NameError, AttributeError) as e:
        print(error_message)
        print(e)
    except KeyError as ke:
        print("KeyError occurred:")
        print(ke)
    except ValueError as ve:
        print("ValueError occurred:")
        print(ve)
    except Exception as e:
        print("An unexpected error occurred:")
        print(e)


def prepare_figure_multi_5(rsz=4., csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Prepare the figure and axes given the configuration
    :param rsz: row size of the figure in inches, default: 4.0
    :type rsz: float, optional
    :param csz: column size of the figure in inches, default: 4.0
    :type csz: float, optional
    :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type wspace: float, optional
    :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type hspace: float, optional
    :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
    :type left: float, optional
    :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
    :type right: float, optional
    :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
    :type bottom: float, optional
    :param top: the top position of the subplots of the figure in fraction, default: 0.9
    :type top: float, optional
    """
    fig, axs = plt.subplots(1, 5, figsize=(csz, rsz))
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)
    return fig, axs

def prepare_figure_multi_4(rsz=4., csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Prepare the figure and axes given the configuration
    :param rsz: row size of the figure in inches, default: 4.0
    :type rsz: float, optional
    :param csz: column size of the figure in inches, default: 4.0
    :type csz: float, optional
    :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type wspace: float, optional
    :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type hspace: float, optional
    :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
    :type left: float, optional
    :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
    :type right: float, optional
    :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
    :type bottom: float, optional
    :param top: the top position of the subplots of the figure in fraction, default: 0.9
    :type top: float, optional
    """
    fig, axs = plt.subplots(1, 4, figsize=(csz, rsz))
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)
    return fig, axs

def plot_pSM_multi_5(adata, adata1, adata2, adata3, adata4, scatter_sz=1., rsz=4.,
             csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Plot the domain segmentation for ST data in spatial
    :param pSM_figure_save_filepath: the default save path for the figure
    :type pSM_figure_save_filepath: class:`str`, optional, default: "./Spatiotemporal-Map.pdf"
    :param colormap: The colormap to use. See `https://www.fabiocrameri.ch/colourmaps-userguide/` for name list of colormaps
    :type colormap: str, optional, default: roma
    :param scatter_sz: The marker size in points**2
    :type scatter_sz: float, optional, default: 1.0
    :param rsz: row size of the figure in inches, default: 4.0
    :type rsz: float, optional
    :param csz: column size of the figure in inches, default: 4.0
    :type csz: float, optional
    :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type wspace: float, optional
    :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type hspace: float, optional
    :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
    :type left: float, optional
    :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
    :type right: float, optional
    :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
    :type bottom: float, optional
    :param top: the top position of the subplots of the figure in fraction, default: 0.9
    :type top: float, optional
    """
    error_message = "No pseudo Spatiotemporal Map data found, please ensure you have run the pseudo_Spatiotemporal_Map() method."
    # try:
    fig, ax = prepare_figure_multi_5(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right,
                                      bottom=bottom, top=top)
    x0, y0 = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    ax[0].scatter(x0, y0, s=scatter_sz, c=adata.obs['pSM_values'], cmap='summer', marker=".")
    ax[0].invert_yaxis()
    # clb = fig.colorbar(st0)
    # clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax[0].set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    ax[0].spines['right'].set_visible(False)  # 去掉边框
    ax[0].spines['top'].set_visible(False)  # 去掉边框
    ax[0].spines['left'].set_visible(False)  # 去掉边框
    ax[0].spines['bottom'].set_visible(False)  # 去掉边框
    ax[0].get_yaxis().set_visible(False)
    ax[0].get_xaxis().set_visible(False)
    # ax[0].set_title("SpaceFlow pSM", fontsize=12)
    ax[0].set_facecolor("none")

    x1, y1 = adata1.obsm["spatial"][:, 0], adata1.obsm["spatial"][:, 1]
    ax[1].scatter(x1, y1, s=scatter_sz, c=adata1.obs['pSM_values'], cmap='summer', marker=".")
    ax[1].invert_yaxis()
    ax[1].spines['right'].set_visible(False)  # 去掉边框
    ax[1].spines['top'].set_visible(False)  # 去掉边框
    ax[1].spines['left'].set_visible(False)  # 去掉边框
    ax[1].spines['bottom'].set_visible(False)  # 去掉边框
    ax[1].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    # clb = fig.colorbar(st1)
    # clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax[1].set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax[1].set_title("STAGATE pSM", fontsize=12)
    ax[1].set_facecolor("none")

    x2, y2 = adata2.obsm["spatial"][:, 0], adata2.obsm["spatial"][:, 1]
    ax[2].scatter(x2, y2, s=scatter_sz, c=adata2.obs['pSM_values'], cmap='summer', marker=".")
    ax[2].invert_yaxis()
    ax[2].spines['right'].set_visible(False)  # 去掉边框
    ax[2].spines['top'].set_visible(False)  # 去掉边框
    ax[2].spines['left'].set_visible(False)  # 去掉边框
    ax[2].spines['bottom'].set_visible(False)  # 去掉边框
    ax[2].get_yaxis().set_visible(False)
    ax[2].get_xaxis().set_visible(False)
    # clb = fig.colorbar(st2)
    # clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax[2].set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax[2].set_title("SEDR pSM", fontsize=12)
    ax[2].set_facecolor("none")

    x3, y3 = adata3.obsm["spatial"][:, 0], adata3.obsm["spatial"][:, 1]
    ax[3].scatter(x3, y3, s=scatter_sz, c=adata3.obs['pSM_values'], cmap='summer', marker=".")
    ax[3].invert_yaxis()
    ax[3].spines['right'].set_visible(False)  # 去掉边框
    ax[3].spines['top'].set_visible(False)  # 去掉边框
    ax[3].spines['left'].set_visible(False)  # 去掉边框
    ax[3].spines['bottom'].set_visible(False)  # 去掉边框
    ax[3].get_yaxis().set_visible(False)
    ax[3].get_xaxis().set_visible(False)
    # clb = fig.colorbar(st3)
    # clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax[2].set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax[3].set_title("Scanpy pSM", fontsize=12)
    ax[3].set_facecolor("none")

    x4, y4 = adata4.obsm["spatial"][:, 0], adata4.obsm["spatial"][:, 1]
    ax[4].scatter(x4, y4, s=scatter_sz, c=adata4.obs['pSM_values'], cmap='summer', marker=".")
    ax[4].invert_yaxis()
    ax[4].spines['right'].set_visible(False)  # 去掉边框
    ax[4].spines['top'].set_visible(False)  # 去掉边框
    ax[4].spines['left'].set_visible(False)  # 去掉边框
    ax[4].spines['bottom'].set_visible(False)  # 去掉边框
    ax[4].get_yaxis().set_visible(False)
    ax[4].get_xaxis().set_visible(False)
    # st = ax[4].scatter(x4, y4, s=scatter_sz, c=adata.obs['pSM_values'], cmap='plasma', marker=".")
    # #ax.invert_yaxis()
    # clb = fig.colorbar(st, shrink=0.4)
    # clb.ax[4].set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax.set_title("SpaceFlow PSM", fontsize=14)
    # ax.set_facecolor("none")
    # clb = fig.colorbar(st3)
    # clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax[2].set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # ax[4].set_title("PearlST pSM", fontsize=12)
    ax[4].set_facecolor("none")



    '''
        save_dir = os.path.dirname(pSM_figure_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(pSM_figure_save_filepath, dpi=300)
        print(f"Plotting complete, pseudo-Spatiotemporal Map figure saved at {pSM_figure_save_filepath} !")
        plt.close('all')
    '''
    # except NameError:
    #     print(error_message)
    # except AttributeError:
    #     print(error_message)



# from mpl_toolkits.axes_grid1 import make_axes_locatable  # 导入工具

# def plot_pSM_multi_5s(adata, adata1, adata2, adata3, adata4, scatter_sz=1., rsz=4.,
#                      csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
#     """
#     Plot the domain segmentation for ST data in spatial
#     """
#     error_message = "No pseudo Spatiotemporal Map data found, please ensure you have run the pseudo_Spatiotemporal_Map() method."
    
#     fig, ax = prepare_figure_multi_5(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right,
#                                      bottom=bottom, top=top)
    
#     # 定义一个函数来添加颜色条
#     def add_colorbar(ax, scatter_plot, label):
#         divider = make_axes_locatable(ax)  # 创建可分割的坐标轴
#         cax = divider.append_axes("bottom", size="5%", pad=0.1)  # 在正下方添加一个坐标轴
#         clb = fig.colorbar(scatter_plot, cax=cax, orientation="horizontal")  # 创建水平颜色条
#         clb.set_label(label, fontsize=10, weight='bold')  # 设置颜色条标签
#         return clb
    
#     # Plot for adata
#     x0, y0 = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
#     st0 = ax[0].scatter(x0, y0, s=scatter_sz, c=adata.obs['pSM_values'], cmap='summer', marker=".")
#     ax[0].invert_yaxis()
#     ax[0].spines['right'].set_visible(False)
#     ax[0].spines['top'].set_visible(False)
#     ax[0].spines['left'].set_visible(False)
#     ax[0].spines['bottom'].set_visible(False)
#     ax[0].get_yaxis().set_visible(False)
#     ax[0].get_xaxis().set_visible(False)
#     ax[0].set_facecolor("none")
#     add_colorbar(ax[0], st0, "pseudotime")  # 添加颜色条
    
#     # Plot for adata1
#     x1, y1 = adata1.obsm["spatial"][:, 0], adata1.obsm["spatial"][:, 1]
#     st1 = ax[1].scatter(x1, y1, s=scatter_sz, c=adata1.obs['pSM_values'], cmap='summer', marker=".")
#     ax[1].invert_yaxis()
#     ax[1].spines['right'].set_visible(False)
#     ax[1].spines['top'].set_visible(False)
#     ax[1].spines['left'].set_visible(False)
#     ax[1].spines['bottom'].set_visible(False)
#     ax[1].get_yaxis().set_visible(False)
#     ax[1].get_xaxis().set_visible(False)
#     ax[1].set_facecolor("none")
#     add_colorbar(ax[1], st1, "pseudotime")  # 添加颜色条
    
#     # Plot for adata2
#     x2, y2 = adata2.obsm["spatial"][:, 0], adata2.obsm["spatial"][:, 1]
#     st2 = ax[2].scatter(x2, y2, s=scatter_sz, c=adata2.obs['pSM_values'], cmap='summer', marker=".")
#     ax[2].invert_yaxis()
#     ax[2].spines['right'].set_visible(False)
#     ax[2].spines['top'].set_visible(False)
#     ax[2].spines['left'].set_visible(False)
#     ax[2].spines['bottom'].set_visible(False)
#     ax[2].get_yaxis().set_visible(False)
#     ax[2].get_xaxis().set_visible(False)
#     ax[2].set_facecolor("none")
#     add_colorbar(ax[2], st2, "pseudotime")  # 添加颜色条
    
#     # Plot for adata3
#     x3, y3 = adata3.obsm["spatial"][:, 0], adata3.obsm["spatial"][:, 1]
#     st3 = ax[3].scatter(x3, y3, s=scatter_sz, c=adata3.obs['pSM_values'], cmap='summer', marker=".")
#     ax[3].invert_yaxis()
#     ax[3].spines['right'].set_visible(False)
#     ax[3].spines['top'].set_visible(False)
#     ax[3].spines['left'].set_visible(False)
#     ax[3].spines['bottom'].set_visible(False)
#     ax[3].get_yaxis().set_visible(False)
#     ax[3].get_xaxis().set_visible(False)
#     ax[3].set_facecolor("none")
#     add_colorbar(ax[3], st3, "pseudotime")  # 添加颜色条
    
#     # Plot for adata4
#     x4, y4 = adata4.obsm["spatial"][:, 0], adata4.obsm["spatial"][:, 1]
#     st4 = ax[4].scatter(x4, y4, s=scatter_sz, c=adata4.obs['pSM_values'], cmap='summer', marker=".")
#     ax[4].invert_yaxis()
#     ax[4].spines['right'].set_visible(False)
#     ax[4].spines['top'].set_visible(False)
#     ax[4].spines['left'].set_visible(False)
#     ax[4].spines['bottom'].set_visible(False)
#     ax[4].get_yaxis().set_visible(False)
#     ax[4].get_xaxis().set_visible(False)
#     ax[4].set_facecolor("none")
#     add_colorbar(ax[4], st4, "pseudotime")  # 添加颜色条
    
#     plt.show()


def plot_pSM_multi_5s(adata, adata1, adata2, adata3, adata4, scatter_sz=1., rsz=4.,
                     csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Plot the domain segmentation for ST data in spatial
    """
    error_message = "No pseudo Spatiotemporal Map data found, please ensure you have run the pseudo_Spatiotemporal_Map() method."
    
    fig, ax = prepare_figure_multi_5(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right,
                                     bottom=bottom, top=top)
    
    # Plot for adata
    x0, y0 = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    st0 = ax[0].scatter(x0, y0, s=scatter_sz, c=adata.obs['pSM_values'], cmap='summer', marker=".")
    ax[0].invert_yaxis()
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_facecolor("none")
    
    # Plot for adata1
    x1, y1 = adata1.obsm["spatial"][:, 0], adata1.obsm["spatial"][:, 1]
    st1 = ax[1].scatter(x1, y1, s=scatter_sz, c=adata1.obs['pSM_values'], cmap='summer', marker=".")
    ax[1].invert_yaxis()
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].set_facecolor("none")
    
    # Plot for adata2
    x2, y2 = adata2.obsm["spatial"][:, 0], adata2.obsm["spatial"][:, 1]
    st2 = ax[2].scatter(x2, y2, s=scatter_sz, c=adata2.obs['pSM_values'], cmap='summer', marker=".")
    ax[2].invert_yaxis()
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    ax[2].get_xaxis().set_visible(False)
    ax[2].set_facecolor("none")
    
    # Plot for adata3
    x3, y3 = adata3.obsm["spatial"][:, 0], adata3.obsm["spatial"][:, 1]
    st3 = ax[3].scatter(x3, y3, s=scatter_sz, c=adata3.obs['pSM_values'], cmap='summer', marker=".")
    ax[3].invert_yaxis()
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].get_yaxis().set_visible(False)
    ax[3].get_xaxis().set_visible(False)
    ax[3].set_facecolor("none")
    
    # Plot for adata4
    x4, y4 = adata4.obsm["spatial"][:, 0], adata4.obsm["spatial"][:, 1]
    st4 = ax[4].scatter(x4, y4, s=scatter_sz, c=adata4.obs['pSM_values'], cmap='summer', marker=".")
    ax[4].invert_yaxis()
    ax[4].spines['right'].set_visible(False)
    ax[4].spines['top'].set_visible(False)
    ax[4].spines['left'].set_visible(False)
    ax[4].spines['bottom'].set_visible(False)
    ax[4].get_yaxis().set_visible(False)
    ax[4].get_xaxis().set_visible(False)
    ax[4].set_facecolor("none")
    
    # 在最后一个子图（ax[4]）的右侧添加共享的颜色条
    clb = fig.colorbar(st4, ax=ax[4], location='right', pad=0.02, shrink=0.8)
    clb.set_label("pseudotime", fontsize=10, weight='bold', labelpad=10)
    
    plt.show()



def plot_pSM_multi_4(adata, adata1, adata2, adata3, scatter_sz=1., rsz=4.,
                     csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Plot the domain segmentation for ST data in spatial
    """
    error_message = "No pseudo Spatiotemporal Map data found, please ensure you have run the pseudo_Spatiotemporal_Map() method."
    
    fig, ax = prepare_figure_multi_4(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right,
                                     bottom=bottom, top=top)
    
    # Plot for adata
    x0, y0 = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    st0 = ax[0].scatter(x0, y0, s=scatter_sz, c=adata.obs['pSM_values'], cmap='summer', marker=".")
    
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_facecolor("none")
    
    # Plot for adata1
    x1, y1 = adata1.obsm["spatial"][:, 0], adata1.obsm["spatial"][:, 1]
    st1 = ax[1].scatter(x1, y1, s=scatter_sz, c=adata1.obs['pSM_values'], cmap='summer', marker=".")
    
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].set_facecolor("none")
    
    # Plot for adata2
    x2, y2 = adata2.obsm["spatial"][:, 0], adata2.obsm["spatial"][:, 1]
    st2 = ax[2].scatter(x2, y2, s=scatter_sz, c=adata2.obs['pSM_values'], cmap='summer', marker=".")
    
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    ax[2].get_xaxis().set_visible(False)
    ax[2].set_facecolor("none")
    
    # Plot for adata3
    x3, y3 = adata3.obsm["spatial"][:, 0], adata3.obsm["spatial"][:, 1]
    st3 = ax[3].scatter(x3, y3, s=scatter_sz, c=adata3.obs['pSM_values'], cmap='summer', marker=".")
    
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].get_yaxis().set_visible(False)
    ax[3].get_xaxis().set_visible(False)
    ax[3].set_facecolor("none")
    
    # # Plot for adata4
    # x4, y4 = adata4.obsm["spatial"][:, 0], adata4.obsm["spatial"][:, 1]
    # st4 = ax[4].scatter(x4, y4, s=scatter_sz, c=adata4.obs['pSM_values'], cmap='summer', marker=".")
    # ax[4].invert_yaxis()
    # ax[4].spines['right'].set_visible(False)
    # ax[4].spines['top'].set_visible(False)
    # ax[4].spines['left'].set_visible(False)
    # ax[4].spines['bottom'].set_visible(False)
    # ax[4].get_yaxis().set_visible(False)
    # ax[4].get_xaxis().set_visible(False)
    # ax[4].set_facecolor("none")
    
    # # 在最后一个子图（ax[4]）的右侧添加共享的颜色条
    # clb = fig.colorbar(st4, ax=ax[4], location='right', pad=0.02, shrink=0.8)
    # clb.set_label("pseudotime", fontsize=10, weight='bold', labelpad=10)
    
    # plt.show()





