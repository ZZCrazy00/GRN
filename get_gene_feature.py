import ast
import tqdm
import torch
import anndata
import liana as li
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack


def calculate_ave_features(gene_set, gene_list, adata, num_features):
    ave_feature = []
    for genes in gene_set:
        if pd.isna(genes):
            ave_feature.append(np.zeros(num_features))
        else:
            gene_list_filtered = [i for i in ast.literal_eval(genes) if i in gene_list]
            if gene_list_filtered:
                feature_mean = np.mean(adata[:, gene_list_filtered].X, axis=1).flatten()
                ave_feature.append(feature_mean)
            else:
                ave_feature.append(np.zeros(num_features))
    ave_feature_array = np.vstack(ave_feature).T
    return ave_feature_array


def _norm_max(X):
    X = X.A
    return np.where(np.isnan(X), 0, X)


def hill(dot, AG_lr, AN_lr):
    Kh = 0.5
    dot = dot / (Kh + dot)
    ag = (1 + AG_lr / (Kh + AG_lr)).squeeze(-1)
    an = (1 - AN_lr / (Kh + AN_lr)).squeeze(-1)
    score = dot * torch.ger(ag, ag) * torch.ger(an, an)
    return score


def _vectorized_cosine(x_mat, y_mat, ag_mat, an_mat, weight, device='cuda'):
    x_mat = torch.tensor(x_mat, dtype=torch.float16, device=device)
    y_mat = torch.tensor(y_mat, dtype=torch.float16, device=device)
    ag_mat = torch.tensor(ag_mat, dtype=torch.float16, device=device)
    an_mat = torch.tensor(an_mat, dtype=torch.float16, device=device)
    weight = weight.todense()
    weight = torch.tensor(weight, dtype=torch.float16, device=device)

    num_cells, num_lrs = x_mat.shape
    receptor_mat = torch.zeros((num_lrs, num_cells), dtype=torch.float16, device=device)
    ligand_mat = torch.zeros((num_lrs, num_cells), dtype=torch.float16, device=device)
    for lr in tqdm.tqdm(range(num_lrs)):
        x_lr = x_mat[:, lr].unsqueeze(1)  # (num_cells, 1)
        y_lr = y_mat[:, lr].unsqueeze(1)  # (num_cells, 1)
        ag_lr = ag_mat[:, lr].unsqueeze(1)
        an_lr = an_mat[:, lr].unsqueeze(1)

        xy_dot = (x_lr * y_lr.T) * weight
        xy_dot = hill(xy_dot, ag_lr, an_lr)

        receptor_mat[lr] = sum(xy_dot)
        ligand_mat[lr] = sum(xy_dot.T)

    return receptor_mat.cpu().numpy(), ligand_mat.cpu().numpy()


def _rename_means(lr_stats, entity):
    df = lr_stats.copy()
    df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
    return df.rename(columns={'gene': entity})


def _get_props(X_mask):
    return X_mask.getnnz(axis=0) / X_mask.shape[0]


def select_resource(resource_name):
    resource_name = resource_name.lower()
    resource = pd.read_csv("resource/omni_resource.csv", index_col=False)
    resource = resource[resource['resource'] == resource_name]
    resource = resource[['source_genesymbol', 'target_genesymbol']]
    resource = resource.rename(columns={'source_genesymbol': 'ligand',
                                        'target_genesymbol': 'receptor'})
    return resource


def _add_complexes_to_var(adata, entities):
    complexes = [comp for comp in entities if '_' in comp]
    X_list = []
    for comp in complexes:
        subunits = comp.split('_')
        # 只保留 subunits 都在 adata.var.index 中的 complexes
        if all([subunit in adata.var.index for subunit in subunits]):
            adata.var.loc[comp, :] = None
            new_array = csr_matrix(adata[:, subunits].X.min(axis=1))
            X_list.append(new_array)
    X_combined = hstack([adata.X] + X_list)
    adata = anndata.AnnData(X=X_combined, obs=adata.obs, var=adata.var,
                            obsm=adata.obsm, obsp=adata.obsp, uns=adata.uns)
    return adata


# Visium_19_CK297 4113 × 17703  160

input_data = 'Visium_19_CK297'
# adata = sc.datasets.visium_sge(sample_id='V1_Mouse_Kidney')
adata = anndata.read_h5ad('input_data/{}.h5ad'.format(input_data))
adata.X = sp.csr_matrix(adata.X)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata = adata[:, ~adata.var_names.duplicated()]
gene_expression = _norm_max(adata.X)
gene_expression = pd.DataFrame(gene_expression.T, index=adata.var_names)
gene_expression.to_csv("dataset/{}/gene_expression.csv".format(input_data))

# 加载资源库
resource = select_resource(resource_name='consensus')  # consensus mouseconsensus

# 在数据集中添加资源库中的复合体
adata = _add_complexes_to_var(adata, np.union1d(resource['ligand'].astype(str), resource['receptor'].astype(str)))

# 仅保留数据集中存在的基因并去除自连接
resource = resource[(np.isin(resource['ligand'], adata.var_names)) &
                    (np.isin(resource['receptor'], adata.var_names))]
self_interactions = resource['ligand'] == resource['receptor']
resource = resource[~self_interactions]

# 获取LR_pair
xy_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                         'props': _get_props(adata.X)},
                        index=adata.var_names
                        ).reset_index().rename(columns={'index': 'gene'})
xy_stats = xy_stats.rename(columns={'gene_short_name': 'gene'})
xy_stats = resource.merge(_rename_means(xy_stats, entity='ligand')).merge(_rename_means(xy_stats, entity='receptor'))
xy_stats['interaction'] = xy_stats['ligand'] + '^' + xy_stats['receptor']

agonist_data = pd.read_csv("resource/human/LRdata_with_agonist_and_antagonist.csv", index_col=0)
xy_stats = pd.merge(xy_stats, agonist_data[['interaction', 'agonist', 'antagonist']], on='interaction', how='left')

# # 归一化x_mat和y_mat
x_mat = _norm_max(adata[:, xy_stats['ligand']].X)
y_mat = _norm_max(adata[:, xy_stats['receptor']].X)

# 获取agonist基因和antagonist基因的平均表达
gene_list = adata.var_names.tolist()
agonist_genes = xy_stats['agonist'].tolist()
antagonist_genes = xy_stats['antagonist'].tolist()
agonist_mat = calculate_ave_features(agonist_genes, gene_list, adata, adata.n_obs)
antagonist_mat = calculate_ave_features(antagonist_genes, gene_list, adata, adata.n_obs)

# 用于获取weight，细胞的邻接矩阵
li.ut.spatial_neighbors(adata, bandwidth=160, cutoff=0.1, kernel='gaussian', set_diag=True)

# 计算得到receptor_mat和ligand_mat
cell_receptor, cell_ligand = _vectorized_cosine(x_mat=x_mat, y_mat=y_mat, ag_mat=agonist_mat, an_mat=antagonist_mat,
                                                weight=adata.obsp['spatial_connectivities'])
cell_receptor = pd.DataFrame(cell_receptor, index=xy_stats['interaction'])
cell_receptor.to_csv("dataset/{}/cell_receptor.csv".format(input_data))
cell_ligand = pd.DataFrame(cell_ligand, index=xy_stats['interaction'])
cell_ligand.to_csv("dataset/{}/cell_ligand.csv".format(input_data))
