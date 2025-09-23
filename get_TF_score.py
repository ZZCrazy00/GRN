import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Visium_19_CK297 4113 × 17703
# V1_Mouse_Kidney 1438 × 32285
# MOB_processed 2050 × 10000    1200
# GSM4745615_DBiT-seq_processed     3
# GSM5173925_OB1-Slide01_slide-seqv2_processed 30827 × 21354    140
# E9.5_E1S1.MOSTA
# E9.5_E2S1.MOSTA
# seqFISH(embryo1)

data_name = 'Visium_19_CK297'
gene_expression = pd.read_csv("dataset/{}/gene_expression.csv".format(data_name), index_col=0)     # 17703 4113

cell_receptor = pd.read_csv("dataset/{}/cell_receptor.csv".format(data_name))    # 2873 4113
cell_receptor['receptor'] = cell_receptor['interaction'].str.split('^').str[1]
cell_receptor = cell_receptor.drop(['interaction'], axis=1)
cell_receptor = cell_receptor.groupby('receptor').mean().reset_index()
cell_receptor.set_index('receptor', inplace=True)   # 718 4113

# human mouse
data_type = 'human'
igraph_type = 'KEGG'
receptor_TF = pd.read_csv("resource/{}_igraph/{}/receptor_TF_detail.csv".format(igraph_type, data_type))     # 69163
pathway_strength = []

cell_pathway_score = []
pathway_name = []
for row in tqdm.tqdm(receptor_TF.values):
    receptor, TF = row[0], row[2]
    if receptor in cell_receptor.index and TF in gene_expression.index:     # 确保受体有收到信号，转录因子在细胞中存在
        paths = row[1]
        path = eval(paths)

        receive_score = cell_receptor.loc[receptor]
        pathway_score = receive_score.copy()
        receive_array = np.copy(receive_score)

        TF_score = gene_expression.loc[TF]
        TF_array = np.copy(TF_score)

        receptor_indices = np.nonzero(receive_array)[0]     # 获取有接收到信号的细胞坐标
        TF_indices = np.nonzero(TF_array)[0]     # 获取转录因子有表达的细胞坐标
        non_zero_indices = np.intersect1d(receptor_indices, TF_indices)

        select_index = np.zeros_like(receive_array)
        select_index[non_zero_indices] = 1

        for gene in path:
            if gene in gene_expression.index:
                # 获取基因表达并乘以 new_array 中的筛选结果
                score = gene_expression.loc[gene] * select_index
                # 累加到通路分数
                pathway_score += score
        pathway_score /= len(path)  # 计算通路中各个基因的平均表达，作为通路强度
        if sum(pathway_score) != 0:
            pathway_name.append(receptor + "->" + TF)
            cell_pathway_score.append(pathway_score)
        else:
            continue
pathway_scores = pd.DataFrame(cell_pathway_score, index=pathway_name)  # (26913, 4113)
pathway_scores.to_csv("dataset/{}/pathway_scores.csv".format(data_name))

# 求平均得到TF分数
pathway_scores = pd.read_csv("dataset/{}/pathway_scores.csv".format(data_name))   # 26913 4113
TF_dict = {}
for row in pathway_scores.values:
    receptor, TF = row[0].split("->")
    score = row[1:]
    if TF not in TF_dict:
        TF_dict[TF] = {'score_sum': np.array(score, dtype=float), 'count': 1}
    else:
        TF_dict[TF]['score_sum'] += np.array(score, dtype=float)
        TF_dict[TF]['count'] += 1
TF_avg_score = {TF: values['score_sum'] / values['count'] for TF, values in TF_dict.items()}
TF_score_df = pd.DataFrame.from_dict(TF_avg_score, orient='index')
TF_score_df.to_csv("dataset/{}/TF_feature.csv".format(data_name))   # 309 4113

