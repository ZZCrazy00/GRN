import os
import tqdm
import pandas as pd

# Visium_19_CK297 4113 × 17703
# V1_Mouse_Kidney 1438 × 32285
# MOB_processed 2050 × 10000    1200
# GSM4745615_DBiT-seq_processed     3
# GSM5173925_OB1-Slide01_slide-seqv2_processed 30827 × 21354    140
# E9.5_E1S1.MOSTA
# seqFISH(embryo1)
# 获取基因的表达矩阵和TF的得分矩阵。
data_name = 'Visium_19_CK297'
gene_feature = pd.read_csv("dataset/{}/gene_expression.csv".format(data_name), index_col=0)
gene_list = gene_feature.index.values
TF_feature = pd.read_csv("dataset/{}/TF_feature.csv".format(data_name), index_col=0)
TF_list = TF_feature.index.values

# receptor_score
cell_receptor = pd.read_csv("dataset/{}/cell_receptor.csv".format(data_name))    # 2873 4113
cell_receptor['receptor'] = cell_receptor['interaction'].str.split('^').str[1]
cell_receptor = cell_receptor.drop(['interaction'], axis=1)
cell_receptor = cell_receptor.groupby('receptor').mean().reset_index()
cell_receptor.set_index('receptor', inplace=True)   # 718 4113
receptor_gene_list = list(cell_receptor.index)

# TF-TG
# human mouse
data_type = 'human'
GRN_database = pd.read_csv("resource/{}/TF_TG{}db.csv".format(data_type, data_type), index_col=0).drop_duplicates()    # 9572491 280835 230571
GRNs = []
for TF, target in tqdm.tqdm(GRN_database.values):
    if TF in TF_list and target in gene_list:
        TF_nonzero = gene_feature.loc[TF] > 0
        target_nonzero = gene_feature.loc[target] > 0
        overlap = TF_nonzero.values & target_nonzero.values
        if overlap.sum() > 0:
            GRNs.append([TF, target])
GRN = pd.DataFrame(GRNs, index=None, columns=['TF', 'target'])
GRN.to_csv("dataset/{}/GRN.csv".format(data_name))
GRN = pd.read_csv("dataset/{}/GRN.csv".format(data_name), index_col=0)
GRN_list = GRN.values.tolist()
TF_gene_list = list(GRN['TF'].drop_duplicates())    # 256
target_gene_list = list(GRN['target'].drop_duplicates())    # 9191

# receptor-TF
pathway_score = pd.read_csv("dataset/{}/pathway_scores.csv".format(data_name), index_col=0)     # 26002

# TF_score
TF_score = pd.read_csv("dataset/{}/TF_feature.csv".format(data_name), index_col=0)  # 288

total_data = []
for receptor in tqdm.tqdm(receptor_gene_list):  # 6599138
    for target in target_gene_list:
        total_data.append([receptor, target])
print(len(total_data))

pos_data = []
receptor_TF_list = list(pathway_score.index)
for pair in tqdm.tqdm(receptor_TF_list):
    receptor, TF = pair.split("->")
    pathway_nonzero = pathway_score.loc[receptor + '->' + TF] > 0
    for target in target_gene_list:
        if [TF, target] in GRN_list:
            target_nonzero = gene_feature.loc[target] > 0
            overlap = pathway_nonzero.values & target_nonzero.values
            if overlap.any():
                pos_data.append([receptor, target])
print(len(pos_data))
# print(pos_data)
pos_data = pd.DataFrame(pos_data, columns=['receptor', 'target'])
pos_data.to_csv("dataset/{}/post_data.csv".format(data_name), index=False)

