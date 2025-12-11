import tqdm
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data

# GSM4745615_DBiT-seq_processed
# MOB_processed     # seqFISH+
# GSM5173925_OB1-Slide01_slide-seqv2_processed
# V1_Mouse_Kidney
# Visium_19_CK297
# E9.5_E1S1.MOSTA
# seqFISH(embryo1)
data_name = 'GSM5173925_OB1-Slide01_slide-seqv2_processed'

# 读取特征
gene_feature = pd.read_csv(f"dataset/{data_name}/gene_expression.csv", index_col=0)  # target 的特征
gene_list = gene_feature.index.values
TF_feature = pd.read_csv("dataset/{}/TF_feature.csv".format(data_name), index_col=0)
TF_list = TF_feature.index.values
cell_receptor = pd.read_csv(f"dataset/{data_name}/cell_receptor.csv")  # 2873 4113
cell_receptor['receptor'] = cell_receptor['interaction'].str.split('^').str[1]
cell_receptor = cell_receptor.drop(['interaction'], axis=1)
cell_receptor = cell_receptor.groupby('receptor').mean().reset_index().set_index("receptor")  # 718 4113
receptor_gene_list = cell_receptor.index.tolist()  # 718

# 读取 GRN 数据并提取 TF 和 target 基因列表
# human mouse
data_type = 'mouse'
GRN_database = pd.read_csv("resource/{}/TF_TG{}db.csv".format(data_type, data_type),
                           index_col=0).drop_duplicates()  # 9572491 280835 230571
GRNs = []
for TF, target in tqdm.tqdm(GRN_database.values):
    if TF in TF_list and target in gene_list:
        TF_nonzero = gene_feature.loc[TF] > 0
        target_nonzero = gene_feature.loc[target] > 0
        overlap = TF_nonzero.values & target_nonzero.values
        if overlap.sum() > 0:
            GRNs.append([TF, target])
GRNs = pd.DataFrame(GRNs, index=None, columns=['TF', 'target'])
GRNs.to_csv("dataset/{}/GRN.csv".format(data_name))

GRN = pd.read_csv(f"dataset/{data_name}/GRN.csv", index_col=0)
TF_gene_list = GRN['TF'].drop_duplicates().tolist()  # 256
target_gene_list = GRN['target'].drop_duplicates().tolist()  # 9191

# 生成 total_data 所有的 receptor-target 组合
total_data = pd.DataFrame(product(receptor_gene_list, target_gene_list), columns=['receptor', 'target'])  # 718x9191

# 读取正样本
post_data = pd.read_csv(f"dataset/{data_name}/post_data.csv").drop_duplicates()  # 643019

# 按比例划分正样本
train_pos, test_pos = train_test_split(post_data, test_size=0.2, random_state=42)  # 80%训练集，20%测试+验证集
val_pos, test_pos = train_test_split(test_pos, test_size=0.5, random_state=42)  # 50%验证集，50%测试集
train_pos['label'] = 1
val_pos['label'] = 1
test_pos['label'] = 1

# 生成负样本：训练集的 post_data 中没有的 receptor-target 组合
neg_data = total_data.merge(train_pos, on=['receptor', 'target'], how='outer', indicator=True)
neg_data = neg_data[neg_data['_merge'] == 'left_only'].drop('_merge', axis=1)


# 排除在相同细胞内有表达的基因对
def filter_coexpressed_pairs(neg_data, cell_receptor, gene_feature):
    """
    过滤掉在相同细胞内有表达的受体-靶基因对
    """
    filtered_neg_data = []

    for idx, row in tqdm.tqdm(neg_data.iterrows(), total=len(neg_data), desc="Filtering co-expressed pairs"):
        receptor = row['receptor']
        target = row['target']

        # 检查受体和靶基因是否在同一个细胞中同时表达
        if receptor in cell_receptor.index and target in gene_feature.index:
            receptor_expression = cell_receptor.loc[receptor] > 0
            target_expression = gene_feature.loc[target] > 0

            # 如果存在至少一个细胞中两者都表达，则排除该对
            coexpressed_cells = receptor_expression & target_expression
            if not coexpressed_cells.any():
                filtered_neg_data.append(row)
        else:
            # 如果基因不在表达矩阵中，保留该对
            filtered_neg_data.append(row)

    return pd.DataFrame(filtered_neg_data)


# 应用过滤函数
print("Filtering co-expressed pairs from negative samples...")
neg_data_filtered = filter_coexpressed_pairs(neg_data, cell_receptor, gene_feature)
print(f"Original negative samples: {len(neg_data)}")
print(f"After filtering co-expressed pairs: {len(neg_data_filtered)}")

# 从负样本中随机选择与正样本数量相等的样本
train_neg = neg_data_filtered.sample(n=len(train_pos), random_state=43)
val_neg = neg_data_filtered.sample(n=len(val_pos), random_state=43)
test_neg = neg_data_filtered.sample(n=len(test_pos), random_state=43)
train_neg['label'] = 0
val_neg['label'] = 0
test_neg['label'] = 0

# 将正负样本合并为最终的训练集、验证集和测试集
train_data = pd.concat([train_pos, train_neg]).reset_index(drop=True)  # 1028830
val_data = pd.concat([val_pos, val_neg]).reset_index(drop=True)  # 128604
test_data = pd.concat([test_pos, test_neg]).reset_index(drop=True)  # 128604
node_features = pd.concat([cell_receptor, gene_feature])
node_features = torch.tensor(node_features.values, dtype=torch.float)


def get_pyg_data(node_feature, cell_receptor, gene_feature, data, edge_index=None):
    pos_edge_index = []
    neg_edge_index = []
    for pair in tqdm.tqdm(data.values):
        receptor, target, label = pair
        receptor_index = cell_receptor.index.get_loc(receptor)
        target_index = gene_feature.index.get_loc(target) + len(cell_receptor)
        if label == 1:
            pos_edge_index.append([receptor_index, target_index])
        else:
            neg_edge_index.append([receptor_index, target_index])

    if edge_index is not None:
        pyg_data = Data(x=node_feature, edge_index=edge_index,
                        pos_edge_label_index=torch.LongTensor(pos_edge_index).T,
                        neg_edge_label_index=torch.LongTensor(neg_edge_index).T)
    else:
        pyg_data = Data(x=node_feature, edge_index=torch.LongTensor(pos_edge_index).T,
                        pos_edge_label_index=torch.LongTensor(pos_edge_index).T,
                        neg_edge_label_index=torch.LongTensor(neg_edge_index).T)
    return pyg_data


train_data = get_pyg_data(node_features, cell_receptor, gene_feature, train_data)
torch.save(train_data, 'dataset/{}/train_data_new.pt'.format(data_name))
val_data = get_pyg_data(node_features, cell_receptor, gene_feature, val_data, train_data.edge_index)
torch.save(val_data, 'dataset/{}/val_data_new.pt'.format(data_name))
test_data = get_pyg_data(node_features, cell_receptor, gene_feature, test_data, train_data.edge_index)
torch.save(test_data, 'dataset/{}/test_data_new.pt'.format(data_name))

print(train_data)
print(val_data)
print(test_data)
