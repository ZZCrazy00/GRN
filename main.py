import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def draw(train_losses, test_aucs,  test_aupr, test_f1, dataset):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(test_aucs, label='Val AUC', color='blue')
    plt.title('Test AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(test_aupr, label='Test AUPR', color='red')
    plt.title('Test AUPR')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(test_f1, label='Test F1', color='red')
    plt.title('Test F1')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../dataset/{}/GCN.png'.format(dataset))
    plt.close()


class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return z


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x, data.edge_index)
    pos_pred = model.decode(z, data.pos_edge_label_index)
    neg_pred = model.decode(z, data.neg_edge_label_index)

    pos_loss = -torch.log(pos_pred.sigmoid() + 1e-15).mean()
    neg_loss = -torch.log((1 - neg_pred.sigmoid()) + 1e-15).mean()
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)

        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        pos_pred = model.decode(z, pos_edge_index).sigmoid()
        neg_pred = model.decode(z, neg_edge_index).sigmoid()

    # Calculate metrics
    pos_labels = torch.ones(pos_pred.size(0))
    neg_labels = torch.zeros(neg_pred.size(0))
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    auc = roc_auc_score(labels.cpu(), pred.cpu())
    aupr = average_precision_score(labels.cpu(), pred.cpu())
    pred_labels = (pred > 0.5).float()
    f1 = f1_score(labels.cpu(), pred_labels.cpu())

    return auc, aupr, f1


if __name__ == '__main__':
    data_name = 'Visium_19_CK297'
    train_data = torch.load("dataset/{}/train_data.pt".format(data_name))
    val_data = torch.load("dataset/{}/val_data.pt".format(data_name))
    test_data = torch.load("dataset/{}/test_data.pt".format(data_name))
    print(train_data)
    print(val_data)
    print(test_data)
    train_data, val_data, test_data = train_data.cuda(), val_data.cuda(), test_data.cuda()

    model = GCNLinkPredictor(input_dim=train_data.x.shape[1], hidden_dim=16, output_dim=16).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_aucs = []
    test_aupr = []
    test_f1 = []
    best_auc = 0
    for epoch in range(1000):
        loss = train(model, optimizer, train_data)
        val_result = test(model, val_data)
        test_result = test(model, test_data)

        if epoch % 10 == 0:  # 每10个epoch打印一次
            print(f'Epoch {epoch}, Loss: {loss:.4f},'
                  f' Val AUC: {val_result[0]:.4f}, AUPR: {val_result[1]:.4f}, F1: {val_result[2]:.4f}'
                  f' Test AUC: {test_result[0]:.4f}, AUPR: {test_result[1]:.4f}, F1: {test_result[2]:.4f}')

        train_losses.append(loss)
        test_aucs.append(test_result[0])
        test_aupr.append(test_result[1])
        test_f1.append(test_result[2])
        draw(train_losses, test_aucs, test_aupr, test_f1, data_name)


