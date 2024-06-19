import torch.nn as nn
import torch_geometric.nn as gnn
import torch
from torch_geometric.nn import MessagePassing
import sys


class GCN(nn.Module):
    def __init__(self,
                 latent_dim, #プロセッサ潜在変数次元数
                 latent_dim_dec, #デコーダ潜在変数次元数
                 num_layer, #プロセッサレイヤー数
                 num_layer_dec, #デコーダレイヤー数
                 activation = nn.ReLU(),
                 activation_dec = nn.ReLU(),
                 in_dim = 3,
                 out_dim = 3):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.activaion = activation
        self.layer_s = gnn.GCNConv(in_dim, latent_dim)
        self.layer_e = gnn.GCNConv(latent_dim, latent_dim)
        self.layers = nn.ModuleList([gnn.GCNConv(latent_dim, latent_dim)]*num_layer)

        ####decoderの定義
        self.activation_dec = activation_dec
        self.layer_s_dec = nn.Linear(latent_dim, latent_dim_dec)
        self.layer_e_dec = nn.Linear(latent_dim_dec, out_dim)
        self.layers_dec = nn.ModuleList([nn.Linear(latent_dim_dec, latent_dim_dec)]*num_layer_dec)
    
    def forward(self, dataBatch):
        h = self.activaion(self.layer_s(dataBatch.x, dataBatch.edge_index))
        for layer in self.layers:
            h = self.activaion(layer(h, dataBatch.edge_index))
        h = self.activaion(self.layer_e(h, dataBatch.edge_index))

        h = self.activation_dec(self.layer_s_dec(h))
        for layer in self.layers_dec:
            h = self.activation_dec(layer(h))
        y = self.layer_e_dec(h)

        x_out = y + dataBatch.x
        
        return x_out

# 注意：multi-headじゃない
class GAT(nn.Module):
    def __init__(self,
                 latent_dim, #プロセッサ潜在変数次元数
                 latent_dim_dec, #デコーダ潜在変数次元数
                 num_layer, #プロセッサレイヤー数
                 num_layer_dec, #デコーダレイヤー数
                 activation = nn.ReLU(),
                 activation_dec = nn.ReLU(),
                 in_dim = 3,
                 out_dim = 3,
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.activaion = activation

        self.layer_s = gnn.GATConv(in_dim, latent_dim)
        self.layer_e = gnn.GATConv(latent_dim, latent_dim)
        self.layers = nn.ModuleList([gnn.GATConv(latent_dim, latent_dim)]*num_layer)

        self.activation_dec = activation_dec
        self.layer_s_dec = nn.Linear(latent_dim, latent_dim_dec)
        self.layer_e_dec = nn.Linear(latent_dim_dec, out_dim)
        self.layers_dec = nn.ModuleList([nn.Linear(latent_dim_dec, latent_dim_dec)]*num_layer_dec)
    
    def forward(self, dataBatch):
        h = self.activaion(self.layer_s(dataBatch.x, dataBatch.edge_index))
        for layer in self.layers:
            h = self.activaion(layer(h, dataBatch.edge_index))
        h = self.activaion(self.layer_e(h, dataBatch.edge_index))

        h = self.activation_dec(self.layer_s_dec(h))
        for layer in self.layers_dec:
            h = self.activation_dec(layer(h))
        y = self.layer_e_dec(h)

        x_out = y + dataBatch.x
        
        return x_out


"""
GCN_edgeで利用するレイヤー
ノード特徴量とエッジ特徴量を結合し、それを入力としてノード特徴量を更新する。
"""
class NodeProc(MessagePassing):
    def __init__(self, in_features, out_features, activation):
        super().__init__(aggr = "mean")
        self.lin = nn.Linear(in_features, out_features, bias = False)
        self.activation = activation
    
    def forward(self, x, edge_attr, edge_index):
        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self, x_j, edge_attr):
        h = torch.cat((x_j, edge_attr), dim = 1)
        return self.activation(self.lin(h))

#####エッジ特徴量を基に学習するモデル
class GCN_edge(nn.Module):
    def __init__(self,
                 num_layer = 2, #エッジ特徴量およびノード特徴量のプロセッサレイヤー数
                 latent_dim_node = 32, #ノード特徴量プロセッサレイヤーの潜在空間dim
                 latent_dim_edge = 32, #エッジ特徴量プロセッサレイヤーの潜在空間dim
                 num_layer_enc_node = 2, #ノード特徴量のエンコーダレイヤー数。以下同上。
                 num_layer_enc_edge = 2,
                 latent_dim_enc_node = 32,
                 latent_dim_enc_edge = 32,
                 in_features_node = 1,
                 in_features_edge = 3,
                 activation_node = nn.ReLU(),
                 activation_edge = nn.ReLU(),
                 activation_enc_node = nn.ReLU(),
                 activation_enc_edge = nn.ReLU(),
                 ):
        super().__init__()
        self.num_layer = num_layer

        ####ノード特徴量エンコーダの定義
        self.activation_enc_node = activation_enc_node
        self.layer_s_enc_node = nn.Linear(in_features_node, latent_dim_enc_node)
        self.layer_e_enc_node = nn.Linear(latent_dim_enc_node, latent_dim_enc_node)
        self.layers_enc_node = nn.ModuleList([nn.Linear(latent_dim_enc_node, latent_dim_enc_node)]*num_layer_enc_node)

        ####エッジ特徴量エンコーダの定義
        self.activation_enc_edge = activation_enc_edge
        self.layer_s_enc_edge = nn.Linear(in_features_edge, latent_dim_enc_edge)
        self.layer_e_enc_edge = nn.Linear(latent_dim_enc_edge, latent_dim_enc_edge)
        self.layers_enc_edge = nn.ModuleList([nn.Linear(latent_dim_enc_edge, latent_dim_enc_edge)]*num_layer_enc_edge)

        #####ノード特徴量プロセッサの定義
        self.layer_s_node = NodeProc(latent_dim_enc_node + latent_dim_enc_edge, latent_dim_node, activation_node)
        self.layers_node = nn.ModuleList([NodeProc(latent_dim_node + latent_dim_edge, latent_dim_node, activation_node)]*num_layer)
        self.layer_e_node = NodeProc(latent_dim_node + latent_dim_edge, latent_dim_node, activation_node)

        #####エッジ特徴量プロセッサの定義
        self.activation_edge = activation_edge
        self.layer_s_edge = nn.Linear(2*latent_dim_node + latent_dim_enc_edge, latent_dim_edge)
        self.layers_edge = nn.ModuleList([nn.Linear(2*latent_dim_node + latent_dim_edge, latent_dim_edge)]*num_layer)
        self.layer_e_edge = nn.Linear(2*latent_dim_node + latent_dim_edge, 3)

    def forward(self, data):
        #####ノード特徴量エンコード
        h_node = self.activation_enc_node(self.layer_s_enc_node(data.x))
        for layer in self.layers_enc_node:
            h_node = self.activation_enc_node(layer(h_node))
        h_node = self.activation_enc_node(self.layer_e_enc_node(h_node))

        #####エッジ特徴量エンコード
        h_edge = self.activation_enc_edge(self.layer_s_enc_edge(data.edge_attr))
        for layer in self.layers_enc_edge:
            h_edge = self.activation_enc_edge(layer(h_edge))
        h_edge = self.activation_enc_edge(self.layer_e_enc_edge(h_edge))

        #####ノード特徴量とエッジ特徴量のプロセッシング
        index_i = data.edge_index[0]
        index_j = data.edge_index[1]
        h_node = self.layer_s_node(h_node, h_edge, data.edge_index)
        h_edge = torch.cat((h_edge, h_node[index_i], h_node[index_j]), dim = 1)
        h_edge = self.activation_edge(self.layer_s_edge(h_edge))

        for i in range(self.num_layer):
            h_node = self.layers_node[i](h_node, h_edge, data.edge_index)
            h_edge = torch.cat((h_edge, h_node[index_i], h_node[index_j]), dim = 1)
            h_edge = self.activation_edge(self.layers_edge[i](h_edge))
        
        h_node = self.layer_e_node(h_node, h_edge, data.edge_index)
        h_edge = torch.cat((h_edge, h_node[index_i], h_node[index_j]), dim = 1)
        h_edge = self.layer_e_edge(h_edge)

        y = data.edge_attr + h_edge

        return y


#####GCN_edge同様エッジ特徴量を基に学習する。
# ただし、e_ij = -e_jiとなるように工夫。
class GCN_edge_antisym(GCN_edge):
    def __init__(self,
                 num_layer = 2, #エッジ特徴量およびノード特徴量のプロセッサレイヤー数
                 latent_dim_node = 32, #ノード特徴量プロセッサレイヤーの潜在空間dim
                 latent_dim_edge = 32, #エッジ特徴量プロセッサレイヤーの潜在空間dim
                 num_layer_enc_node = 2, #ノード特徴量のエンコーダレイヤー数。以下同様。
                 num_layer_enc_edge = 2,
                 latent_dim_enc_node = 32,
                 latent_dim_enc_edge = 32,
                 in_features_node = 1,
                 in_features_edge = 3,
                 activation_node = nn.ReLU(),
                 activation_edge = nn.ReLU(),
                 activation_enc_node = nn.ReLU(),
                 activation_enc_edge = nn.ReLU(),
                 ):
        super().__init__(
            num_layer,
            latent_dim_node,
            latent_dim_edge,
            num_layer_enc_node,
            num_layer_enc_edge,
            latent_dim_enc_node,
            latent_dim_enc_edge,
            in_features_node,
            in_features_edge,
            activation_node,
            activation_edge,
            activation_enc_node,
            activation_enc_edge)
        self.layer_e_edge = nn.Linear(2*latent_dim_node, 3)

    def forward(self, data):
        #####ノード特徴量エンコード
        h_node = self.activation_enc_node(self.layer_s_enc_node(data.x))
        for layer in self.layers_enc_node:
            h_node = self.activation_enc_node(layer(h_node))
        h_node = self.activation_enc_node(self.layer_e_enc_node(h_node))

        #####エッジ特徴量エンコード
        h_edge = self.activation_enc_edge(self.layer_s_enc_edge(data.edge_attr))
        for layer in self.layers_enc_edge:
            h_edge = self.activation_enc_edge(layer(h_edge))
        h_edge = self.activation_enc_edge(self.layer_e_enc_edge(h_edge))

        #####ノード特徴量とエッジ特徴量のプロセッシング
        index_i = data.edge_index[0]
        index_j = data.edge_index[1]
        h_node = self.layer_s_node(h_node, h_edge, data.edge_index)
        h_edge = torch.cat((h_edge, h_node[index_i], h_node[index_j]), dim = 1)
        h_edge = self.activation_edge(self.layer_s_edge(h_edge))

        for i in range(self.num_layer):
            h_node = self.layers_node[i](h_node, h_edge, data.edge_index)
            h_edge = torch.cat((h_edge, h_node[index_i], h_node[index_j]), dim = 1)
            h_edge = self.activation_edge(self.layers_edge[i](h_edge))
        
        h_node = self.layer_e_node(h_node, h_edge, data.edge_index)

        mask = torch.where(index_i < index_j, 1., -1.).view(index_i.shape[0], 1)
        edge_index_sort, _ = torch.sort(data.edge_index, 0)
        h_edge = torch.cat((h_node[edge_index_sort[0]], h_node[edge_index_sort[1]]), dim = 1)
        h_edge = self.layer_e_edge(h_edge) * mask

        y = data.edge_attr + h_edge

        return y