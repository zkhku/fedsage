import numpy as np
from src.utils import config
from stellargraph.core import StellarGraph
import pandas as pd
import stellargraph as sg
import torch
import scipy.sparse as sp

def accuracy(pred, true):
    acc = 0.0
    for predi, truei in zip(pred, true):
        if np.argmax(predi) == np.argmax(truei):
            acc += 1.0
    acc /= len(pred)
    return acc


def accuracy_missing(pred, true):
    acc = 0.0
    for predi, truei in zip(pred, true):
        if int(predi) == int(truei):
            acc += 1.0
    acc /= len(pred)
    return acc


def fill_graph(impaired_graph:StellarGraph, original_graph:StellarGraph,missing, new_feats,feat_shape):
    new_feats = new_feats.reshape((-1, config.num_pred, feat_shape))
    original_node_ids=[id_i for id_i in original_graph.nodes()]
    fill_node_ids = [id_i for id_i in impaired_graph.nodes()]
    fill_node_feats=[]
    org_feats=original_graph.node_features()
    for i in range(len(list(original_graph.nodes()))):
        fill_node_feats.append(np.asarray(org_feats[i].reshape(-1)))
    org_edges = np.copy(original_graph.edges())
    fill_edges_source = [edge[0] for edge in org_edges]
    fill_edges_target = [edge[1] for edge in org_edges]

    start_id = -1
    for new_i in range(len(missing)):
        if int(missing[new_i]) > 0:
            new_ids_i = np.arange(start_id, start_id - min(config.num_pred, int(missing[new_i])), -1)

            i_pred = 0
            for i in new_ids_i:
                original_node_ids.append(int(i))
                if isinstance(new_feats[new_i][i_pred], np.ndarray) == False:
                    if config.cuda:
                        new_feats = new_feats.cpu()
                    new_feats = new_feats.detach().numpy()
                fill_node_feats.append(np.asarray(new_feats[new_i][i_pred].reshape(-1)))
                i_pred += 1
                fill_edges_source.append(fill_node_ids[new_i])
                fill_edges_target.append(int(i))

            start_id = start_id - min(config.num_pred, int(missing[new_i]))

    fill_edges_source = np.asarray(fill_edges_source).reshape((-1))
    fill_edges_target = np.asarray(fill_edges_target).reshape((-1))
    fill_edges = pd.DataFrame()
    fill_edges['source'] = fill_edges_source
    fill_edges['target'] = fill_edges_target
    fill_node_feats_np = np.asarray(fill_node_feats).reshape((-1,feat_shape))
    fill_node_ids_np = np.asarray(original_node_ids).reshape(-1)

    fill_nodes = sg.IndexedArray(fill_node_feats_np, fill_node_ids_np)
    fill_G = sg.StellarGraph(nodes=fill_nodes, edges=fill_edges)
    return fill_nodes, fill_G




def get_adj(edges, node_len):
    if config.cuda:
        edges=edges.cpu()
    adj = sp.coo_matrix((np.ones(edges.shape[0]),
                         (edges[:, 0], edges[:, 1])),
                        shape=(node_len, node_len),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    if config.cuda:
        adj=adj.cuda()
    return adj


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)