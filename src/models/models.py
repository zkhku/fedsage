import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import GraphConvolution
from src.utils import config
import torch
import numpy as np
import scipy.sparse as sp

class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, inputs):
        rand = torch.normal(0, 1, size=inputs.shape)
        if config.cuda:
            return inputs + rand.cuda()
        else:
            return inputs + rand

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(self.shape)


class MendGraph(nn.Module):
    def __init__(self, node_len, num_pred, feat_shape,node_ids):
        super(MendGraph, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        self.org_node_len=node_len
        self.node_len=self.org_node_len+self.org_node_len*self.num_pred
        self.node_ids=node_ids
        for param in self.parameters():
            param.requires_grad=False
    def mend_graph(self,org_feats,org_edges,pred_degree,gen_feats):
        new_edges=[]
        gen_feats=gen_feats.view(-1,self.num_pred,self.feat_shape)
        if config.cuda:
            pred_degree=pred_degree.cpu()
        pred_degree=torch._cast_Int(pred_degree).detach()
        org_feats=org_feats.detach()
        fill_feats = torch.vstack((org_feats, gen_feats.view(-1,self.feat_shape)))

        for i in range(self.org_node_len):
            for j in range(min(self.num_pred,max(0,pred_degree[i]))):
                new_edges.append(np.asarray([i,self.org_node_len+i*self.num_pred+j]))

        new_edges=torch.tensor(np.asarray(new_edges).reshape((-1,2)))
        if config.cuda:
            new_edges=new_edges.cuda()
        if len(new_edges)>0:
            fill_edges=torch.vstack((org_edges,new_edges))
        else:
            fill_edges=torch.clone(org_edges)
        return fill_edges,fill_feats

    def get_adj(self,edges):
        if config.cuda:
            edges=edges.cpu()
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                 shape=(self.node_len, self.node_len),
                                 dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        if config.cuda:
            adj=adj.cuda()
        return adj

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self,org_feats,org_edges, pred_missing,gen_feats):
        fill_edges,fill_feats=self.mend_graph(org_feats,org_edges,pred_missing,gen_feats)
        adj=self.get_adj(fill_edges)
        return fill_feats,adj

class LocalSage_Plus(nn.Module):

    def __init__(self, feat_shape,  node_len, n_classes,node_ids):
        super(LocalSage_Plus, self).__init__()
        self.encoder_model = GNN(nfeat=feat_shape,
                                 nhid=config.hidden,
                                 nclass=config.latent_dim,
                                 dropout=config.dropout)
        self.reg_model = RegModel(latent_dim=config.latent_dim)
        self.gen = Gen(latent_dim=config.latent_dim,
                          dropout=config.dropout, num_pred=config.num_pred, feat_shape=feat_shape)
        self.mend_graph=MendGraph(node_len=node_len, num_pred=config.num_pred,
                                  feat_shape=feat_shape, node_ids=node_ids)
        self.classifier=GNN(nfeat=feat_shape,
                            nhid=config.hidden,
                            nclass=n_classes,
                            dropout=config.dropout)


    def forward(self, feat, edges,adj):
        x = self.encoder_model(feat, adj)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats,mend_adj=self.mend_graph(feat,edges, degree,gen_feat)
        nc_pred=self.classifier(mend_feats,mend_adj)
        return degree, gen_feat,nc_pred


class Gen(nn.Module):
    def __init__(self,latent_dim, dropout,num_pred,feat_shape):
        super(Gen, self).__init__()
        self.num_pred=num_pred
        self.feat_shape=feat_shape
        self.sample = Sampling()

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256,2048)
        self.fc_flat = nn.Linear(2048, self.num_pred * self.feat_shape)

        self.dropout = dropout

    def forward(self, x):
        x = self.sample(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        return x

class RegModel(nn.Module):
    def __init__(self,latent_dim):
        super(RegModel,self).__init__()
        self.reg_1 = nn.Linear(latent_dim, 1)

    def forward(self,x):
        x = F.relu(self.reg_1(x))
        return x

class FedSage_Plus(nn.Module):
    def __init__(self, local_graph:LocalSage_Plus):
        super(FedSage_Plus, self).__init__()
        self.encoder_model =local_graph.encoder_model
        self.reg_model = local_graph.reg_model
        self.gen = local_graph.gen
        self.mend_graph=local_graph.mend_graph
        self.classifier=local_graph.classifier
        self.encoder_model.requires_grad_(False)
        self.reg_model.requires_grad_(False)
        self.mend_graph.requires_grad_(False)
        self.classifier.requires_grad_(False)

    def forward(self, feat, edges,adj):
        x = self.encoder_model(feat, adj)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats,mend_adj=self.mend_graph(feat,edges, degree,gen_feat)
        nc_pred=self.classifier(mend_feats,mend_adj)
        return degree, gen_feat,nc_pred
