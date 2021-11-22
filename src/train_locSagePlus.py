from __future__ import print_function, division
from torch import optim
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
from stellargraph.core import StellarGraph
from pandas import Series
from src.utils import config
from src.models.models import FedSage_Plus,LocalSage_Plus
from src.models import feat_loss
from src.classifiers import Classifier
import time
from sklearn import preprocessing, model_selection


class LocalOwner():
    def __init__(self,do_id:int,subG:StellarGraph,node_subjects:Series,
                 all_classes,
                 model_path:list,
                 num_samples:list):

        self.do_id=do_id
        self.subG = subG
        self.num_samples=num_samples # [central_1_hop,central_2_hop]
        self.node_subjects=node_subjects
        self.all_classes=all_classes


        self.feat_shape = subG.node_features()[0].shape[-1]
        self.num_classes = len(all_classes)


        self.train_subjects, self.test_subjects = model_selection.train_test_split(
            self.node_subjects, train_size=0.6, test_size=0.2, stratify=self.node_subjects
        )
        
        self.wn_hide_subjects,self.hide_ids = self.wn_hide_subj()
        self.hasG_hide = self.hide_graph()
        self.n_nodes_hide = len(list(self.hasG_hide.nodes()))
        self.num_pred = config.num_pred

        self.get_train_test_feat_targets()
        self.neighgen=LocalSage_Plus(feat_shape=self.feat_shape,node_len=len(self.all_ids),
                                     n_classes=self.num_classes,node_ids=self.all_ids)

        self.model_path=model_path

        self.optimizer = optim.Adam(self.neighgen.parameters(),
                                    lr=config.lr, weight_decay=config.weight_decay)
        if config.cuda:
            torch.cuda.empty_cache()
            self.neighgen.cuda()
            self.all_feat = self.all_feat.cuda()
            self.adj = self.adj.cuda()
            self.edges=self.edges.cuda()
            self.all_targets_missing = self.all_targets_missing.cuda()
            self.all_targets_feat = self.all_targets_feat.cuda()
            self.train_ilocs = torch.tensor(self.train_ilocs).cuda()
            self.val_ilocs = torch.tensor(self.train_ilocs).cuda()
            self.test_ilocs = torch.tensor(self.test_ilocs).cuda()



    def hide_graph(self):
        self.wn_hide_ids=list(set(self.subG.nodes()).difference(self.hide_ids))
        rm_hide_G= self.subG.subgraph(self.wn_hide_ids)
        return rm_hide_G


    def wn_hide_subj(self):
        hide_len=int((len(list(self.node_subjects.index))-
                      len(list(self.train_subjects.index))-
                      len(list(self.test_subjects.index)))*config.hidden_portion)
        could_hide_ids=self.node_subjects.index.difference(
            self.train_subjects.index).difference(self.test_subjects.index)

        hide_ids=np.random.choice(could_hide_ids,hide_len,replace=False)
        return self.node_subjects[self.node_subjects.index.difference(hide_ids).difference(
            self.train_subjects.index).difference(self.test_subjects.index)],hide_ids


    def get_adj(self,edges,node_len):
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                 shape=(node_len, node_len),
                                 dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        if config.cuda:
            adj=adj.cuda()
        return adj

    def get_train_test_feat_targets(self):
        self.all_ids=list(self.hasG_hide.nodes())
        self.train_ids = self.train_subjects.keys()
        self.test_ids = self.test_subjects.keys()

        self.all_targets_missing=[]
        self.all_targets_feat=[]
        self.all_targets_subj=[]

        self.all_feat = self.hasG_hide.node_features()

        self.train_targets_subj =  preprocessing.label_binarize(self.train_subjects,self.all_classes)
        self.test_targets_subj = preprocessing.label_binarize(self.test_subjects,self.all_classes)
        self.wn_hide_targets_subj = preprocessing.label_binarize(self.wn_hide_subjects,self.all_classes)
        self.all_node_targets_subj = preprocessing.label_binarize(self.node_subjects,self.all_classes)

        for id_i in self.all_ids:
            missing_ids = list(set(self.subG.neighbors(id_i)).difference(list(self.hasG_hide.neighbors(id_i))))
            missing_len = len(missing_ids)

            if missing_len > 0:
                if len(missing_ids)<=self.num_pred:
                    zeros = np.zeros((max(0, self.num_pred - missing_len), self.feat_shape))
                    missing_feat_all = np.vstack((self.subG.node_features(missing_ids), zeros)).\
                        reshape((1, self.num_pred, self.feat_shape))
                else:
                    missing_feat_all = np.copy(self.subG.node_features(missing_ids[:self.num_pred])).\
                        reshape((1, self.num_pred, self.feat_shape))
            else:
                missing_feat_all = np.zeros((1, self.num_pred, self.feat_shape))
            self.all_targets_missing.append(missing_len)
            self.all_targets_feat.append(missing_feat_all)
        self.all_targets_missing = np.asarray(self.all_targets_missing).reshape((-1,1))
        self.all_targets_feat = np.asarray(self.all_targets_feat).reshape((-1, self.num_pred, self.feat_shape))
        self.all_targets_subj = np.asarray(self.all_node_targets_subj).reshape((-1,self.num_classes))

        self.edges=np.asarray(self.hasG_hide.edges(use_ilocs=True))
        self.adj = self.get_adj(self.edges,len(self.all_ids))
        self.edges = torch.tensor(self.edges.astype(np.int32))
        self.all_feat=torch.tensor(self.all_feat)
        self.all_targets_missing = torch.tensor(self.all_targets_missing)
        self.all_targets_feat = torch.tensor(self.all_targets_feat)
        self.all_targets_subj=torch.tensor(self.all_targets_subj)
        self.train_ilocs=self.hasG_hide.node_ids_to_ilocs(self.train_ids).astype(np.int32)
        self.test_ilocs = self.hasG_hide.node_ids_to_ilocs(self.test_ids).astype(np.int32)
        return self.adj, self.all_feat,self.edges, \
               [self.all_targets_missing,self.all_targets_feat,self.all_targets_subj], \
               self.train_ilocs, self.train_ilocs, self.test_ilocs

    def normalize(self,mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def accuracy_missing(self,output, labels):
        output=output.cpu()
        preds = output.detach().numpy().astype(int)
        correct=0.0
        for pred,label in zip(preds,labels):
            if int(pred)==int(label):
                correct+=1.0
        return correct / len(labels)
    def accuracy(self,pred,true):
        acc=0.0
        for predi,truei in zip(pred,true):
            if torch.argmax(predi) == torch.argmax(truei):
                acc+=1.0
        acc/=len(pred)
        return acc
    def l2_feat(self,output,labels):
        output=output.view(-1,self.num_pred,self.feat_shape)
        return F.mse_loss(
            output, labels).float()


    def greedy_l2_feat(self, pred_feats, true_feats, pred_missing, true_missing):
        pred_feats = pred_feats.view(-1, self.num_pred, self.feat_shape)
        return feat_loss.greedy_loss(pred_feats, true_feats, pred_missing, true_missing).unsqueeze(0).mean().float()


    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def train_neighgen(self,epoch):
        t = time.time()
        self.neighgen.train()
        self.optimizer.zero_grad()
        input_feat=self.all_feat
        input_edge=self.edges
        input_adj=self.adj
        output_missing,output_feat,output_nc = self.neighgen(input_feat, input_edge,input_adj)
        output_missing=torch.flatten(output_missing)
        output_feat=output_feat.view(len(self.all_ids),self.num_pred,self.feat_shape)
        output_nc=output_nc.view(len(self.all_ids)+len(self.all_ids)*self.num_pred,self.num_classes)
        loss_train_missing = F.smooth_l1_loss(output_missing[self.train_ilocs].float(),
                                       self.all_targets_missing[self.train_ilocs].reshape(-1).float())

        loss_train_feat = feat_loss.greedy_loss(output_feat[self.train_ilocs],
                                           self.all_targets_feat[self.train_ilocs],
                                           output_missing[self.train_ilocs],
                                           self.all_targets_missing[self.train_ilocs]).unsqueeze(0).mean().float()

        true_nc_label=torch.argmax(self.all_targets_subj[self.train_ilocs], dim=1).view(-1)
        if config.cuda:
            true_nc_label=true_nc_label.cuda()
        loss_train_label=F.cross_entropy(output_nc[self.train_ilocs],true_nc_label)

        acc_train_missing = self.accuracy_missing(output_missing[self.train_ilocs], self.all_targets_missing[self.train_ilocs])
        acc_train_nc = self.accuracy(output_nc[self.train_ilocs],
                                                  self.all_targets_subj[self.train_ilocs])

        loss=(config.a * loss_train_missing + config.b * loss_train_feat + config.c * loss_train_label).float()
        loss.backward()

        self.optimizer.step()

        self.neighgen.eval()
        val_missing, val_feat,val_nc = self.neighgen(self.all_feat, self.edges,self.adj)
        val_feat = val_feat.view(len(self.all_ids), self.num_pred, self.feat_shape)
        acc_val_missing = self.accuracy_missing(val_missing[self.train_ilocs], self.all_targets_missing[self.train_ilocs])
        l2_val_feat = self.greedy_l2_feat(val_feat[self.train_ilocs],
                                   self.all_targets_feat[self.train_ilocs],
                                   val_missing[self.train_ilocs]
                                   ,self.all_targets_missing[self.train_ilocs]
                                   )
        acc_val_nc = self.accuracy(val_nc[self.train_ilocs],
                                                self.all_targets_subj[self.train_ilocs])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'missing_train: {:.4f}'.format(acc_train_missing),
              'nc_train: {:.4f}'.format(acc_train_nc),
              'loss_miss: {:.4f}'.format(loss_train_missing.item()),
              'loss_nc: {:.4f}'.format(loss_train_label.item()),
              'loss_feat: {:.4f}'.format(loss_train_feat.item()),
              'missing_val: {:.4f}'.format(acc_val_missing),
              'nc_val: {:.4f}'.format(acc_val_nc),
              'l2_val: {:.4f}'.format(l2_val_feat),
              'time: {:.4f}s'.format(time.time() - t))





    def train(self):
        for epoch in range(config.gen_epochs):
            self.train_neighgen(epoch)

        print("NeighGen Finished!")


    def save_model(self):
        torch.save(self.neighgen,self.model_path[1])
    def load_model(self):
        if config.cuda:
            self.neighgen = torch.load(self.model_path[1],map_location=torch.device('cuda'))
        else:
            self.neighgen=torch.load(self.model_path[1],map_location=torch.device('cpu'))

    def evaluate_downstream(self, classifier: Classifier, test_flag=False, save_flag=False,
                            global_task=None):
        pred_missing,pred_feats=classifier.eval_pred_Gnew(generator_model=self.neighgen,
                                impaired_graph=self.hasG_hide,org_graph=self.subG,
                                 all_feat=self.all_feat, edges=self.edges, adj=self.adj,
                                 acc_path=classifier.acc_path, test_flag=test_flag,
                                  save_flag=save_flag,global_task=global_task)
        return pred_missing,pred_feats


    def set_fed_model(self):
        self.fed_model=FedSage_Plus(self.neighgen)

    def save_fed_model(self):
        torch.save(self.fed_model,self.model_path[0])
    def load_fed_model(self):
        if config.cuda:
            self.fed_model = torch.load(self.model_path[0])
        else:
            self.fed_model=torch.load(self.model_path[0],map_location=torch.device('cpu'))