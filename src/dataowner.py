import numpy as np
import pandas as pd
import os
import stellargraph as sg
from src.utils import config
import dill as pickle


MIN=1e-10
class DataOwner:
    def __init__(self,do_id,subG=None,sub_ids=None,node_subj=None,node_target=None,
                 pubG=None,pub_subj=None,pub_target=None,pub_ids=None):

        self.do_id=do_id
        edges = np.copy(subG.edges())
        self.edges = pd.DataFrame()
        self.edges['source'] = [edge[0] for edge in edges]
        self.edges['target'] = [edge[1] for edge in edges]

        self.node_ids=subG.nodes()
        self.node_feats = np.copy(subG.node_features())
        if pubG is not None:
            self.node_feats=np.copy(subG.node_features()+pubG.node_features())
        self.nodes=sg.IndexedArray(self.node_feats, self.node_ids)
        self.G=sg.StellarGraph(nodes=self.nodes, edges=self.edges)
        self.lost_nei={id:[] for id in self.node_ids}

        self.feat_shape = self.G.node_features()[0].shape
        self.node_target=node_target
        if pub_target is not None:
            self.node_target=node_target+pub_target
        self.target_shape=node_target.shape[-1]

        self.has_node_ids=set(sub_ids)
        if pub_ids is not None:
            self.has_node_ids.update(pub_ids)
        self.has_node_ids=list(self.has_node_ids)
        self.no_feat_ids = set(self.node_ids)
        self.no_feat_ids.difference(sub_ids)



        self.node_subj = pd.Series.copy(node_subj)
        if pub_subj is not None:
            for node_id in self.node_ids:
                if pub_subj.values[0].__class__ == str :
                    if pub_subj[node_id] != "":
                        self.node_subj[node_id] += pub_subj[node_id]
                else:
                    if pub_subj[node_id] != 0:
                        self.node_subj[node_id] += pub_subj[node_id]

        self.get_lost_nei_ids()
        self.construct_extendG()
        self.construct_hasG()

        self.info_path = config.local_dataowner_info_dir + "_" + str(self.do_id) +".pkl"
        self.test_acc_path = config.local_test_acc_dir + "_" + str(self.do_id) + ".txt"

        self.get_edge_nodes()

        self.fedgen_model_path = config.local_gen_dir + "_" + str(self.do_id) + "_fedg.h5"
        self.gen_model_path = config.local_gen_dir + "_" + str(self.do_id) + "_g.h5"




    def set_classifier_path(self):
        self.classifier_path = [
            config.local_gen_dir + "_" + str(self.do_id)+ "_classifier_locsage.h5",
            config.local_gen_dir + "_" + str(self.do_id) +  "_classifier_"]
        self.downstream_task_path=config.local_downstream_task_dir+ "_" + str(self.do_id) +  ".pkl"

    def set_gan_path(self):
        self.fedgen_model_path = config.local_gen_dir + "_" + str(self.do_id) + "_fedg.h5"
        self.gen_model_path = config.local_gen_dir + "_" + str(self.do_id) + "_g.h5"

    def get_lost_nei_ids(self):
        self.nei_ids=set()
        self.has_edges=pd.DataFrame()
        has_edges_source=[]
        has_edges_target = []
        self.extend_edges=pd.DataFrame()
        extend_edges_source = []
        extend_edges_target = []

        for edge_i in range(len(self.edges)):
            ilocs=self.G.node_ids_to_ilocs([self.edges['source'][edge_i],self.edges['target'][edge_i]])
            if self.edges['source'][edge_i] in self.has_node_ids and self.edges['target'][edge_i] not in self.has_node_ids:
                self.lost_nei[self.edges['source'][edge_i]].append(self.edges['target'][edge_i])
                self.no_feat_ids.add(self.edges['target'][edge_i])
                self.nei_ids.add(self.edges['target'][edge_i])
                a = {"source": self.edges['source'][edge_i], "target": self.edges['target'][edge_i]}
                extend_edges_source.append(a['source'])
                extend_edges_target.append(a['target'])
            elif self.edges['target'][edge_i] in self.has_node_ids and self.edges['source'][edge_i] not in self.has_node_ids:
                self.lost_nei[self.edges['target'][edge_i]].append(self.edges['source'][edge_i])
                self.no_feat_ids.add(self.edges['source'][edge_i])
                self.nei_ids.add(self.edges['source'][edge_i])
                a = {"source": self.edges['source'][edge_i], "target": self.edges['target'][edge_i]}
                extend_edges_source.append(a['source'])
                extend_edges_target.append(a['target'])
            elif self.edges['source'][edge_i] not in self.has_node_ids and self.edges['target'][edge_i] not in self.has_node_ids:
                self.no_feat_ids.add(self.edges['target'][edge_i])
                self.no_feat_ids.add(self.edges['source'][edge_i])
            else:
                a = {"source": self.edges['source'][edge_i], "target": self.edges['target'][edge_i]}
                extend_edges_source.append(a['source'])
                extend_edges_target.append(a['target'])
                has_edges_source.append(a['source'])
                has_edges_target.append(a['target'])
        self.has_edges['source']=[edge for edge in has_edges_source]
        self.has_edges['target'] = [edge for edge in has_edges_target]
        self.extend_edges['source'] = [edge for edge in extend_edges_source]
        self.extend_edges['target'] = [edge for edge in extend_edges_target]
        self.extend_emb_shape_for_gan=(len(self.nei_ids)+len(self.node_ids)-len(self.no_feat_ids),self.feat_shape)

        self.extend_node_ids=set()
        self.extend_node_ids.update(self.nei_ids)
        self.extend_node_ids.update(self.has_node_ids)
        self.extend_node_ids=list(self.extend_node_ids)
        self.extend_subj = pd.Series.copy(self.node_subj[self.extend_node_ids])
        self.has_subj = pd.Series.copy(self.node_subj[self.has_node_ids])

    def construct_extendG(self):
        self.extend_nodes = sg.IndexedArray(self.G.node_features(self.extend_node_ids), self.extend_node_ids)
        self.extendG = sg.StellarGraph(nodes=self.extend_nodes, edges=self.extend_edges)

    def construct_hasG(self):
        self.has_nodes = sg.IndexedArray(self.G.node_features(self.has_node_ids), self.has_node_ids)
        self.hasG = self.G.subgraph(self.has_node_ids)


    def get_pos_pairs(self,global_pos_pairs):
        pos_pairs=[]
        for pair in global_pos_pairs:
            ids=self.G.node_ilocs_to_ids(pair)
            if ids[0] in self.has_node_ids and ids[1] in self.has_node_ids:
                pos_pairs.append(pair)
        self.pos_pairs=list.copy(pos_pairs)

    def get_neg_pairs(self):
        neg_pairs=[]
        for inner,outter in zip(self.lost_nei.keys(),self.lost_nei.values()):
            if len(outter)>0:
                for outter_i in outter:
                    ilocs=self.G.node_ids_to_ilocs([inner,outter_i])
                    neg_pairs.append(ilocs)
        self.neg_pairs=list.copy(neg_pairs)



    def save_do_info(self):
        if os.path.isfile(self.info_path):
            return
        pickle.dump(self, open(self.info_path, "wb"))
        return





    def get_edge_nodes(self):
        self.edge_nodes=[]

        for id_i in self.node_ids:
            if len(self.lost_nei[id_i])>0:
                self.edge_nodes.append(id_i)
        self.edge_subj=pd.Series.copy(self.node_subj[self.edge_nodes])

