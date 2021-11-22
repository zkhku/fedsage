import louvain.community as community_louvain
from stellargraph.core.graph import StellarGraph
import stellargraph as sg
import numpy as np
from sklearn import preprocessing
import pandas as pd
from src.utils import config


def louvain_graph_cut(whole_graph:StellarGraph,node_subjects):
    edges = np.copy(whole_graph.edges())
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in edges]
    df['target'] = [edge[1] for edge in edges]
    G = StellarGraph.to_networkx(whole_graph)

    partition = community_louvain.best_partition(G)

    groups=[]

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)
    partition_groups = {group_i:[] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    group_len_max=len(list(whole_graph.nodes()))//config.num_owners-config.delta
    for group_i in groups:
        while len(partition_groups[group_i])>group_len_max:
            long_group=list.copy(partition_groups[group_i])
            partition_groups[group_i]=list.copy(long_group[:group_len_max])
            new_grp_i=max(groups)+1
            groups.append(new_grp_i)
            partition_groups[new_grp_i]=long_group[group_len_max:]

    print(groups)

    len_list=[]
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict={}

    for i in range(len(groups)):
        len_dict[groups[i]]=len_list[i]
    sort_len_dict={k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1],reverse=True)}

    owner_node_ids={owner_id:[] for owner_id in range(config.num_owners)}

    owner_nodes_len=len(list(G.nodes()))//config.num_owners
    owner_list=[i for i in range(config.num_owners)]
    owner_ind=0


    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) >= owner_nodes_len + config.delta:
            owner_ind = (owner_ind + 1) % len(owner_list)
        owner_node_ids[owner_list[owner_ind]]+=partition_groups[group_i]

    for owner_i in owner_node_ids.keys():
        print('nodes len for '+str(owner_i)+' = '+str(len(owner_node_ids[owner_i])))

    nodes_id = whole_graph.nodes()
    local_G = []
    local_node_subj = []
    local_nodes_ids = []
    target_encoding = preprocessing.LabelBinarizer()
    target = target_encoding.fit_transform(node_subjects)
    local_target = []
    subj_set = list(set(node_subjects.values))
    local_node_subj_0=[]
    for owner_i in range(config.num_owners):
        partition_i = owner_node_ids[owner_i]
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        sbj_i = node_subjects.copy(deep=True)
        sbj_i.values[:] = "" if node_subjects.values[0].__class__ == str else 0
        sbj_i.values[locs_i] = node_subjects.values[locs_i]
        local_node_subj_0.append(sbj_i)
    count=[]
    for owner_i in range(config.num_owners):
        count_i={k:[] for k in subj_set}
        sbj_i=local_node_subj_0[owner_i]
        for i in sbj_i.index:
            if sbj_i[i]!=0 and sbj_i[i]!="":
                count_i[sbj_i[i]].append(i)
        count.append(count_i)
    for k in subj_set:
        for owner_i in range(config.num_owners):
            if len(count[owner_i][k])<2:
                for j in range(config.num_owners):
                    if len(count[j][k])>2:
                        id=count[j][k][-1]
                        count[j][k].remove(id)
                        count[owner_i][k].append(id)
                        owner_node_ids[owner_i].append(id)
                        owner_node_ids[j].remove(id)
                        j=config.num_owners



    for owner_i in range(config.num_owners):
        partition_i =owner_node_ids[owner_i]
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        sbj_i = node_subjects.copy(deep=True)
        sbj_i.values[:] = "" if node_subjects.values[0].__class__ == str else 0
        sbj_i.values[locs_i] = node_subjects.values[locs_i]

        local_node_subj.append(sbj_i)
        local_target_i = np.zeros(target.shape, np.int32)
        local_target_i[locs_i] += target[locs_i]
        local_target.append(local_target_i)
        local_nodes_ids.append(partition_i)

        feats_i = np.zeros(whole_graph.node_features().shape)
        feats_i[locs_i] = feats_i[locs_i] + whole_graph.node_features()[locs_i]

        nodes = sg.IndexedArray(feats_i, nodes_id)
        graph_i = StellarGraph(nodes=nodes, edges=df)
        local_G.append(graph_i)


    return local_G, local_node_subj, local_target, local_nodes_ids