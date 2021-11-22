import pandas as pd
import numpy as np
from typing import Dict,  Any
import warnings
from stellargraph.core import StellarGraph
import stellargraph as sg

def normalize(edge):
    n1, n2 = edge
    if n1 > n2:
        n1, n2 = n2, n1
    return (n1, n2)

def from_flat_dict(data_dict: Dict[str, Any]):
    init_dict = {}
    del_entries = []
    features=[]
    edge_iloc=[]
    for key in data_dict.keys():
        if key.endswith('_data') or key.endswith('.data'):
            if key.endswith('_data'):
                sep = '_'
                warnings.warn(
                    "The separator used for sparse matrices during export (for .npz files) "
                    "is now '.' instead of '_'. Please update (re-save) your stored graphs.",
                    DeprecationWarning, stacklevel=2)
            else:
                sep = '.'
            matrix_name = key[:-5]
            mat_data = key
            mat_indices = '{}{}indices'.format(matrix_name, sep)
            mat_indptr = '{}{}indptr'.format(matrix_name, sep)
            mat_shape = '{}{}shape'.format(matrix_name, sep)
            if matrix_name == 'adj' or matrix_name == 'attr':
                warnings.warn(
                    "Matrices are exported (for .npz files) with full names now. "
                    "Please update (re-save) your stored graphs.",
                    DeprecationWarning, stacklevel=2)
                matrix_name += '_matrix'

            if mat_data == "adj_data":
                for i in range(data_dict[mat_shape][0]):
                    for j in range(data_dict[mat_indptr][i],data_dict[mat_indptr][i+1]):
                        edge_iloc.append((i,data_dict[mat_indices][j]))


            elif mat_data == "attr_data":
                for i in range(data_dict[mat_shape][0]):
                    feature_i=np.zeros(data_dict[mat_shape][1],np.float32)
                    feature_i[data_dict[mat_indices]
                              [data_dict[mat_indptr][i]:data_dict[mat_indptr][i + 1]]]+=\
                        data_dict[mat_data][data_dict[mat_indptr][i]:data_dict[mat_indptr][i + 1]]
                    features.append(feature_i)


            del_entries.extend([mat_data, mat_indices, mat_indptr, mat_shape])

    for del_entry in del_entries:
        del data_dict[del_entry]

    for key, val in data_dict.items():
        if ((val is not None) and (None not in val)):
            init_dict[key] = val

    node_subj = {}
    for node_id,subj in zip(init_dict["node_names"],init_dict["labels"]):
        node_subj[node_id]=init_dict["class_names"][subj]
    edge_source_ids=[]
    edge_target_ids=[]
    unique_edges = list(set(map(normalize, edge_iloc)))

    for i in range(len(unique_edges)):
        edge_target_ids.append(init_dict["node_names"][unique_edges[i][0]])
        edge_source_ids.append(init_dict["node_names"][unique_edges[i][1]])

    nodes=sg.IndexedArray(values=np.asarray(features).reshape((len(init_dict["node_names"]),-1)),
                          index=init_dict["node_names"])

    node_subjects=pd.Series(node_subj)
    edges = pd.DataFrame()
    edges['source'] = [edge for edge in edge_source_ids]
    edges['target'] = [edge for edge in edge_target_ids]
    G=StellarGraph(nodes=nodes,edges=edges)


    return G,node_subjects

def load_from_npz(file_name: str) :
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        G,node_subjects = from_flat_dict(loader)
    return G,node_subjects


