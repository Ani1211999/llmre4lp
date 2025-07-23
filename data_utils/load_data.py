import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
import torch
import random
import json
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
import networkx
from typing import Tuple, List, Dict, Set, Any
 
FILE_PATH = 'data/'
def load_graph_cora(use_mask) -> Data:
    path = f'{FILE_PATH}cora_orig/cora'
    idx_features_labels = np.genfromtxt(f"{path}.content", dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)

    # if use_mask:
    #     labels = idx_features_labels[:, -1]
    #     class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
    #                                              'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning',
    #                                              'Theory'])}
    #     data_Y = np.array([class_map[l] for l in labels])

    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}.cites", dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    #data_edges = np.vstack((data_edges, np.fliplr(data_edges)))

    # dataset = Planetoid('./generated_dataset', 'cora',
    #                    transform=T.NormalizeFeatures())

    x = torch.tensor(data_X).float()
    edge_index = torch.LongTensor(data_edges).T.clone().detach().long()
    num_nodes = len(data_X)

    # if use_mask:
    #     y = torch.tensor(data_Y).long()

    # train_id, val_id, test_id, train_mask, val_mask, test_mask = get_node_mask(num_nodes)

    # if use_mask:
    #     return Data(x=x,
    #                 edge_index=edge_index,
    #                 y=y,
    #                 num_nodes=num_nodes,
    #                 train_mask=train_mask,
    #                 test_mask=test_mask,
    #                 val_mask=val_mask,
    #                 node_attrs=x,
    #                 edge_attrs=None,
    #                 graph_attrs=None,
    #                 train_id=train_id,
    #                 val_id=val_id,
    #                 test_id=test_id
    #                 ), data_citeid

   
    return Data(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_attrs=x,
        edge_attrs=None,
        graph_attrs=None,
    ), data_citeid

def load_text_cora(data_citeid) -> List[str]:
    with open(f'{FILE_PATH}cora_orig/mccallum/cora/papers') as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = f'{FILE_PATH}cora_orig/mccallum/cora/extractions/'

    # for debug
    # save file list
    # with open('extractions.txt', 'w') as txt_file:
    #     # Write each file name to the text file
    #     for file_name in os.listdir(path):
    #         txt_file.write(file_name + '\n')

    text = []
    not_loaded = []
    i, j = 0, 0
    for pid in data_citeid:
        fn = pid_filename[pid]
        try:
            if os.path.exists(path + fn):
                pathfn = path + fn
            elif os.path.exists(path + fn.replace(":", "_")):
                pathfn = path + fn.replace(":", "_")
            elif os.path.exists(path + fn.replace("_", ":")):
                pathfn = path + fn.replace("_", ":")

            with open(pathfn) as f:
                lines = f.read().splitlines()

            for line in lines:
                if 'Title:' in line:
                    ti = line
                if 'Abstract:' in line:
                    ab = line
            text.append(ti + '\n' + ab)
        except Exception:
            not_loaded.append(pathfn)
            i += 1

    print(f"not loaded {i} papers.")
    print(f"not loaded {i} paperid.")
    return text

def load_tag_cora() -> Tuple[Data, List[str]]:
    data, data_citeid = load_graph_cora(use_mask=False)  # nc True, lp False

    text = load_text_cora(data_citeid)
    print(f"Number of texts: {len(text)}")
    print(f"first text: {text[0]}")
    return data, text