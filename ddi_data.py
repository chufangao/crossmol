from ogb.linkproppred import DglLinkPropPredDataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from rdkit import Chem
import random
import torch
import sys
sys.path.append("./grover")
from grover.data import mol2graph


class CrossMolDatasetDDI(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        smile_1 = self.data[idx][0]
        smile_2 = self.data[idx][1]
        label = self.data[idx][2]
        return smile_1, smile_2, label

def collate(batch_list):
    args = Namespace()
    args.bond_drop_rate = 0
    args.no_cache = True
    shared_dict = {}

    edge_1_smiles = []
    edge_2_smiles = []
    labels = []
    # try:
    for smile1, smile2, label in batch_list:
        try:
            # print(type(smile1), type(smile2))
            edge_1_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile1))
            edge_2_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile2))
            # mol2graph([smile1], shared_dict, args)
            # mol2graph([smile2], shared_dict, args)

            edge_1_smiles.append(edge_1_smile)
            edge_2_smiles.append(edge_2_smile)
            labels.append(label)
        except Exception as e:
            # print(e)
            continue

    if len(edge_2_smiles) == 0:
        return None, None, None
    else:

        batched_graph_object_1 = mol2graph(edge_1_smiles, shared_dict, args)
        batched_graph_object_2 = mol2graph(edge_2_smiles, shared_dict, args)
        return batched_graph_object_1, batched_graph_object_2, ["None"]*len(edge_2_smiles), torch.Tensor(labels)

    # except Exception as e:
    #     return None, None
        
def get_dataloaders(data, batch_size=64):
    dataset_train = CrossMolDatasetDDI(data["train"])
    dataset_test = CrossMolDatasetDDI(data["test"])
 
    # CREATE 
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate)

    return dataloader_train, dataloader_test

def get_data(edge_list, id_map, drugbank_map, mode):
    assert mode in ['positive', 'negative']

    u_list = []
    v_list = []
    for edge in edge_list:
        u = edge[0].item()
        v = edge[1].item()
        u_db_id = id_map[u]
        v_db_id = id_map[v]
        if u_db_id in drugbank_map.keys() and v_db_id in drugbank_map.keys():
            u_list.append(drugbank_map[u_db_id])
            v_list.append(drugbank_map[v_db_id])

    if mode == 'negative':
        random.shuffle(v_list)

    data = []
    label = 0 if mode == 'negative' else 1
    for i in range(len(u_list)):
        data.append([u_list[i], v_list[i], label])

    return data

def get_ddi_dataset():
    dataset = DglLinkPropPredDataset(name="ogbl-ddi")

    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]

    id_map = pd.read_csv("dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz")
    id_map = dict(zip(id_map["node idx"],id_map["drug id"]))
    drugbank_map = pd.read_csv("dataset/drugbank_map.csv")
    drugbank_map = drugbank_map[(~drugbank_map["SMILES"].isna()) & (~drugbank_map["DrugBank ID"].isna())]

    drugbank_map = dict(zip(list(drugbank_map["DrugBank ID"]), list(drugbank_map["SMILES"])))

    train_pos_data = get_data(valid_edge["edge"], id_map, drugbank_map, mode='positive')
    train_neg_data = get_data(valid_edge["edge"], id_map, drugbank_map, mode='negative')
    test_pos_data = get_data(test_edge["edge"], id_map, drugbank_map, mode='positive')
    test_neg_data = get_data(test_edge["edge"], id_map, drugbank_map, mode='negative')

    data = {}
    data["train"] = train_pos_data + train_neg_data
    data["test"] = test_pos_data + test_neg_data

    return data