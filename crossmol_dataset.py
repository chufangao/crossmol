import sys
sys.path.append("./grover")

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from grover.data import mol2graph

class CrossMolDataset(Dataset):

    def __init__(self,dataset_path,mode):
        self.data_df = pd.read_csv(dataset_path,delimiter="\t")
        self.data_df = self.data_df[self.data_df["mode"] == mode]
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self,idx):
        smile = self.data_df.iloc[idx]["SMILES"]
        description = self.data_df.iloc[idx]["description"]
        return smile,description

def collate(batch_list):
    args = Namespace()
    args.bond_drop_rate = 0
    args.no_cache = True
    shared_dict = {}
    try:
        batched_graph_object = mol2graph([smile for smile,text in batch_list],shared_dict,args)
        descriptions = [text for smile,text in batch_list]
        return batched_graph_object,descriptions
    except Exception as e:
        return None, None
        
def get_dataloaders(data_path,batch_size):

    dataset_train = CrossMolDataset(data_path,"train")
    dataset_val = CrossMolDataset(data_path,"val")
    dataset_test = CrossMolDataset(data_path,"test")

    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, collate_fn=collate, shuffle=True
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, collate_fn=collate
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, collate_fn=collate
    )
    return dataloader_train, dataloader_val, dataloader_test

if __name__=='__main__':
    data_path = "chebi20.csv"
    batch_size = 64
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders(data_path,batch_size)

    for graph,text in dataloader_test:
        print(graph,text)

