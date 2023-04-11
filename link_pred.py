import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import ddi_data
import main
import tqdm
import argparse
import numpy as np

class MLPPredictor(nn.Module):
    def __init__(self, crossmol, use_ca=True):
        super().__init__()
        
        self.crossmol = crossmol
        self.use_ca = use_ca
        if self.use_ca:
            h_feats = 768
        else:
            h_feats = 1200

        self.W1 = nn.Linear(h_feats*2, 256)
        self.W2 = nn.Linear(256, 128)
        self.W3 = nn.Linear(128, 1)

    def get_features(self, batch_graph, text):
        with torch.no_grad():
            if self.use_ca:
                e1_mol, e1_text, e2_text, e3_mol = self.crossmol.forward(text, batch_graph.get_components())
                return e3_mol
            else:
                # else just use grover embedding
                f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch_graph.get_components()
                grover_output = self.crossmol.grover_model.grover((f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a))
                f_atoms = grover_output['atom_from_atom']

                mol_features_simple = f_atoms
                atom_sizes = [i[1] for i in a_scope]
                mol_features_simple = torch.split(mol_features_simple[1:], atom_sizes)

                mol_features_simple_mean = []
                for idx in range(len(atom_sizes)):
                    mol_features_simple_mean.append(torch.mean(mol_features_simple[idx],dim=0))

                e3_mol = torch.stack(mol_features_simple_mean)
                return e3_mol


    def forward(self, batch_graph_1, batch_graph_2, text):
        g = self.get_features(batch_graph_1, text)
        h = self.get_features(batch_graph_2, text)

        feat = F.leaky_relu(self.W1(torch.cat([g,h], dim=1)))
        feat = F.leaky_relu(self.W2(feat))
        preds = self.W3(feat)

        # feat_g = F.leaky_relu(self.W1(g))
        # feat_g = F.leaky_relu(self.W2(feat_g))
        # feat_g = self.W3(feat_g)
        # feat_h = F.leaky_relu(self.W1(h))
        # feat_h = F.leaky_relu(self.W2(feat_h))
        # feat_h = self.W3(feat_h)
        # preds = (feat_g * feat_h).sum(dim=1)
        return preds

# def compute_auc(pos_score, neg_score):
#     scores = torch.cat([pos_score, neg_score]).numpy()
#     labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
#     return roc_auc_score(y_true=labels, y_score=scores)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-use_ca", action='store_true', default=False)
    main_args = parser.parse_args()

    device = torch.device('cuda')

    # in this case, loss will in training loop
    grover_args = main.get_grover_args()
    num_ca_layers = 2
    crossmol = main.CrossMol(grover_args=grover_args, \
        grover_path='./grover/grover_large.pt', bert_path='dmis-lab/biobert-base-cased-v1.2', \
        num_ca_layers=num_ca_layers, device=device)
    crossmol.load_state_dict(torch.load("trained_models/epoch_29_val_loss_0.33_with_ca.pth"))
    # crossmol.load_state_dict(torch.load("trained_models/epoch_19_val_loss_1.00_no_ca.pth"))

    data = ddi_data.get_ddi_dataset()
    dataloader_train, dataloader_test = ddi_data.get_dataloaders(data, batch_size=128)

    model = MLPPredictor(crossmol=crossmol, use_ca=main_args.use_ca)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # # ----------- 4. training -------------------------------- #
    # n_epochs = 1

    # model.train()
    # # model.load_state_dict(torch.load('link_pred_ca={}.pth'.format(str(main_args.use_ca))))
    # # criterion = torch.nn.CrossEntropyLoss()

    # for e in range(n_epochs):
    #     # forward
    #     for batch in tqdm.tqdm(dataloader_train):
    #         if batch[0] is None:
    #             continue
    #         optimizer.zero_grad()
            
    #         scores = model(batch[0], batch[1], batch[2])
    #         labels = batch[3].unsqueeze(dim=1).to(device)
    #         # print(scores.shape, labels.shape, len(pos_batch[2])); quit()
    #         loss = F.binary_cross_entropy_with_logits(scores, labels)
    #         # loss = criterion(scores, labels)
    #         tqdm.tqdm.write(str(loss))
            
    #         loss.backward()
    #         optimizer.step()
    #         # break

    # torch.save(model.state_dict(), 'link_pred_ca={}.pth'.format(str(main_args.use_ca)))
    
    # ----------- 4. testing -------------------------------- #
    model.load_state_dict(torch.load('link_pred_ca={}.pth'.format(str(main_args.use_ca))))
    scores = []
    labels = []
    u_list = [] # mol u linked with mol v
    v_list = []
    
    with torch.no_grad():
        model.eval()
        for batch in tqdm.tqdm(dataloader_test):
            if batch[0] is None:
                continue

            # # scores.append(model(batch[0], batch[1], batch[2]).cpu())
            # u = model.get_features(batch[0], batch[2]).cpu().numpy()
            # v = model.get_features(batch[1], batch[2]).cpu().numpy()
            # u_list.append(u)
            # v_list.append(v)
            labels.append(batch[3].long())
            # break
        
        # print(torch.cat(labels).shape, torch.cat(scores).shape)
        # print('AUC', roc_auc_score(y_true=torch.cat(labels).numpy(), y_score=torch.cat(scores).numpy()))

        # np.save('link_pred_ca={}_u_list.npy'.format(str(main_args.use_ca)), np.concatenate(u_list))
        # np.save('link_pred_ca={}_v_list.npy'.format(str(main_args.use_ca)), np.concatenate(v_list))
        np.save('link_pred_ca={}_labels.npy'.format(str(main_args.use_ca)), np.concatenate(labels))
