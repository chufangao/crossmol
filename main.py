import argparse 
import torch
import transformers
from tqdm import tqdm
import copy
import numpy as np
import datetime
import crossmol_dataset

import sys
sys.path.append('./grover/')
import grover.util.parsing
import grover.model.models
import task.train


class CrossEncoderLayer(torch.nn.Module):
    def __init__(self, grover_args, bert_config, hidden_dim=768, nhead=8, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=1)

        self.grover_layer = grover.model.layers.GTransEncoder(
            args=grover_args,
            hidden_size=hidden_dim,
            edge_fdim=hidden_dim,
            node_fdim=hidden_dim,
            dropout=grover_args.dropout,
            activation=grover_args.activation,
            num_mt_block=1,
            num_attn_head=grover_args.num_attn_head,
            atom_emb_output="atom",
            bias=grover_args.bias,
            cuda=grover_args.cuda)
            
        self.bert_layer = transformers.models.bert.BertLayer(config=bert_config)
    

    def forward(self, new_text_batch, new_grover_batch,input_type="both"):

        last_hidden_state, extended_attention_mask, attention_mask = new_text_batch
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = new_grover_batch
        
        if input_type == "both" or input_type == "text":
            last_hidden_state = self.bert_layer(last_hidden_state, extended_attention_mask)[0]
            text_sizes = [sum(i).item() for i in attention_mask]
 
        if input_type == "both" or input_type == "mol":
            grover_out = self.grover_layer((f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a))
            f_atoms = grover_out[0]
            atom_sizes = [i[1] for i in a_scope]
            split_f_atoms = torch.split(f_atoms[1:],atom_sizes)
        
        if input_type == "both":
            total_lengths = [atom_sizes[i]+text_sizes[i] for i in range(len(atom_sizes))]
            combined_features = [torch.cat((split_f_atoms[i],last_hidden_state[i]),dim=0) for i in range(len(atom_sizes))]
            max_feat_size = max([combined_features[i].shape[0] for i in range(len(combined_features))])+1

        if input_type == "mol":
            total_lengths = atom_sizes
            max_feat_size = max(atom_sizes) + 1
            combined_features = [split_f_atoms[i] for i in range(len(atom_sizes))]
        
        if input_type == "text":
            combined_features = last_hidden_state
            max_feat_size = max(text_sizes) + 1
            total_lengths = text_sizes
        
        pad_array = []
        combined_features_padded = []
        for i in range(len(combined_features)):
            feature = combined_features[i]
            feature_size = feature.shape[0]
            pad_tensor = torch.zeros(max_feat_size - feature_size, 768).to(self.device)
            combined_features_padded.append(torch.cat((feature,pad_tensor)))
            pad_array.append([1 for _ in range(total_lengths[i])] + [0 for _ in range(max_feat_size - total_lengths[i])])
            
        pad_array = torch.tensor(pad_array).to(self.device)
        combined_features_padded = torch.stack(combined_features_padded)

        pad_array = pad_array > 0
        trans_out = self.transformer_layer(combined_features_padded,src_key_padding_mask = pad_array)


        for idx in range(len(trans_out)):
            curr_feature = trans_out[idx]
            num_atoms = 0
            
            if input_type == "mol" or input_type == "both":
                begin_atom_idx = a_scope[idx][0]
                num_atoms = atom_sizes[idx]
                f_atoms[begin_atom_idx:begin_atom_idx+num_atoms] = curr_feature[0:num_atoms]
            
            if input_type == "text" or input_type == "both":
                text_size = text_sizes[idx]
                last_hidden_state[idx][0:text_size] = curr_feature[num_atoms:num_atoms+text_size]


        return (last_hidden_state, extended_attention_mask,attention_mask), (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)

class CrossMol(torch.nn.Module):
    def __init__(self, grover_args, grover_path='./grover/grover_large.pt', bert_path='dmis-lab/biobert-base-cased-v1.2', 
        num_ca_layers=2, device='cpu'):
        super().__init__()
        
        self.device = device
        self.num_ca_layers = num_ca_layers
        self.grover_args = grover_args
        
        self.grover_model = grover.util.utils.load_checkpoint(path=grover_path, current_args=grover_args, logger=None)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=bert_path)
        self.bert_model = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=bert_path)
        self.grover_linear_layer = torch.nn.Linear(1200, 768)

        self.cross_encoder_layers = torch.nn.ModuleList()
        for i in range(self.num_ca_layers):
            self.cross_encoder_layers.append(CrossEncoderLayer(grover_args=grover_args, bert_config=self.bert_model.config, device=self.device))

    def forward(self, text_batch, grover_batch):
        # ========== Initial pass through grover molecule encoder ==========
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = grover_batch
        grover_output = self.grover_model.grover((f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a))

        f_atoms = self.grover_linear_layer(grover_output['atom_from_atom'])
        f_bonds = self.grover_linear_layer(grover_output['bond_from_atom'])
        new_grover_batch = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
        
        # ========== Initial pass through bert text encoder ==========
        encoded_input = self.tokenizer(text=text_batch, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
        bert_output  = self.bert_model(
            input_ids=encoded_input['input_ids'], 
            token_type_ids=encoded_input['token_type_ids'], 
            attention_mask=encoded_input['attention_mask'])
        last_hidden_state = bert_output['last_hidden_state']
        attention_mask = encoded_input['attention_mask']
        extended_attention_mask = self.bert_model.get_extended_attention_mask(
            attention_mask=encoded_input['attention_mask'], 
            input_shape=encoded_input['input_ids'].shape)
        new_text_batch = (last_hidden_state, extended_attention_mask, attention_mask)

        new_text_batch_copy_1 = (last_hidden_state.clone(), extended_attention_mask, attention_mask)
        new_grover_batch_copy_1 = (f_atoms.clone(), f_bonds.clone(), a2b, b2a, b2revb, a_scope, b_scope, a2a)

        new_text_batch_copy_2 = (last_hidden_state.clone(), extended_attention_mask, attention_mask)
        new_grover_batch_copy_2 = (f_atoms.clone(), f_bonds.clone(), a2b, b2a, b2revb, a_scope, b_scope, a2a)

        # ========== Cross Attention Encoder Layers ==========
        for layer in self.cross_encoder_layers:
            new_text_batch, new_grover_batch = layer(new_text_batch, new_grover_batch,input_type="both")

        for layer in self.cross_encoder_layers:
            new_text_batch_copy_1, new_grover_batch_copy_1 = layer(new_text_batch_copy_1, new_grover_batch_copy_1,input_type="text")

        for layer in self.cross_encoder_layers:
            new_text_batch_copy_2, new_grover_batch_copy_2 = layer(new_text_batch_copy_2, new_grover_batch_copy_2,input_type="mol")
        
        atom_sizes = [i[1] for i in a_scope]

        # ========== e1 (combined) ==========
        text_features_combined = new_text_batch[0]*attention_mask.unsqueeze(dim=2)
        mol_features_combined = new_grover_batch[0]
        mol_features_combined = torch.split(mol_features_combined[1:],atom_sizes)

        mol_features_combined_mean = []
        for idx in range(len(atom_sizes)):
            mol_features_combined_mean.append(torch.mean(mol_features_combined[idx],dim=0))

        e1_mol = torch.stack(mol_features_combined_mean)
        e1_text = torch.mean(text_features_combined,dim=1 )

        # e1 = (text_features_combined_mean + mol_features_combined_mean)/2

        # ========== e2 (only text) ==========
        text_features_simple = new_text_batch_copy_1[0]*attention_mask.unsqueeze(dim=2)
        # print(text_features_simple.isnan().any(), attention_mask.isnan().any())
        e2_text = torch.mean(text_features_simple,dim=1)
        
        # ========== e3 (only mol) ==========
        mol_features_simple = new_grover_batch_copy_2[0]
        mol_features_simple = torch.split(mol_features_simple[1:],atom_sizes)

        mol_features_simple_mean = []
        for idx in range(len(atom_sizes)):
            mol_features_simple_mean.append(torch.mean(mol_features_simple[idx],dim=0))

        e3_mol = torch.stack(mol_features_simple_mean)

        return e1_mol, e1_text, e2_text, e3_mol

    def cos_dist(self,x,y):
        return 1 - torch.nn.CosineSimilarity(dim=1)(x,y)

    def loss_fn(self, e1_mol, e1_text, e2_text, e3_mol):
        sim = torch.nn.CosineSimilarity(dim=1)

        num = torch.exp(sim(e1_mol, e2_text)) + \
            torch.exp(sim(e1_text, e3_mol)) + \
            torch.exp(sim(e1_mol, e1_text)) + \
            torch.exp(sim(e2_text, e3_mol)) + \
            torch.exp(sim(e1_mol, e3_mol)) + \
            torch.exp(sim(e1_text, e2_text))

        denom = torch.zeros_like(num)
        for pos_ind in range(len(e1_mol)):
            neg_inds = [i for i in range(len(e1_mol)) if i != pos_ind]
            denom[pos_ind] = torch.exp(sim(e1_mol[pos_ind], e1_mol[neg_inds])).sum()

        for pos_ind in range(len(e1_text)):
            neg_inds = [i for i in range(len(e1_text)) if i != pos_ind]
            denom[pos_ind] = torch.exp(sim(e1_text[pos_ind], e1_text[neg_inds])).sum()

        for pos_ind in range(len(e2_text)):
            neg_inds = [i for i in range(len(e2_text)) if i != pos_ind]
            denom[pos_ind] += torch.exp(sim(e2_text[pos_ind], e2_text[neg_inds])).sum()

        for pos_ind in range(len(e3_mol)):
            neg_inds = [i for i in range(len(e3_mol)) if i != pos_ind]
            denom[pos_ind] += torch.exp(sim(e3_mol[pos_ind], e3_mol[neg_inds])).sum()

        loss = -1 * torch.log(torch.divide(num,denom))
        return loss

    def fit(self, train_loader, valid_loader, test_loader, epochs, lr, weight_decay, verbose=True, epoch_offset=0):
        self.to(self.device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss_record = [] 
        # valid_loss, _ = self.test(valid_loader)
        # valid_loss_record = [valid_loss]
        valid_loss_record = [np.inf]
        best_model = copy.deepcopy(self)
        if verbose: print('Done evaluating')
        for ep in tqdm(range(epochs)):
            # ======= Training =======
            for batch in tqdm(train_loader):
                optimizer.zero_grad() 
                graph,text = batch
                if graph is None or text is None:
                    print('Bad batch: error')
                    continue
                f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = graph.get_components()
                f_atoms.to(self.device), f_bonds.to(self.device)
                e1_mol, e1_text, e2_text, e3_mol = self.forward(text, (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a))
                loss = self.loss_fn(e1_mol, e1_text, e2_text, e3_mol).mean()
                train_loss_record.append(loss.detach().cpu().numpy())

                loss.backward() 
                optimizer.step()

                if verbose: tqdm.write('loss: '+str(loss))        
            # ======= Validation =======
            valid_loss, _ = self.test(valid_loader)
            valid_loss_record.append(valid_loss)
            if valid_loss < np.max(valid_loss_record):
                best_model = copy.deepcopy(self)

                torch.save(self.state_dict(), 'trained_models/epoch_{}_val_loss_{:.2f}_with_ca.pth'.format(str(ep+epoch_offset), valid_loss))

        self = copy.deepcopy(best_model)
        test_loss, test_output = self.test(test_loader)
        print("final test loss", test_loss)
        return self, test_loss, test_output

    def test(self, data_loader):
        self.to(self.device)
        self.eval()

        losses = []
        outputs = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                graph, text = batch
                if graph is None or text is None:
                    print('Bad batch: error')    
                    continue

                f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = graph.get_components()
                f_atoms.to(self.device), f_bonds.to(self.device)
                e1_mol, e1_text, e2_text, e3_mol = self.forward(text, (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a))
                loss = self.loss_fn(e1_mol, e1_text, e2_text, e3_mol).mean()

                losses.append(loss.item())
                outputs.append([e1_mol.cpu().numpy(),e1_text.cpu().numpy(), e2_text.cpu().numpy(), e3_mol.cpu().numpy()])
        self.train()

        return np.mean(losses), np.array(outputs)

def get_grover_args():
    ## ========== obtain grover_args ==========
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title="subcommands", dest="parser_name", help="Subcommands for finetune, prediction, and fingerprint.")
    parser_finetune = subparser.add_parser('finetune', help="Fine tune the pre-trained model.")
    grover.util.parsing.add_finetune_args(parser_finetune)
    parser_eval = subparser.add_parser('eval', help="Evaluate the results of the pre-trained model.")
    grover.util.parsing.add_finetune_args(parser_eval)
    parser_predict = subparser.add_parser('predict', help="Predict results from fine tuned model.")
    grover.util.parsing.add_predict_args(parser_predict)
    parser_fp = subparser.add_parser('fingerprint', help="Get the fingerprints of SMILES.")
    grover.util.parsing.add_fingerprint_args(parser_fp)
    parser_pretrain = subparser.add_parser('pretrain', help="Pretrain with unlabelled SMILES.")
    grover.util.parsing.add_pretrain_args(parser_pretrain)
    grover_args = parser.parse_args("finetune --data_path grover/exampledata/finetune/bbbp.csv \
                            --features_path grover/exampledata/finetune/bbbp.npz \
                            --save_dir grover/model/finetune/bbbp/ \
                            --checkpoint_path grover/model/tryout/model.ep3 \
                            --dataset_type classification \
                            --split_type scaffold_balanced \
                            --ensemble_size 1 \
                            --num_folds 3 \
                            --no_features_scaling \
                            --ffn_hidden_size 200 \
                            --batch_size 32 \
                            --epochs 10 \
                            --init_lr 0.00015 \
                            --no_cuda".split())

    grover.util.parsing.modify_train_args(grover_args)
    _ = task.train.load_data(grover_args, print, None)

    return grover_args


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", help="must be in ['train', 'test']", required=True)
    parser.add_argument("-model_path", default=None)
    main_args = parser.parse_args()
    print(main_args)
    assert main_args.mode in ['train', 'test']

    grover_args = get_grover_args()
    data_path = "chebi20.csv"
    batch_size = 8
    epochs = 30
    lr = 1e-5
    weight_decay = False
    num_ca_layers = 2

    dataloader_train, dataloader_val, dataloader_test = crossmol_dataset.get_dataloaders(data_path=data_path, batch_size=batch_size)
    crossmol = CrossMol(grover_args=grover_args, grover_path='./grover/grover_large.pt', bert_path='dmis-lab/biobert-base-cased-v1.2', 
        num_ca_layers=num_ca_layers, device=torch.device('cuda'))

    if main_args.mode == 'train':
        if main_args.model_path is not None: # train from checkpoint
            crossmol.load_state_dict(torch.load(main_args.model_path))

        crossmol, test_loss, test_output = crossmol.fit(dataloader_train, dataloader_val, dataloader_test, 
            epochs, lr, weight_decay)

        # current_time = datetime.datetime.now().strftime("%H:%M:%S")
        # torch.save(crossmol.state_dict(), '{}_{:.6f}_with_ca.pth'.format(current_time, test_loss))

    else: # main_args.mode == 'test':
        assert main_args.model_path is not None
        crossmol.load_state_dict(torch.load(main_args.model_path))
        test_loss, test_output = crossmol.test(dataloader_test)
        np.save('test_output.npy', test_output)
