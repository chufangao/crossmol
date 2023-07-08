import argparse 
import sys
sys.path.append('./grover/')
import grover.util.parsing
import grover.model.models
import task.train

import torch
import transformers
from tqdm import tqdm
import copy
import numpy as np
from datetime import datetime
import crossmol_dataset


## Use pretrain config
parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(title="subcommands",
                                    dest="parser_name",
                                    help="Subcommands for finetune, prediction, and fingerprint.")
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
# train_args = grover.util.parsing.get_newest_train_args()
_ = task.train.load_data(grover_args, print, None)

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
        num_ca_layers=5, device='cpu'):
        super().__init__()
        
        self.device = device
        self.num_ca_layers = num_ca_layers
        self.grover_args = grover_args
        
        self.grover_model = grover.util.utils.load_checkpoint(path=grover_path, current_args=grover_args, logger=None)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=bert_path)
        self.bert_model = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=bert_path)
        self.grover_linear_layer = torch.nn.Linear(1200, 768)

        # self.cross_encoder_layers = torch.nn.ModuleList()
        # for i in range(self.num_ca_layers):
        #     self.cross_encoder_layers.append(CrossEncoderLayer(grover_args=grover_args, bert_config=self.bert_model.config, device=self.device))

    def forward(self, text_batch, grover_batch):
        # ========== Initial pass through grover molecule encoder ==========
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = grover_batch
        grover_output = self.grover_model.grover((f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a))

        f_atoms = self.grover_linear_layer(grover_output['atom_from_atom'])
        # f_bonds = self.grover_linear_layer(grover_output['bond_from_atom'])
        # new_grover_batch = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
        
        # ========== Initial pass through bert text encoder ==========
        encoded_input = self.tokenizer(text=text_batch, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
        bert_output  = self.bert_model(
            input_ids=encoded_input['input_ids'], 
            token_type_ids=encoded_input['token_type_ids'], 
            attention_mask=encoded_input['attention_mask'])
        last_hidden_state = bert_output['last_hidden_state']
        attention_mask = encoded_input['attention_mask']
        # extended_attention_mask = self.bert_model.get_extended_attention_mask(
        #     attention_mask=encoded_input['attention_mask'], 
        #     input_shape=encoded_input['input_ids'].shape)
        # new_text_batch = (last_hidden_state, extended_attention_mask, attention_mask)

        atom_sizes = [i[1] for i in a_scope]

        # # ========== e1 (combined) ==========
        # text_features_combined = last_hidden_state*attention_mask.unsqueeze(dim=2)
        # mol_features_combined = f_atoms
        # mol_features_combined = torch.split(mol_features_combined[1:],atom_sizes)

        # mol_features_combined_mean = []
        # for idx in range(len(atom_sizes)):
        #     mol_features_combined_mean.append(torch.mean(mol_features_combined[idx],dim=0))

        # mol_features_combined_mean = torch.stack(mol_features_combined_mean)
        # text_features_combined_mean = torch.mean(text_features_combined,dim=1 )

        # e1 = (text_features_combined_mean + mol_features_combined_mean)/2

        # ========== e2 (only text) ==========
        text_features_simple = last_hidden_state*attention_mask.unsqueeze(dim=2)
        text_features_simple_mean = torch.mean(text_features_simple,dim=1)
        e2 = text_features_simple_mean

        # ========== e3 (only mol) ==========
        mol_features_simple = torch.split(f_atoms[1:], atom_sizes)

        mol_features_simple_mean = []
        for idx in range(len(atom_sizes)):
            mol_features_simple_mean.append(torch.mean(mol_features_simple[idx], dim=0))

        mol_features_simple_mean = torch.stack(mol_features_simple_mean)
        e3 = mol_features_simple_mean

        return None, e2,e3

    # def cos_dist(self,x,y):
    #     return 1 - torch.nn.CosineSimilarity(dim=1)(x,y)

    # def loss_fn(self,e1,e2,e3):
    #     sim = torch.nn.CosineSimilarity(dim=1)

    #     num = torch.exp(sim(e1, e2)) + torch.exp(sim(e2, e3)) + torch.exp(sim(e1, e3))

    #     denom = torch.zeros_like(num)
    #     for pos_ind in range(len(e1)):
    #         neg_inds = [i for i in range(len(e1)) if i != pos_ind]
    #         denom[pos_ind] = torch.exp(sim(e1[pos_ind], e1[neg_inds])).sum()

    #     for pos_ind in range(len(e2)):
    #         neg_inds = [i for i in range(len(e2)) if i != pos_ind]
    #         denom[pos_ind] += torch.exp(sim(e2[pos_ind], e2[neg_inds])).sum()

    #     for pos_ind in range(len(e3)):
    #         neg_inds = [i for i in range(len(e3)) if i != pos_ind]
    #         denom[pos_ind] += torch.exp(sim(e3[pos_ind], e3[neg_inds])).sum()

    #     loss = -1 * torch.log(torch.divide(num,denom))
    #     return loss
    def clip_loss_fn(self, text_embeddings, image_embeddings, temperature=1.0):
        logits = (text_embeddings @ image_embeddings.T) / temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = torch.nn.functional.softmax(
            (images_similarity + texts_similarity) / 2 * temperature, dim=-1
        )
        images_loss = (-targets.T * torch.nn.LogSoftmax(dim=-1)(logits.T)).sum(1)
        texts_loss = (-targets * torch.nn.LogSoftmax(dim=-1)(logits)).sum(1)
        # print(images_loss.shape, texts_loss.shape); quit()
        return (images_loss + texts_loss) / 2.0

    def fit(self, train_loader, valid_loader, test_loader, epochs, lr, weight_decay, verbose=True):
        self.to(self.device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss_record = [] 
        # valid_loss, _ = self.test(valid_loader)
        # valid_loss_record = [valid_loss]
        valid_loss_record = []
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
                e1,e2,e3 = self.forward(text, (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a))
                loss = self.clip_loss_fn(e2,e3).mean()
                train_loss_record.append(loss.detach().cpu().numpy())

                loss.backward() 
                optimizer.step()

                if verbose: tqdm.write('loss: '+str(loss))        
            # ======= Validation =======
            valid_loss, _ = self.test(valid_loader)
            valid_loss_record.append(valid_loss)
            if valid_loss <= np.max(valid_loss_record):
                best_model = copy.deepcopy(self)

                torch.save(self.state_dict(), 'epoch_{}_val_loss_{:.2f}_no_ca.pth'.format(str(ep), valid_loss))

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
                e1,e2,e3 = self.forward(text, (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a))
                loss = self.clip_loss_fn(e2,e3).mean()

                losses.append(loss.item())
                outputs.append([e2.cpu().numpy(), e3.cpu().numpy()])
        self.train()

        return np.mean(losses), np.array(outputs)



crossmol = CrossMol(grover_args=grover_args, device=torch.device('cuda'))
data_path = "chebi20.csv"
batch_size = 8
dataloader_train, dataloader_val, dataloader_test = crossmol_dataset.get_dataloaders(data_path=data_path,batch_size=batch_size)

epochs = 20
lr = 1e-5
weight_decay = False
# crossmol, test_loss, test_output = crossmol.fit(dataloader_train, dataloader_val, dataloader_test, epochs, lr, weight_decay, verbose=True)
# now = datetime.now()
# current_time = now.strftime("%H:%M:%S")
# torch.save(crossmol.state_dict(), '{}_{:.6f}_no_ca.pth'.format(current_time, test_loss))

crossmol.load_state_dict(torch.load('epoch_15_val_loss_0.05_no_ca.pth'))
test_loss, test_output = crossmol.test(dataloader_test)
np.save('test_output_no_ca.npy', test_output)
