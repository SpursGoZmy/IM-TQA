from GNN_models import Manual_CTC_Feats, AutoEncoder
import numpy as np
import argparse
import logging
import time
import os
import pickle
import json
import paddle
import paddle.nn as nn
from paddle.io import Dataset
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class Manual_Cell_Feats_Dataset(Dataset):
    def __init__(self,train_tables,Manual_CTC_Feats_Module):
        super(Manual_Cell_Feats_Dataset,self).__init__()
        self.train_tables = train_tables
        self.cell_feats_module = Manual_CTC_Feats_Module
        self.train_cell_feats_list = []
        for table in train_tables:
            layout = table['cell_ID_matrix']
            cell_values = table['chinese_cell_value_list']
        
            cell_feats = Manual_CTC_Feats_Module.build_features(layout,cell_values)  # np.array, [cell_num,24]
            self.train_cell_feats_list.append(cell_feats)
        self.final_train_cell_feats = np.concatenate(self.train_cell_feats_list,axis=0)  # [all_cell_num,24]
        print("final train cell feats.shape: ",self.final_train_cell_feats.shape)
    def __getitem__(self, idx):
        one_cell_feats = self.final_train_cell_feats[idx] #[24,]
        data = paddle.to_tensor(one_cell_feats, dtype='float32')
        label = paddle.to_tensor(one_cell_feats, dtype='float32')
        
        return data,label
    
    def __len__(self):
        return self.final_train_cell_feats.shape[0]

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="../data/",help="dir path of table data")
    parser.add_argument("--feats_save_dir",type=str,default="../data/",help="dir path to save encoded manual cell feats")
    parser.add_argument("--model_save_dir",type=str,default="./saved_models/ctc_autoencoder/",help="dir path to save trained autoencoder")
    parser.add_argument("--enc_hidden_dim",type=int,default=32,help="the encoder output dim in auto encoder, used to represent manual cell feats")
    parser.add_argument("--manual_feat_dim",type=int,default=24,help="manual feat dim")
    parser.add_argument("--random_seed",type=int,default=5678,help="random seed")
    parser.add_argument("--run_num",type=int,default=0,help="allocate a new run-ID to every new run")
    args=parser.parse_args()
    return args

@paddle.no_grad()
def save_encoded_cell_feats(args,model,Manual_CTC_Feats_Module,train_tables,valid_tables,test_tables):
    model.eval()
    os.makedirs(args.feats_save_dir, exist_ok=True)
    feats_save_dir = args.feats_save_dir
    train_table_cell_feats_save_path = os.path.join(feats_save_dir,'train_tables_manual_cell_feats_%d.pkl'%(args.enc_hidden_dim))
    valid_table_cell_feats_save_path = os.path.join(feats_save_dir,'valid_tables_manual_cell_feats_%d.pkl'%(args.enc_hidden_dim))
    test_table_cell_feats_save_path = os.path.join(feats_save_dir,'test_tables_manual_cell_feats_%d.pkl'%(args.enc_hidden_dim))
    split_name_to_manual_cell_feats = defaultdict(dict)
    tables_of_differnet_splits = [train_tables,valid_tables,test_tables]
    split_name_list = ['train','valid','test']
    for split_name, tables_of_one_split in zip(split_name_list,tables_of_differnet_splits):
        for table in tables_of_one_split:
            table_id = table['table_id']
            layout = table['cell_ID_matrix']
            cell_values = table['chinese_cell_value_list']
            cell_feats = Manual_CTC_Feats_Module.build_features(layout,cell_values)  # np.array, [cell_num,24]
            cell_feats = paddle.to_tensor(cell_feats,dtype='float32')  #[cell_num, 24]
            dec,enc = model(cell_feats)
            split_name_to_manual_cell_feats[split_name][table_id] = enc.detach().cpu().numpy()
    
    encoded_cell_feats_of_train_tables = split_name_to_manual_cell_feats['train']
    encoded_cell_feats_of_valid_tables = split_name_to_manual_cell_feats['valid']
    encoded_cell_feats_of_test_tables = split_name_to_manual_cell_feats['test']
    assert len(encoded_cell_feats_of_train_tables)==len(train_tables)
    assert len(encoded_cell_feats_of_valid_tables)==len(valid_tables)
    assert len(encoded_cell_feats_of_test_tables)==len(test_tables)
    pickle.dump(encoded_cell_feats_of_train_tables,open(train_table_cell_feats_save_path,"wb"))
    pickle.dump(encoded_cell_feats_of_valid_tables,open(valid_table_cell_feats_save_path,"wb"))
    pickle.dump(encoded_cell_feats_of_test_tables,open(test_table_cell_feats_save_path,"wb"))
    

    

def train_auto_encoder(args,model,Manual_CTC_Feats_Module,train_tables):
    # train an auto encoder to convert 24-dim discrete manual feats to 32-dim continuous feats
    run_num = args.run_num
    model.train()
    Train_Dataset = Manual_Cell_Feats_Dataset(train_tables,Manual_CTC_Feats_Module)
    batch_size = 200
    epoch_num = 100
    learning_rate = 0.0005
    train_loader = paddle.io.DataLoader(Train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("batch size: ",batch_size)
    print("epoch num: ",epoch_num)
    print("learning rate: ",learning_rate)
    opt = paddle.optimizer.Adam(learning_rate=learning_rate,parameters=model.parameters())
    mse_loss = paddle.nn.MSELoss()
    history_loss = []
    flag = 1
    for epoch in range(epoch_num): 
        for batch_id,data in enumerate(train_loader()):
            input_feats = data[0]
            output_label_feats = data[1]
            
            dec,enc = model(input_feats)
            if flag == 1:
                print("input_feats.shape: ", input_feats.shape)  
                print("enc cell feats.shape: ", enc.shape)
                flag = 0
            avg_loss = mse_loss(dec,output_label_feats)  # reconstruction loss
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
        history_loss.append(avg_loss.numpy()[0])
    os.makedirs(args.model_save_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_save_dir,f'run_{run_num}_{model.model_name}_enc_hidden_size_{args.enc_hidden_dim}_lr_{learning_rate}_epoch_{epoch_num}.pdparams')
    paddle.save(model.state_dict(),model_save_path)
    print("save auto encoder params to %s"%(model_save_path))
    print("train loss: ",history_loss)

    return model

    
def main():
    args=parse_args()
    print("used args:")
    print(args)
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
    np.random.seed(args.random_seed)
    paddle.seed(args.random_seed)
    start_time = time.time()
    # load table data from json file
    train_tables = json.load(open(os.path.join(args.data_dir,"train_tables.json")))
    valid_tables = json.load(open(os.path.join(args.data_dir,"valid_tables.json")))
    test_tables = json.load(open(os.path.join(args.data_dir,"test_tables.json")))
    # load module for constructing manual cell features
    Manual_CTC_Feats_Module = Manual_CTC_Feats()
    # load auto encoder model
    Auto_Encoder = AutoEncoder(args.enc_hidden_dim,args.manual_feat_dim)
    # train an auto encoder model based on tables from the train split
    trained_model = train_auto_encoder(args,Auto_Encoder,Manual_CTC_Feats_Module,train_tables)
    # save encoded cell features of train, valid and test tables
    save_encoded_cell_feats(args,trained_model,Manual_CTC_Feats_Module,train_tables,valid_tables,test_tables)
    print(f'train AutoEncoder took {(time.time()-start_time)/60} minutes')

if __name__ == '__main__':
    main() 

