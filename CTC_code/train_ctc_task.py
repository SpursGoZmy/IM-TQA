from GNN_models import RGCN
import numpy as np
import paddle
import paddle.nn as nn
import pgl
import argparse
import logging
import time
import json
import os
import pickle
from sklearn.metrics import classification_report
import random
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="../data/",help="dir path of table data and init hetergraph data")
    parser.add_argument("--saved_graphs_dir",type=str,default="../data/",help="dir path to save processed graphs in ctc task, with last hidden state as node feat")
    parser.add_argument("--model_save_dir",type=str,default="./saved_models/ctc_gnn/",help="dir path to save trained model")
    parser.add_argument("--pred_save_dir",type=str,default="./pred_results/",help="dir path to save pred results of ctc task")
    parser.add_argument("--epoch_num",type=int,default=5,help="epoch num")
    parser.add_argument("--input_size",type=int,default=800,help="dim of init cell repr")
    parser.add_argument("--hidden_size",type=int,default=1024,help="hidden sise of GNN or FNN model")
    parser.add_argument("--layer_num",type=int,default=3,help="layer num of GNN model")
    parser.add_argument("--num_class",type=int,default=2,help="num_class in ctc_task")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning_rate of Adam optimizer")
    parser.add_argument("--random_seed",type=int,default=5678,help="random seed")
    parser.add_argument("--n_filters",type=int,default=32,help="n_filters of CNN_BERT baseline")
    parser.add_argument("--run_num",type=int,default=0,help="allocate a new run-ID to every new train program ")
    parser.add_argument("--model_name",type=str,default="GNN",help="model_name")
    args=parser.parse_args()
    return args

@paddle.no_grad()
def evaluate(args,num_class,test_graphs,test_tables,model):
    model.eval()
    criterion = paddle.nn.loss.CrossEntropyLoss()
    loss_list = []
    y_true = []
    y_pred = []
    y_true_hierarchical = []
    y_pred_hierarchical = []
    y_true_complex = []
    y_pred_complex = []
    y_true_standard = []
    y_pred_standard = []
    y_true_horizontal = []
    y_pred_horizontal = []
    new_test_graphs = {}
    test_table_ctc_pred_result = {}

    for table in test_tables:
        table_id = table['table_id']
        table_type = table['table_type']
        heter_g = test_graphs[table_id]
        heter_g.tensor()
        labels = create_label_list(num_class,table)     
        assert len(labels) == heter_g.num_nodes           
        label_tensor = paddle.to_tensor(labels)
        bert_feats = heter_g.node_feat['features']  #[cell_num,768]
        manual_feats = heter_g.node_feat['manual_features']  #[cell_num,32]
        node_feats = paddle.concat([bert_feats,manual_feats],axis=1)  #[cell_num,800]
        logits,resulted_node_feat=model(heter_g,node_feats)
        # get predicted results
        preds = paddle.argmax(logits,axis=1).detach().cpu().numpy().tolist()
        loss = criterion(logits, label_tensor)
        loss_list.append(loss.item())

        y_pred.extend(preds)
        y_true.extend(labels)
        if table_type == 'vertical':
            y_true_standard.extend(labels)
            y_pred_standard.extend(preds)
        if table_type == 'horizontal':
            y_true_horizontal.extend(labels)
            y_pred_horizontal.extend(preds)
        if table_type == 'hierarchical':
            y_true_hierarchical.extend(labels)
            y_pred_hierarchical.extend(preds)
        if table_type == 'complex':
            y_true_complex.extend(labels)
            y_pred_complex.extend(preds)
    
    assert len(y_true) == len(y_pred)
    avg_loss = np.mean(loss_list)
    #if num_class == 5:
    target_names = ['pure-data','column_attribute','row_attribute','column_index','row_index']
    possible_labels = [0,1,2,3,4]

    print("(1) report on all tables: ")
    # compute classification results based on sklearn.metrics
    report_results=classification_report(y_true, y_pred,labels=possible_labels, target_names=target_names,output_dict=True)
    print(classification_report(y_true, y_pred,labels=possible_labels, target_names=target_names))
    print("(2) report on vertical tables: ")
    print(classification_report(y_true_standard, y_pred_standard, labels=possible_labels,target_names=target_names))
    print("(3) report on horizontal tables: ")
    print(classification_report(y_true_horizontal, y_pred_horizontal, labels=possible_labels,target_names=target_names))
    print("(4) report on hierarchical tables:")
    print(classification_report(y_true_hierarchical, y_pred_hierarchical,labels=possible_labels, target_names=target_names))
    print("(5) report on complex tables: ")
    print(classification_report(y_true_complex, y_pred_complex,labels=possible_labels ,target_names=target_names))
    return report_results,avg_loss,target_names

def eval_on_tables(args,model,tables,graphs):
    model.eval()
    # pred results of ctc_task
    eval_results = {}
    # new graphs with updated node feats from RGCN
    new_graphs = {}
    for table in tables:
        table_id = table['table_id']
        heter_g = graphs[table_id]
        heter_g.tensor()
        bert_feats=heter_g.node_feat['features']  #[cell_num,768]
        manual_feats=heter_g.node_feat['manual_features']  #[cell_num,32]
        node_feats=paddle.concat([bert_feats,manual_feats],axis=1)  #[cell_num,800]
        #logits,resulted_node_feat=model(heter_g,heter_g.node_feat['features'])
        logits,resulted_node_feat=model(heter_g,node_feats)
        
        labels = create_label_list(args.num_class,table) 
        preds = paddle.argmax(logits,axis=1).detach().cpu().numpy().tolist()
        heter_g = heter_g.numpy()
        num_nodes = int(heter_g.num_nodes)
        edges = heter_g._edges_dict
        # building new hetergraph based on updated node feats from RGCN
        node_feats={}
        node_feats['final_ctc_features'] = resulted_node_feat.detach().cpu().numpy()
        node_types = [(i, 'table_node' ) for i in range(num_nodes)]
        new_g = pgl.HeterGraph(edges=edges,
                   node_types=node_types,
                   node_feat=node_feats)
        new_graphs[table_id] = new_g
        result = {}
        result['logits'] = logits.detach().cpu().numpy()
        result['preds'] = preds
        result['labels'] = labels
        eval_results[table_id] = result
    return eval_results, new_graphs

@paddle.no_grad()
def save_model_eval_results_of_ctc_task(args,model,train_tables,train_graphs,valid_tables,valid_graphs,test_tables,test_graphs):
    model.eval()
    graph_save_dir = args.saved_graphs_dir
    graph_save_dir = os.path.join(graph_save_dir,f'run_{args.run_num}_{args.num_class}_ctc_task_epoch_{args.epoch_num}_gnn_layer_num_{args.layer_num}_hidden_size_{args.hidden_size}_lr_{str(args.learning_rate)}')
    os.makedirs(graph_save_dir, exist_ok=True)
    resulted_train_graph_save_path = os.path.join(graph_save_dir,'resulted_train_graphs_of_ctc.pkl')
    resulted_valid_graph_save_path = os.path.join(graph_save_dir,'resulted_valid_graphs_of_ctc.pkl')
    resulted_test_graph_save_path = os.path.join(graph_save_dir,'resulted_test_graphs_of_ctc.pkl')
    
    pred_save_dir = os.path.join(args.pred_save_dir,f'run_{args.run_num}_{model.model_name}_{args.num_class}-class_ctc_pred_results_epoch_{args.epoch_num}_gnn_layer_num_{args.layer_num}_hidden_size_{args.hidden_size}_lr_{str(args.learning_rate)}')
    os.makedirs(pred_save_dir, exist_ok=True)
    train_table_ctc_pred_result_save_path = os.path.join(pred_save_dir,'ctc_train_pred_results.pkl')
    valid_table_ctc_pred_result_save_path = os.path.join(pred_save_dir,'ctc_valid_pred_results.pkl')
    test_table_ctc_pred_result_save_path = os.path.join(pred_save_dir,'ctc_test_pred_results.pkl')

    train_eval_results,new_train_graphs = eval_on_tables(args,model,train_tables,train_graphs)
    valid_eval_results,new_valid_graphs = eval_on_tables(args,model,valid_tables,valid_graphs)
    test_eval_results,new_test_graphs = eval_on_tables(args,model,test_tables,test_graphs)

    pickle.dump(train_eval_results,open(train_table_ctc_pred_result_save_path,"wb"))
    pickle.dump(valid_eval_results,open(valid_table_ctc_pred_result_save_path,"wb"))
    pickle.dump(test_eval_results,open(test_table_ctc_pred_result_save_path,"wb"))
    # if you don't need the updated graphs with new node feats from RGCN, you can annotate the following codes
    pickle.dump(new_train_graphs,open(resulted_train_graph_save_path,"wb"))
    pickle.dump(new_valid_graphs,open(resulted_valid_graph_save_path,"wb"))
    pickle.dump(new_test_graphs,open(resulted_test_graph_save_path,"wb"))


def create_label_list(num_class,table):
    # build labels of one table for cell type classification task
    num_nodes = len(table['chinese_cell_value_list'])
    if num_class == 5:
        labels = []
        for cell_id in range(num_nodes):
            if cell_id in table['column_attribute']:
                labels.append(1)
            elif cell_id in table['row_attribute']:
                labels.append(2)
            elif cell_id in table['column_index']:
                labels.append(3)
            elif cell_id in table['row_index']:
                labels.append(4)
            else:
                labels.append(0)

    return labels


def compute_Macro_F1(report_results,target_names):
    Macro_F1 = report_results['macro avg']['f1-score']
    header_class_f1_list = []
    for class_name in target_names[1:]:
        header_class_f1_list.append(report_results[class_name]['f1-score'])
    Macro_F1_of_header = np.mean(header_class_f1_list)
    return Macro_F1, Macro_F1_of_header

def train_ctc_task(args,model,train_graphs,train_tables,valid_graphs,valid_tables):
    os.makedirs(args.model_save_dir, exist_ok=True)
    model_save_dir = args.model_save_dir
    epoch_num = args.epoch_num
    hidden_size = args.hidden_size
    num_class = args.num_class
    layer_num = args.layer_num
    lr = args.learning_rate
    run_num = args.run_num
    ## set weights of cross-entropy based on cell nums
    col_header_no_data_num = 3836
    row_header_no_data_num = 4903
    col_header_and_data_num = 989
    row_header_and_data_num = 2350
    data_num = 17127
    total_cell_num = 29205
    w = np.array([data_num,col_header_no_data_num,row_header_no_data_num,col_header_and_data_num,row_header_and_data_num])
    w = 1-w/total_cell_num
    weight_tensor = paddle.to_tensor(w,dtype='float32')    # dtype must be float32!!
    print("train loss weight tensor: ", weight_tensor)
    criterion = paddle.nn.loss.CrossEntropyLoss(weight=weight_tensor)
    #criterion = paddle.nn.loss.CrossEntropyLoss()
    optim = paddle.optimizer.Adam(learning_rate=lr,
                            parameters=model.parameters())
    train_loss = []
    valid_loss = []
    best_Macro_F1 = 0
    best_Macro_F1_of_header = 0
    best_results_based_on_Macro_F1 = {'epoch_num':0,'results':None}
    best_results_based_on_Macro_F1_of_header = {'epoch_num':0,'results':None}
    best_model_state_dict = None
    flag = 1
    for epoch in range(epoch_num):
        epoch_loss = []
        model.train()
        random.shuffle(train_tables)
        for table in train_tables:
            table_id = table['table_id']
            heter_g = train_graphs[table_id]
            heter_g.tensor()
            labels = create_label_list(num_class,table)
            assert len(labels) == heter_g.num_nodes
            label_tensor = paddle.to_tensor(labels)
            
            bert_feats = heter_g.node_feat['features']  #[cell_num,768]
            manual_feats = heter_g.node_feat['manual_features']  #[cell_num,32]
            node_feats = paddle.concat([bert_feats,manual_feats],axis=1)  #[cell_num,800]
            logits,resulted_node_feat=model(heter_g,node_feats)
            #preds=paddle.argmax(logits,axis=1).detach().cpu().numpy().tolist()
            loss = criterion(logits, label_tensor)
            loss.backward()
            epoch_loss.append(loss.item())
            if np.isnan(loss.item()):
                print("table_id of problem tables:",table_id)
            optim.step()
            optim.clear_grad()
        #print(epoch_loss)
        print("epoch: %d | train loss: %.4f" % (epoch, np.mean(epoch_loss)))
        train_loss.append(np.mean(epoch_loss))
        report_results,avg_valid_loss,target_names = evaluate(args,num_class,valid_graphs,valid_tables,model)
        print("epoch: %d | valid loss: %.4f" % (epoch, avg_valid_loss))
        valid_loss.append(avg_valid_loss)
        Macro_F1,Macro_F1_of_header = compute_Macro_F1(report_results,target_names)
        
        if Macro_F1 > best_Macro_F1:
            best_Macro_F1 = Macro_F1
            best_results_based_on_Macro_F1['epoch_num'] = epoch
            best_results_based_on_Macro_F1['results'] = report_results
        if Macro_F1_of_header > best_Macro_F1_of_header:
            best_Macro_F1_of_header = Macro_F1_of_header
            best_results_based_on_Macro_F1_of_header['epoch_num'] = epoch
            best_results_based_on_Macro_F1_of_header['results'] = report_results
            # find best model state dict based on Macro F1 of header
            best_model_state_dict = deepcopy(model.state_dict())

    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir,f'run_{run_num}_{model.model_name}_{num_class}-class_epoch_{epoch_num}_gnn_layer_num_{layer_num}_hidden_size_{hidden_size}_lr_{str(lr)}.pdparams')
    paddle.save(best_model_state_dict, model_save_path)
    print("save model params of epoch %d"%(best_results_based_on_Macro_F1_of_header['epoch_num']))
    #save_resulted_graphs_after_ctc_task(args,new_train_graphs,new_test_graphs)
    print("successfully train a %s on %d-class ctc task!"%(model.model_name,num_class))
    print("train_loss: ")
    print(train_loss)
    print("valid_loss: ")
    print(valid_loss)
    print("best epoch based on Macro F1: %d , best Macro F1 on valid set: %f "%(best_results_based_on_Macro_F1['epoch_num'],best_Macro_F1))
    print("best epoch based on Macro F1 of header %d , best Macro F1 of Header on valid set: %f "%(best_results_based_on_Macro_F1_of_header['epoch_num'],best_Macro_F1_of_header))
    return best_model_state_dict
    

def load_RGCN_model(args,train_graphs):
    table_ids = list(train_graphs.keys())
    edge_types = train_graphs[table_ids[0]].edge_types
    in_dim = args.input_size #768+32=800
    hidden_size = args.hidden_size #800
    num_class = args.num_class
    layer_num = args.layer_num
    num_bases = 8

    model = RGCN(in_dim,hidden_size,num_class,layer_num,edge_types,num_bases)
    return model

def main():
    args = parse_args()
    print("used args:")
    print(args)
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
    np.random.seed(args.random_seed)
    paddle.seed(args.random_seed)
    start_time = time.time()
    #　load table data
    train_tables = json.load(open(os.path.join(args.data_dir,"train_tables.json")))
    valid_tables = json.load(open(os.path.join(args.data_dir,"valid_tables.json")))
    test_tables = json.load(open(os.path.join(args.data_dir,"test_tables.json")))
    # load constructed graph data
    train_graphs = pickle.load(open(os.path.join(args.saved_graphs_dir,"final_graphs_of_train_tables.pkl"),"rb"))
    valid_graphs = pickle.load(open(os.path.join(args.saved_graphs_dir,"final_graphs_of_valid_tables.pkl"),"rb"))
    test_graphs = pickle.load(open(os.path.join(args.saved_graphs_dir,"final_graphs_of_test_tables.pkl"),"rb"))
    # load RGCN model
    model = load_RGCN_model(args,train_graphs)
    # 
    print("start training of RGCN model for CTC task")
    best_model_state_dict = train_ctc_task(args,model,train_graphs,train_tables,valid_graphs,valid_tables)
    model.set_state_dict(best_model_state_dict)
    print("-"*30)
    print("test results: ")
    report_results,avg_test_loss,target_names = evaluate(args,args.num_class,test_graphs,test_tables,model)
    print("test_loss: ",avg_test_loss)
    print(f'train with {args.epoch_num} epoch took {(time.time()-start_time)/60} minutes')
    print("-"*30)
    print("save ctc pred results of tables")
    save_model_eval_results_of_ctc_task(args,model,train_tables,train_graphs,valid_tables,valid_graphs,test_tables,test_graphs)

    

if __name__ == '__main__':
    main() 

