import numpy as np
import paddle
import paddle.nn as nn
import pgl
import pickle
import argparse
import os 
import time
import json
import logging
from collections import defaultdict
from paddlenlp.transformers import BertModel,ErnieModel,ErnieTokenizer ,ErnieTinyTokenizer,BertTokenizer
from paddlenlp.data import Stack, Tuple, Pad

logger = logging.getLogger(__name__)

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tables_dir",type=str,default="../data",help="dir path of table data")
    parser.add_argument("--saved_graphs_dir",type=str,default="../data",help="dir path to save graphs")
    parser.add_argument("--init_embedding_model",type=str,default="bert-base-chinese",help="model used to create init node embedding")
    args=parser.parse_args()
    return args

def convert_table_to_heterogeneous_graph(layout,cell_values,model_for_init_node_feat,tokenizer):
    max_seq_len = 512
    num_nodes = len(cell_values)
    
    node_types = [(i, 'table_node' ) for i in range(num_nodes)]
    edges = {
        'top_to_down': [],
        'left_to_right': []
    }
    num_rows = len(layout)
    num_cols = len(layout[0])
    
    for i in range(num_rows):
        for j in range(num_cols):
            node_id=layout[i][j]   
            #right neighbour
            if j+1 < num_cols:
                right_neighbour = layout[i][j+1]
                if right_neighbour != node_id:
                    edges['left_to_right'].append( (node_id,right_neighbour) )
                else:
                    pass
            #down neighbour
            if i+1 < num_rows:
                down_neighbour = layout[i+1][j]
                if down_neighbour != node_id:
                    edges['top_to_down'].append( (node_id,down_neighbour) )
                else:
                    pass
    # using 'set' to process overlapped edges
    edges['top_to_down'] = list(set(edges['top_to_down']))
    edges['left_to_right'] = list(set(edges['left_to_right']))
    # add reversed edges
    down_to_top = [ (j, i) for i, j in edges['top_to_down']]
    right_to_left = [ (j,i) for i, j in edges['left_to_right'] ]
    edges['down_to_top'] = down_to_top
    edges['right_to_left'] = right_to_left
    # build init node feats
    # make batch using cell_values 
    model_for_init_node_feat.eval()
    batch_size = 8
    num_batches = num_nodes // batch_size
    if num_nodes % batch_size != 0:
            num_batches += 1
    tensor_list_of_all_batches=[]
    for index in range(num_batches):
        if index != num_batches-1:
            batch_cell_values = cell_values[index*batch_size:(index+1)*batch_size]
            
        else:
            batch_cell_values = cell_values[index*batch_size:]
            
        inputs = tokenizer(batch_cell_values)  # {'input_ids':[], 'token_type_ids':[]  }
        #pad inputs
        input_id_pad = Pad(axis=0, pad_val=tokenizer.pad_token_id)
        token_type_id_pad = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
        # limit the token num within 512
        inputs['input_ids'] = [ text_ids[:max_seq_len] for text_ids in inputs['input_ids'] ]
        inputs['token_type_ids'] = [ token_type_ids[:max_seq_len] for token_type_ids in inputs['token_type_ids']]
        
        input_ids_batch = input_id_pad(inputs['input_ids'])    #[batch,max_length]
        token_type_ids_batch = token_type_id_pad(inputs['token_type_ids']) #[batch,max_length]
        input_ids_batch = paddle.to_tensor(input_ids_batch)
        token_type_ids_batch = paddle.to_tensor(token_type_ids_batch)
        sequence_output,pooled_output = model_for_init_node_feat(input_ids_batch,token_type_ids_batch)  
        #sequence_output  : [batch,max_seq_length,hidden_size]  [8,11,768]
        #pooled_output   : [batch,hidden_size]   [8,768]
        tensor_list_of_all_batches.append(pooled_output)
    concat_node_feats = paddle.concat(tensor_list_of_all_batches)  #[num_nodes,hidden_size]
    # check whether embeding vector num equals to node num
    assert concat_node_feats.shape[0] == num_nodes
    node_features = {'features': concat_node_feats.detach().cpu().numpy()}
    # build pgl hetergraph object
    g = pgl.HeterGraph(edges=edges,
                   node_types=node_types,
                   node_feat=node_features)

    return g

def build_graphs_from_tables(tables_dir='../data/',saved_graphs_dir='../data/',init_embedding_model='bert-base-chinese'):
    #load tables from json file
    if os.path.exists(tables_dir):
        train_table_file = os.path.join(tables_dir,'train_tables.json')
        valid_table_file = os.path.join(tables_dir,'valid_tables.json')
        test_table_file = os.path.join(tables_dir,'test_tables.json')
    else:
        raise RuntimeError('tables_dir does not exist, please specify the correct dir path.')
    train_tables = json.load(open(train_table_file,"r"))
    valid_tables = json.load(open(valid_table_file,"r"))
    test_tables = json.load(open(test_table_file,"r"))
    
    logger.info("train tables num :%d"%(len(train_tables)))
    logger.info("valid tables num :%d"%(len(valid_tables)))
    logger.info("test tables num :%d"%(len(test_tables)))
    # load model for building init node embedding
    model_type = init_embedding_model
    logger.info("model type:%s"%(model_type.lower()))
    assert model_type in ["ernie-tiny","ernie","bert-base-chinese"]
    # you can pass a local path of the pre_trained model to model.from_pretrained()
    if model_type=="ernie-tiny":
        model=ErnieModel.from_pretrained('ernie-tiny')
        tokenizer=ErnieTinyTokenizer.from_pretrained('ernie-tiny')
    elif model_type=="ernie":
        model=ErnieModel.from_pretrained('ernie-1.0-base-zh')
        tokenizer=ErnieTokenizer.from_pretrained('ernie-1.0-base-zh')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese')
    #convert all tables to graphs,using model to create init node embedding
    logger.info(f"convert all tables to graphs, using {model_type.lower()} to create init node embedding.")
    split_name_to_constructed_graphs = defaultdict(dict)
    tables_of_differnet_splits = [train_tables,valid_tables,test_tables]
    split_name_list = ['train','valid','test']
    #graphs_of_train_tables = {}
    #graphs_of_valid_tables = {}
    #graphs_of_test_tables = {}
    for split_name, tables_of_one_split in zip(split_name_list,tables_of_differnet_splits):
        for table in tables_of_one_split:
            table_id = table['table_id']
            layout = table['cell_ID_matrix']
            cell_values = table['chinese_cell_value_list']
            g = convert_table_to_heterogeneous_graph(layout,cell_values,model,tokenizer)
            split_name_to_constructed_graphs[split_name][table_id] = g
            # check whether there are non features in node embedding
            if np.any(np.isnan(g.node_feat['features'])):
                print("nan features! train table_id: ",table_id)
    graphs_of_train_tables = split_name_to_constructed_graphs['train']
    graphs_of_valid_tables = split_name_to_constructed_graphs['valid']
    graphs_of_test_tables = split_name_to_constructed_graphs['test']
    logger.info("train graphs num: %d"%(len(graphs_of_train_tables)))
    logger.info("valid graphs num: %d"%(len(graphs_of_valid_tables)))
    logger.info("test graphs num: %d"%(len(graphs_of_test_tables)))

    os.makedirs(saved_graphs_dir, exist_ok=True)
    saved_path_of_train_graph = os.path.join(saved_graphs_dir,'graphs_of_train_tables.pkl')   
    saved_path_of_valid_graph = os.path.join(saved_graphs_dir,'graphs_of_valid_tables.pkl')  
    saved_path_of_test_graph = os.path.join(saved_graphs_dir,'graphs_of_test_tables.pkl')   
    pickle.dump(graphs_of_train_tables,open(saved_path_of_train_graph,"wb"))
    pickle.dump(graphs_of_valid_tables,open(saved_path_of_valid_graph,"wb"))
    pickle.dump(graphs_of_test_tables,open(saved_path_of_test_graph,"wb"))
    logger.info("successfully convert tables to Heterogenous Graphs!")
    logger.info("train_graphs saved to %s"%(saved_path_of_train_graph))
    logger.info("valid_graphs saved to %s"%(saved_path_of_valid_graph))
    logger.info("test_graphs saved to %s"%(saved_path_of_test_graph))

def main():
    args=parse_args()
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
    start_time = time.time()
    build_graphs_from_tables(args.tables_dir,args.saved_graphs_dir,args.init_embedding_model)
    logger.info(f'convert all tables took {(time.time()-start_time)/60} minutes')

if __name__ == '__main__':
    main() 




