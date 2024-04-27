import logging
import numpy as np
import ujson as json
from util.line_corpus import jsonl_lines
from typing import List
import random
import paddle
from paddlenlp.data import Stack, Pad
logger = logging.getLogger(__name__)


def standard_json_mapper(jobj):
    if 'text_b' in jobj and 'question_type' in jobj:
        return jobj['id'], jobj['text_a'], jobj['text_b'], jobj['label'], jobj['question_type'],jobj['schema']
    elif 'text_b' in jobj and not 'question_type' in jobj:
        return jobj['id'], jobj['text_a'], jobj['text_b'], jobj['label']
    else:
        return jobj['id'], jobj['text'], jobj['label']


class NewSeqPairInst:
    __slots__ = 'inst_id', 'text_a','text_b','text_a_ids','text_b_ids' ,'label'

    def __init__(self, inst_id, text_a, text_b, text_a_ids,text_b_ids,label):
        self.inst_id = inst_id
        self.text_a = text_a  # list
        self.text_b = text_b  # list
        self.label = label
        self.text_a_ids= text_a_ids
        self.text_b_ids= text_b_ids


class NewSeqPairDataloader:
    def __init__(self, tokenizer, data_path,batch_size=16,max_seq_len=512,
                 json_mapper=standard_json_mapper):
        super().__init__()

        self.data_dir = data_path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.max_seq_len = max_seq_len
        self.unk_token = self.vocab.unk_token #  '[UNK]'
        self.pad_token = self.vocab.pad_token  # '[PAD]'
        self.unk_id = self.vocab[self.unk_token]  #635963
        self.pad_id = self.vocab[self.pad_token]   #635964
       
        self.json_mapper = json_mapper
        #self.uneven_batches=uneven_batches
        
        self.insts = self.load_data()    #
        self.insts_num = len(self.insts)
        
        self.num_batches = len(self.insts) // self.batch_size
        if len(self.insts) % self.batch_size != 0:
                self.num_batches += 1
        logger.info(f'insts size={len(self.insts)}, batch size = {self.batch_size}, batch count = {self.num_batches}')


    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        if index >= self.num_batches:
            raise IndexError
        
        batch_insts = self.insts[index * self.batch_size:(index + 1) * self.batch_size]
        batch_tensors = self.make_batch(batch_insts)
        return batch_tensors

    def compute_attn_mask(self,input_ids_batch):
        input_ids_unsqueeze = paddle.to_tensor(input_ids_batch).unsqueeze(-1)   # [batch,seq_len,1]
        input_ids_unsqueeze = paddle.cast(input_ids_unsqueeze!=self.pad_id,'float32') # [batch,seq_len,1]
        attn_mask = paddle.matmul(input_ids_unsqueeze,input_ids_unsqueeze.transpose([0,2,1]))  #[ batch,seq_len,seq_len]
        attn_mask = attn_mask == 1
        attn_mask = attn_mask.unsqueeze(1)  # [batch,1,seq_len,seq_len]
        return attn_mask
        
    def make_batch(self, insts: List[NewSeqPairInst]):
        batch_size = len(insts)
        
        input_id_pad = Pad(axis=0, pad_val=self.pad_id)
        #token_type_id_pad=Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id)
        label_stack = Stack(dtype="int64")

        ids = [ inst.inst_id for inst in insts ]

        text_a_ids_batch = input_id_pad([inst.text_a_ids  for inst in insts])    #[batch,max_length],numpy.ndarray
        text_b_ids_batch = input_id_pad([inst.text_b_ids  for inst in insts]) 
        #token_type_ids_batch=token_type_id_pad([ inst.token_type_ids  for inst in insts  ]) #[batch,max_length]
        labels = label_stack([ inst.label  for inst in insts])  #(batch,)  # numpy.ndarray ([1,0,0,1]
        
        
        text_a_attn_mask = self.compute_attn_mask(text_a_ids_batch)  # bool dtype
        text_b_attn_mask = self.compute_attn_mask(text_b_ids_batch)


        text_a_ids_batch = paddle.to_tensor(text_a_ids_batch)
        text_b_ids_batch = paddle.to_tensor(text_b_ids_batch)
        labels = paddle.to_tensor(labels)
        tensors = ids,text_a_ids_batch,text_b_ids_batch,text_a_attn_mask,text_b_attn_mask,labels 
        return tensors

    def load_data(self):
        logger.info('loading data from %s'%(self.data_dir))
        lines = jsonl_lines(self.data_dir)
        insts = []
        for line in lines:
            jobj = json.loads(line)
            
            inst_id, text_a, text_b, label = self.json_mapper(jobj)
            text_a_ids=self.tokenizer.encode(text_a)[:self.max_seq_len]  #  [1,2,5,67]
            text_b_ids=self.tokenizer.encode(text_b)[:self.max_seq_len]  #  [23,45,234,24]

            #multi_seg_input=self.tokenizer(text=text_a,text_pair=text_b,max_seq_len=self.hypers.max_seq_length)
            sp_inst  = NewSeqPairInst(inst_id, text_a,text_b ,text_a_ids, text_b_ids,label)
            insts.append(sp_inst)
            
        return insts

