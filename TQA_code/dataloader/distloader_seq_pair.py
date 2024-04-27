import logging
import numpy as np
import ujson as json
from util.line_corpus import jsonl_lines
from typing import List
import random
import paddle
from paddlenlp.data import Stack, Tuple, Pad

logger = logging.getLogger(__name__)


def standard_json_mapper(jobj):
    if 'text_b' in jobj and 'question_type' in jobj:
        return jobj['id'], jobj['text_a'], jobj['text_b'], jobj['label'], jobj['question_type']
    elif 'text_b' in jobj and not 'question_type' in jobj:
        return jobj['id'], jobj['text_a'], jobj['text_b'], jobj['label']
    else:
        return jobj['id'], jobj['text'], jobj['label']


class SeqPairInst:
    __slots__ = 'inst_id', 'input_ids','token_type_ids' ,'label','question_type','schema'

    def __init__(self, inst_id, input_ids, token_type_ids, label,question_type=None,schema=None):
        self.inst_id = inst_id
        self.input_ids = input_ids  # list
        self.token_type_ids = token_type_ids  # list
        self.label = label
        self.question_type= question_type
        self.schema= schema



class SeqPairDataloader():
    def __init__(self, hypers, per_gpu_batch_size, tokenizer, data_dir,
                 json_mapper=standard_json_mapper, uneven_batches=False):
        super().__init__()
        self.hypers=hypers
        self.per_gpu_batch_size = per_gpu_batch_size
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.cls_id=tokenizer.cls_token_id # [CLS] 3
        self.sep_id=tokenizer.sep_token_id # [SEP] 5
        self.pad_id=tokenizer.pad_token_id # [PAD]  0

        self.json_mapper = json_mapper
        self.uneven_batches = uneven_batches
        
        self.insts = self.load_data()    #
        self.batch_size = self.per_gpu_batch_size * self.hypers.n_gpu  # 16*1
        self.num_batches = len(self.insts) // self.batch_size
        #self.random=random.Random(123)
        #if self.random is not None:
        #    self.random.shuffle(self.insts)

        if self.uneven_batches or self.hypers.world_size == 1:
            if len(self.insts) % self.batch_size != 0:
                self.num_batches += 1
        logger.info(f'insts size={len(self.insts)}, batch size = {self.batch_size}, batch count = {self.num_batches}')

        self.displayer = self.display_batch

        # just load the entire teacher predictions
    def set_random_to_shuffle_data(self,random_seed):
        self.random = random.Random(random_seed)
        self.random.shuffle(self.insts)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        if index >= self.num_batches:
            raise IndexError
        if self.hypers.world_size == 1:
            batch_insts = self.insts[index::self.num_batches]
        else:
            batch_insts = self.insts[index * self.batch_size:(index + 1) * self.batch_size]
        batch_tensors = self.make_batch(batch_insts)

        #if index == 0 and self.displayer is not None:
        #    self.displayer(batch_tensors)
        return batch_tensors

    def make_batch(self, insts: List[SeqPairInst]):
        batch_size = len(insts)
        
        input_id_pad = Pad(axis=0, pad_val=self.tokenizer.pad_token_id)
        token_type_id_pad = Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id)
        label_stack = Stack(dtype="int64")

        ids = [inst.inst_id for inst in insts]
        input_ids_batch = input_id_pad([ inst.input_ids  for inst in insts])    #[batch,max_length]
        token_type_ids_batch = token_type_id_pad([inst.token_type_ids  for inst in insts]) #[batch,max_length]
        labels = label_stack( [1 if inst.label else 0 for inst in insts])  #(batch,)
        
        input_ids_batch = paddle.to_tensor(input_ids_batch)
        token_type_ids_batch = paddle.to_tensor(token_type_ids_batch)
        labels = paddle.to_tensor(labels)
        tensors = ids,input_ids_batch,token_type_ids_batch,labels
        return tensors

    def display_batch(self, batch):

        ids = batch[0]
        input_ids, token_types, labels = [t.cpu().numpy() for t in batch[1:4]]
        print("in dataloader() input_ids.shape:")
        print(input_ids.shape)
        
        for i in range(1):
            toks = [str for str in self.tokenizer.convert_ids_to_tokens(input_ids[i])]
            logger.info(f"id-:{ids[i]}")
            logger.info(f"tokens:{toks}")
            logger.info(f"token_type_ids:{token_types[i]}")
            logger.info(f"labels:{labels[i]}")
        

    def load_data(self):
        logger.info('loading data from %s'%(self.data_dir))
        lines = jsonl_lines(self.data_dir)
        insts = []
        
        for line in lines:
            jobj = json.loads(line)
            # CONSIDER: do multiprocessing?
            question_type=None
            schema = None
            one_item = self.json_mapper(jobj)
            if len(one_item) == 5:
                inst_id, text_a, text_b, label, question_type = one_item
                schema=None
            if len(one_item) == 4:
                inst_id, text_a, text_b, label = one_item
                question_type=None
                schema=None

             
            
            multi_seg_input = self.tokenizer(text=text_a,text_pair=text_b,max_seq_len=self.hypers.max_seq_length)
            
            sp_inst = SeqPairInst(inst_id, multi_seg_input['input_ids'],multi_seg_input['token_type_ids'] ,label, question_type,schema)
            insts.append(sp_inst)
            
        return insts

