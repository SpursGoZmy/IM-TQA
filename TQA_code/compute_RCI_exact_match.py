from util.line_corpus import write_open, jsonl_lines
import ujson as json
import numpy as np
from collections import defaultdict
import pickle

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

def gather_predictions(input_file,intent=0):
    # for the given question, find out row ids or col ids in which the trained RCI model predicts that answer cells exist  
    qid_to_related_idx = defaultdict(list)
    for line in jsonl_lines(input_file):
        jobj = json.loads(line)
        pred = jobj['predictions']
        qid, ndx_str = jobj['id'].split(':')
        # if pred[1]>pred[0], the trained RCI model think answer cells exist in thie row or col.
        # set intent = 1 if the pre-defined row id or col id starts from 1 instead of 0
        if pred[1]>pred[0]:
            qid_to_related_idx[qid].append(int(ndx_str)+intent)
    return qid_to_related_idx

def get_answer_cells(col_ids,row_ids,layout):
    # get the answer cell ids based on target row ids and target col ids, e.g., row_id = 3, col_id = 2, the answer cell locates at (3,2) in the table cell matrix.
    if len(col_ids) == 0 or len(row_ids) == 0:
        return set([])
    answer_set = set()
    
    for i in row_ids:
        for j in col_ids:
            answer_set.add(layout[i][j])
    return answer_set

# load pred results
col_prediction_file = "./datasets/IM_TQA/apply_bert/col_bert/"
row_prediction_file = "./datasets/IM_TQA/apply_bert/row_bert/"
# file of final predictions
pred_test_pred_results_save_path=open('./datasets/IM_TQA/RGCN-RCI_test_pred_results.pkl','wb')
qid_to_related_row_ids = gather_predictions(row_prediction_file)
qid_to_related_col_ids = gather_predictions(col_prediction_file)
print("len(qid_to_related_row_ids):", len(qid_to_related_row_ids))
print("len(qid_to_related_col_ids):", len(qid_to_related_col_ids))
# load ground truth
test_tables = json.load(open('../data/test_tables.json'))
test_questions = json.load(open('../data/test_questions.json'))

# find valid question ids where both positive row_ids and positive col_ids exist, otherwise the model is thought to be failed to answer this question  
existed_q_ids = set(qid_to_related_row_ids.keys()).intersection(set(qid_to_related_col_ids.keys()))

table_id_to_test_tables={}
for table in test_tables:
    table_id = table['table_id']
    table_id_to_test_tables[table_id] = table
q_id_to_test_questions={}
for item in test_questions:
    q_id = item['question_id']
    q_id_to_test_questions[q_id] = item

total_question_num = len(q_id_to_test_questions) # 
total_exact_match = 0
question_num_by_table_types = defaultdict(int)
exact_match_num_by_table_types = defaultdict(int)

test_pred_results = []
# compute exact match
for q_id in q_id_to_test_questions:
    item={}
    question = q_id_to_test_questions[q_id]
    item.update(question)
    gold_answer_cell_list = question['answer_cell_list']
    table_id = question['table_id']
    #question_text = question['chinese_question']
    table = table_id_to_test_tables[table_id]
    #file_name = table['file_name']
    
    layout = table['cell_ID_matrix']
    table_type = table['table_type']
    # For a question, if either positive row_ids or positive col_ids do not exist, i.e., we cannot find predicted answer cells,
    # then the model is thought to be failed to answer this question  
    try:
        related_col_ids = qid_to_related_col_ids[q_id]
        related_row_ids = qid_to_related_row_ids[q_id]
        pred_answer_set = get_answer_cells(related_col_ids,related_row_ids,layout)
    except:
        pred_answer_set= set()
    item['pred_answer_list'] = list(pred_answer_set)
    
    if pred_answer_set == set(gold_answer_cell_list):
        is_correct = 1
    else:
        is_correct = 0
    total_exact_match += is_correct
    question_num_by_table_types[table_type] += 1
    exact_match_num_by_table_types[table_type] += is_correct
    item['is_correct'] = is_correct
    test_pred_results.append(item)
# save pred results    
pickle.dump(test_pred_results,pred_test_pred_results_save_path)
# output overall exact match results
print("(1) report on all tables: ")
print("total exact match score: ",total_exact_match/total_question_num)
print("correct question num: ",total_exact_match)
print("total question num:",total_question_num)
print("-"*20)
# output exact match results on tables of each types
index = 2
for table_type, question_num in question_num_by_table_types.items():
    print(f"({index}) report on {table_type} tables: ")
    print(f"exact match score on {table_type} tables:", exact_match_num_by_table_types[table_type]/question_num_by_table_types[table_type])
    print(f"correct question num on {table_type} tables:",exact_match_num_by_table_types[table_type])
    print(f"total question num on {table_type} tables:",question_num_by_table_types[table_type])
    index += 1
    print("-"*20)
