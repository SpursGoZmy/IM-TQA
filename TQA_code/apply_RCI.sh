CUDA_VISIBLE_DEVICES=0 nohup python apply_RCI_model.py \
--model_type bert-base-chinese \
--model_name_or_path ./datasets/IM_TQA/bert-base-chinese-epoch3-warmup0.1/col_bert_base \
--do_lower_case \
--input_dir ./datasets/IM_TQA/test_cols.jsonl.gz \
--max_seq_length 512 \
--output_dir ./datasets/IM_TQA/apply_bert/col_bert > ./log_files/IM_TQA/apply_rci_col_bert.log &

CUDA_VISIBLE_DEVICES=1 nohup python apply_RCI_model.py \
--model_type bert-base-chinese \
--model_name_or_path ./datasets/IM_TQA/bert-base-chinese-epoch3-warmup0.1/row_bert_base \
--do_lower_case \
--input_dir ./datasets/IM_TQA/test_rows.jsonl.gz \
--max_seq_length 512 \
--output_dir ./datasets/IM_TQA/apply_bert/row_bert > ./log_files/IM_TQA/apply_rci_row_bert.log &