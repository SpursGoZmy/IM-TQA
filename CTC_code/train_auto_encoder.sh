CUDA_VISIBLE_DEVICES=0 nohup python train_auto_encoder.py \
--run_num=1 \
--enc_hidden_dim=32 \
--manual_feat_dim=24 \
--random_seed=12345 \
--data_dir='../data/' \ 
--feats_save_dir='../data/' \ 
--model_save_dir='./saved_models/ctc_auto_encoder/' > ./log_files/train_auto_encoder_to_encode_manual_cell_feats.log &