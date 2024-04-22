import pickle
import pgl
# change path to corresponding data path
train_graphs = pickle.load(open('../data/graphs_of_train_tables.pkl','rb'))
valid_graphs = pickle.load(open('../data/graphs_of_valid_tables.pkl','rb'))
test_graphs = pickle.load(open('../data/graphs_of_test_tables.pkl','rb'))
manual_feats_of_train_tables = pickle.load(open('../data/train_tables_manual_cell_feats_32.pkl','rb'))
manual_feats_of_valid_tables = pickle.load(open('../data/valid_tables_manual_cell_feats_32.pkl','rb'))
manual_feats_of_test_tables = pickle.load(open('../data/test_tables_manual_cell_feats_32.pkl','rb'))
# check num of manual feats and num of existing hetergraph
assert len(train_graphs) == len(manual_feats_of_train_tables)
assert len(valid_graphs) == len(manual_feats_of_valid_tables)
assert len(test_graphs) == len(manual_feats_of_test_tables)

for table_id in train_graphs:
    heter_g = train_graphs[table_id]
    manual_feats = manual_feats_of_train_tables[table_id]
    assert manual_feats.shape[0] == heter_g.node_feat['features'].shape[0]
    heter_g.node_feat['manual_features'] = manual_feats

for table_id in valid_graphs:
    heter_g = valid_graphs[table_id]
    manual_feats = manual_feats_of_valid_tables[table_id]
    assert manual_feats.shape[0] == heter_g.node_feat['features'].shape[0]
    heter_g.node_feat['manual_features'] = manual_feats

for table_id in test_graphs:
    heter_g = test_graphs[table_id]
    manual_feats = manual_feats_of_test_tables[table_id]
    assert manual_feats.shape[0] == heter_g.node_feat['features'].shape[0]
    heter_g.node_feat['manual_features'] = manual_feats
# save resulting graphs
pickle.dump(train_graphs,open('../data/final_graphs_of_train_tables.pkl','wb'))
pickle.dump(valid_graphs,open('../data/final_graphs_of_valid_tables.pkl','wb'))
pickle.dump(test_graphs,open('../data/final_graphs_of_test_tables.pkl','wb'))
print("adding manual feats to hetergraph has been done!!")


