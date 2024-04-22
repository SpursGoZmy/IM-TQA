import numpy as np
import paddle
import paddle.nn as nn
import pgl
from sklearn.ensemble import RandomForestClassifier
from paddlenlp import Taskflow
from collections import defaultdict

class RGCN(nn.Layer):
    """Implementation of R-GCN model.
    """

    def __init__(self, input_size, hidden_size, num_class,
                 num_layers, etypes, num_bases):
        super(RGCN, self).__init__()
        self.model_name="RGCN_CTC_model"
        #self.num_nodes = num_nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.etypes = etypes
        self.num_bases = num_bases   #number of basis decomposition
        self.relu=paddle.nn.ReLU()
        #self.nfeat = self.create_parameter([self.num_nodes, self.input_size])

        self.rgcns = nn.LayerList()
        self.self_loop_linears=nn.LayerList()  #new add 
        out_dim = self.hidden_size
        for i in range(self.num_layers):
            in_dim = self.input_size if i == 0 else self.hidden_size
            self.self_loop_linears.append(nn.Linear(in_dim,out_dim))  # new add
            self.rgcns.append(
                pgl.nn.RGCNConv(
                    in_dim,
                    out_dim,
                    self.etypes,
                    self.num_bases, ))

        self.linear = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, g, feat):
        h = feat
        for i in range(self.num_layers):
            self_node_feat=self.self_loop_linears[i](h)
            neighbour_node_feat = self.rgcns[i](g, h)
            h=self.relu(self_node_feat+neighbour_node_feat)

        logits = self.linear(h)
        return logits,h

class Manual_CTC_Feats(object):
    def __init__(self) -> None:
        super(Manual_CTC_Feats,self).__init__()
        """
        This class is used to create manual cell feats from the paper "A Machine Learning Approach for Layout Inference in Spreadsheets"
        We adopt 24 feats from the paper's selected feats(section 5.1), 
        note that we do not use some feats like cell style and font because we want to process general tables rather than particular spreadsheet tables,
        and we adapt some feats for chinese language like punctuations
        """
        self.module_name='Manual-CTC-Feats-Module'
        # word_segmentation and pos_tagging tools from paddlenlp Taskflow
        self.seg_model = Taskflow("word_segmentation",batch_size=4)
        self.pos_model=Taskflow("pos_tagging",batch_size=4)
        self.punctuation_list=['，','。','-','——','？','?','!','！','.','“','”','；']
        self.special_token_list=['#','$','￥','%','&','*','~','/','<','>']
        #we select some chinese words expressing the 'total' meaning in the OpenHowNet toolkit.
        self.words_like_total=['共计','合计','总数','总量','全部','总共有','总额度','总额','总值',
                                '总和','共有','总共','累计','累计为','总合','合达','total','sum','all']
        self.content_feature_num=15
        self.spatial_feature_num=9


    def build_content_features(self,cell_values):
        cell_num=len(cell_values)
        content_features=np.zeros((cell_num,self.content_feature_num))  # dtype = 'float64'

        processed_cell_values=["Empty_cell" if cell_text=="" or str.isspace(cell_text) else cell_text for cell_text in cell_values]  ##处理空cell_values
        assert len(processed_cell_values)==len(cell_values)
        #segmentation results of cell_values (a nested list)
        segmented_cell_values=self.seg_model(processed_cell_values)    
        # pos tagging results of cell_values (a nested list)
        cell_tag_results=self.pos_model(processed_cell_values)   
        if len(segmented_cell_values)!=len(cell_values):
            print("len(segmented_cell_values): ",len(segmented_cell_values))
            print("len(cell_values): ",len(cell_values))
            print(cell_values)
            print(segmented_cell_values)
        #considered content feat: 
        for cell_id,cell_text in enumerate(cell_values):
            # LENGTH# 
            text_len=len(cell_text)
            if text_len==0:
                continue
            segmented_cell_text=segmented_cell_values[cell_id]
            pos_tag_results=cell_tag_results[cell_id]
            # NUM OF TOKENS#
            if cell_text=="":
                tokens_num=0
            else:
                tokens_num=len(segmented_cell_text)
            # LEADING SPACES#
            leading_spaces_num=0
            for char in cell_text:
                if char==' ':
                    leading_spaces_num+=1
                else:
                    break
            # IS NUMERIC?
            if str.isnumeric(cell_text):
                is_numeric=1
            else:
                is_numeric=0
            #STARTS WITH NUMBER?
            if str.isnumeric(cell_text[0]):
                starts_with_number=1
            else:
                starts_with_number=0
            # STARTS WITH SPECIAL?
            if cell_text[0] in self.special_token_list:
                starts_with_special=1
            else:
                starts_with_special=0
            #IS CAPITALIZED?
            if str.istitle(cell_text):
                is_capitalized=1
            else:
                is_capitalized=0
            #IS UPPER CASE?
            if str.isupper(cell_text):
                is_upper=1
            else:
                is_upper=0
            #IS ALPHABETIC?
            if str.isalpha(cell_text):
                is_alpha=1
            else:
                is_alpha=0
            #CONTAINS SPECIAL CHARS?
            contain_special_char=0
            for char in cell_text:
                if char in self.special_token_list:
                    contain_special_char=1
                    break
            #CONTAINS PUNCTUATIONS?
            contain_punctuation=0
            for char in cell_text:
                if char in self.punctuation_list:
                    contain_punctuation=1
                    break
            #CONTAINS COLON?
            if cell_text.find(':')>=0:
                contain_colon=1
            elif cell_text.find('：')>=0:
                contain_colon=1
            else:
                contain_colon=0
            #WORDS LIKE TOTAL?
            words_like_total=0
            for word in segmented_cell_text:
                if word in self.words_like_total:
                    words_like_total=1
                    break
            #WORDS LIKE TABLE?
            words_like_table=0
            for word in segmented_cell_text:
                if word in ['表','表格']:
                    words_like_table=1
                    break
            #IN YEAR RANGE?
            in_year_range=0
            for tag_result in pos_tag_results:
                if tag_result[1] in ['t','TIME']:
                    in_year_range=1
                    break
            #build feat vectors
            feat_vector=[]
            feat_vector.append(text_len)
            feat_vector.append(tokens_num)
            feat_vector.append(leading_spaces_num)
            feat_vector.append(is_numeric)
            feat_vector.append(starts_with_number)
            feat_vector.append(starts_with_special)
            feat_vector.append(is_capitalized)
            feat_vector.append(is_upper)
            feat_vector.append(is_alpha)
            feat_vector.append(contain_special_char)
            feat_vector.append(contain_punctuation)
            feat_vector.append(contain_colon)
            feat_vector.append(words_like_total)
            feat_vector.append(words_like_table)
            feat_vector.append(in_year_range)
            feat_vector=np.array(feat_vector)
            content_features[cell_id]=feat_vector
        return content_features
    
    def build_spatial_feats(self,layout,cell_values):
        cell_num=len(cell_values)
        spatial_features=np.zeros((cell_num,self.spatial_feature_num))  # dtype = 'float64'
        # considered spatial feat: row_id, col_id,has_1_2_3_4_neighbour,more than 5 neighbor,  is_merged_cell,num_of_cells,
        cell_id_to_row_and_col_id={}
        cell_id_to_neighbor_set=defaultdict(set)
        cell_id_to_cell_num=defaultdict(int)
        row_num=len(layout)
        col_num=len(layout[0])

        for row_id in range(row_num):
            for col_id in range(col_num):
                cell_id=layout[row_id][col_id]
                if cell_id not in cell_id_to_row_and_col_id:
                    cell_id_to_row_and_col_id[cell_id]=(row_id,col_id)

                cell_id_to_cell_num[cell_id]+=1
                #top neighbour
                if row_id-1>=0:
                    top_neighbour=layout[row_id-1][col_id]
                    cell_id_to_neighbor_set[cell_id].add(top_neighbour)
                #left neighbour
                if col_id-1>=0:
                    left_neighbour=layout[row_id][col_id-1]
                    cell_id_to_neighbor_set[cell_id].add(left_neighbour)
                #right neighbour
                if col_id+1<col_num:
                    right_neighbour=layout[row_id][col_id+1]
                    cell_id_to_neighbor_set[cell_id].add(right_neighbour)
                # down neighbour
                if row_id+1<row_num:
                    down_neighbour=layout[row_id+1][col_id]
                    cell_id_to_neighbor_set[cell_id].add(down_neighbour)
        
        for cell_id in range(cell_num):
            row_id=cell_id_to_row_and_col_id[cell_id][0]
            col_id=cell_id_to_row_and_col_id[cell_id][1]
            neighbour_num=len(cell_id_to_neighbor_set[cell_id])
            neighbour_num_feats=[0,0,0,0,0]
            if neighbour_num<=4 and neighbour_num>0:
                neighbour_num_feats[neighbour_num-1]=1
            else:
                neighbour_num_feats[4]=1
            num_cells=cell_id_to_cell_num[cell_id]
            if num_cells>1:
                is_merged_cell=1
            else:
                is_merged_cell=0
            feat_vector=[]
            feat_vector.append(row_id)
            feat_vector.append(col_id)
            feat_vector.extend(neighbour_num_feats)
            feat_vector.append(num_cells)
            feat_vector.append(is_merged_cell)
            feat_vector=np.array(feat_vector)
            spatial_features[cell_id]=feat_vector
        return spatial_features


    def build_features(self,layout,cell_values):
        content_feats=self.build_content_features(cell_values)
        spatial_feats=self.build_spatial_feats(layout,cell_values)
        cell_feats=np.concatenate((content_feats,spatial_feats),axis=1)
        return cell_feats

class AutoEncoder(nn.Layer):
    """Implementation of AutoEncoder for converting 24-dim discrete manual feats to 32-dim continuous feats.
    """
    def __init__(self, enc_dim,manual_feat_dim):
        super(AutoEncoder,self).__init__()
        self.model_name='Auto-Encoder-for-CTC-Model'
        self.enc_dim=enc_dim
        self.manual_feat_dim=manual_feat_dim
        self.encoder=nn.Sequential(
            nn.Linear(manual_feat_dim,enc_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(enc_dim,enc_dim)
        )
        self.decoder=nn.Sequential(
            nn.Linear(enc_dim,manual_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(manual_feat_dim,manual_feat_dim)
        )
    def forward(self, input_vecs):
        assert input_vecs.shape[1]==self.manual_feat_dim
        enc_vec=self.encoder(input_vecs)
        dec_vec=self.decoder(enc_vec)
        return dec_vec,enc_vec 






            