from data_utils import ParseStructure,ProcessRawData,DismatchIndexPadRawData
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data, Batch
from base import ProcessBatchData,AtomInfo
import pandas as pd
import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from torch import nn
import numbers
import torch
from torch.nn import functional as F
import pathlib
import torch
from torch.utils.data import DataLoader

import pickle
from collections import Counter
import numpy as np


one_hot_encoding = {
        'I': [1, 0, 0, 0, 0, 0],
        'O': [0, 1, 0, 0, 0, 0],
        'P': [0, 0, 1, 0, 0, 0],
        'S': [0, 0, 0, 1, 0, 0],
        'M': [0, 0, 0, 0, 1, 0],
        'B': [0, 0, 0, 0, 0, 1]
    }
label_dict = {'I': 0, 'O': 1, 'P': 2, 'S': 3, 'M': 4, 'B': 5}

def batchdata(data_batch):
     batch_pos_data = [data_batch[num]['data']['pos'] for num in range(len(data_batch))]
     batch_num_data=[data_batch[num]['data']['num'] for num in range(len(data_batch))]
     batch_data =Batch.from_data_list([Data(pos=batch_pos_data[num], x=batch_num_data[num]) for num in range(len(batch_num_data))])
     batch_data.edge_index = radius_graph(batch_data.pos, r=8, max_num_neighbors=32,batch=batch_data.batch)
     
     row, col = batch_data.edge_index
     batch_data.edge_weight = (batch_data.pos[row] - batch_data.pos[col]).norm(dim=-1)
     return batch_data


def custom_collate(batch):
    # 检查是否需要处理DataBatch类型的对象
    if isinstance(batch[0], Data):
        return Batch.from_data_list(batch)
    else:
        # 默认情况下，使用PyTorch的默认collate_fn函数处理其他类型的数据
        return torch.utils.data.dataloader.default_collate(batch)
    
def node_accuracy(val_predict_node_label,val_real_node_label):
    ''''
    calculate the accuracy of the node label
    '''
    accuracy_list = [1 if x == y else 0 for x, y in zip(val_predict_node_label, val_real_node_label)]
    correct_count = sum(accuracy_list)
    accuracy = correct_count / len(val_predict_node_label)
    
    return accuracy


class LengthMismatchError(Exception):
    pass
class ValueError(Exception):
    pass



# class CreateDataBeforeBatch():
#     def __init__(self,
#                  path:str):
#         self.output_path_parse_structure_dataset = pathlib.Path(path) / "dataset"/"parse sturcture dataset"


#     def get_data(self, split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']):
#         file_path = self.output_path_parse_structure_dataset/f"{split}.pickle"
#         with open(file_path, 'rb') as file:
#             loaded_data = pickle.load(file)

#         # for name in list(loaded_data.keys()):
#         #     data =Batch.from_data_list([Data(pos=loaded_data[name]['pos'], x=loaded_data[name]['num'])])
#         #     data.edge_index = radius_graph(data.pos, r=8, max_num_neighbors=32,batch=data.batch)
#         #     row, col = data.edge_index
#         #     data.edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
#         #     data_dict_model[name]= data 
#         #     #look for the max length of batches to set the max of static emmbedding 
#         #     single_len_list.append(len(data.x))
#         # single_len_list.sort(reverse=True)
#         return loaded_data

class CreateDataBeforeBatch():
    def __init__(self,
                 path:str):
        self.output_path_parse_structure_dataset = pathlib.Path(path) / "dataset"/"parse sturcture dataset"


    def get_data(self, split: Literal['setup1','setup2','setup3','setup4','setup5']):
        if split == 'setup1':
            train_list =  ['cv0','cv1','cv2']
            val_list = 'cv3'
            test_list = 'cv4'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{'cv3'}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{'cv4'}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)
        
        if split == 'setup2':
            train_list =  ['cv1','cv2','cv3']
            val_list = 'cv4'
            test_list = 'cv0'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{val_list}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{test_list}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)
    
        if split == 'setup3':
            train_list =  ['cv2','cv3','cv4']
            val_list = 'cv0'
            test_list = 'cv1'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{val_list}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{test_list}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)

        if split == 'setup4':
            train_list =  ['cv3','cv4','cv0']
            val_list = 'cv1'
            test_list = 'cv2'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{val_list}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{test_list}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)

        if split == 'setup5':
            train_list =  ['cv4','cv0','cv1']
            val_list = 'cv2'
            test_list = 'cv3'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{val_list}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{test_list}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)

        return train_data,val_data,test_data
    

class CreateLable(ProcessRawData):
    def __init__(self,batchname,data_batch,path,raw_data_name):
        super().__init__(path,raw_data_name)
        self.batchname=batchname
        self.data_batch = data_batch
        self.parsedata_pd = pathlib.Path(path) /"dataset"/ 'parse raw data'
       
    

    def creat_one_hot_label(self, batch_map):
            one_hot_encoded_list = []
            for _, _, label in batch_map:
                one_hot_label = one_hot_encoding[label]
                one_hot_encoded_list.append(one_hot_label)

            return one_hot_encoded_list
    
    def df_dataset(self,split: Literal['setup1','setup2','setup3','setup4','setup5']):

        '''
        amend the dataset assembling the methods to creat a big tabel including the all of data information
        The sequence,label and atom break down like (7,M,I) in order to match the predict label and make the real lable
        '''
        if split == 'setup1':
            train_list =  ['cv0','cv1','cv2']
            val_list = 'cv3'
            test_list = 'cv4'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{'cv3'}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{'cv4'}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe           
            
        if split == 'setup2':
            train_list =  ['cv1','cv2','cv3']
            val_list = 'cv4'
            test_list = 'cv0'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{val_list}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{test_list}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe  

        if split == 'setup3':
            train_list =  ['cv2','cv3','cv4']
            val_list = 'cv0'
            test_list = 'cv1'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{val_list}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{test_list}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe  
        
        if split == 'setup4':
            train_list =  ['cv3','cv4','cv0']
            val_list = 'cv1'
            test_list = 'cv2'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{val_list}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{test_list}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe  

        if split == 'setup5':
            train_list =  ['cv4','cv0','cv1']
            val_list = 'cv2'
            test_list = 'cv3'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{val_list}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{test_list}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe 

        return  train_df,val_df,test_df







    def amend_dataset(self,split: Literal['setup1','setup2','setup3','setup4','setup5'],subset: Literal['train', 'val', 'test'] = 'train'):
        train_df,val_df,test_df = self.df_dataset(split)
        if subset == 'train':
            data_dict_model = {}
            for num in range(len(self.data_batch)):
                data =Batch.from_data_list([Data(pos=self.data_batch[num]['data']['pos'], x=self.data_batch[num]['data']['num'])])
                data.edge_index = radius_graph(data.pos, r=8, max_num_neighbors=32,batch=data.batch)
                row, col = data.edge_index
                data.edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
                name = self.data_batch[num]['name']
                data_dict_model[name]= data 

            precossor = DismatchIndexPadRawData(self.batchname,data_dict_model,train_df)
            after_process_rawdata,dismatch_index_pred,dismatch_index_type= precossor.match_real_geometric()
        
        if subset == 'val':
            data_dict_model = {}
            for num in range(len(self.data_batch)):
                data =Batch.from_data_list([Data(pos=self.data_batch[num]['data']['pos'], x=self.data_batch[num]['data']['num'])])
                data.edge_index = radius_graph(data.pos, r=8, max_num_neighbors=32,batch=data.batch)
                row, col = data.edge_index
                data.edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
                name = self.data_batch[num]['name']
                data_dict_model[name]= data 

            precossor = DismatchIndexPadRawData(self.batchname,data_dict_model,val_df)
            after_process_rawdata,dismatch_index_pred,dismatch_index_type= precossor.match_real_geometric()

        if subset == 'test':
            data_dict_model = {}
            for num in range(len(self.data_batch)):
                data =Batch.from_data_list([Data(pos=self.data_batch[num]['data']['pos'], x=self.data_batch[num]['data']['num'])])
                data.edge_index = radius_graph(data.pos, r=8, max_num_neighbors=32,batch=data.batch)
                row, col = data.edge_index
                data.edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
                name = self.data_batch[num]['name']
                data_dict_model[name]= data 

            precossor = DismatchIndexPadRawData(self.batchname,data_dict_model,test_df)
            after_process_rawdata,dismatch_index_pred,dismatch_index_type= precossor.match_real_geometric()

  
        return after_process_rawdata,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df
    


    def createatomlevellable(self,split: Literal['setup1','setup2','setup3','setup4','setup5'],subset: Literal['train', 'val', 'test'] = 'train'):
  
        after_process_rawdata,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df = self.amend_dataset(split,subset)
    

        atom_level_label_dict = {}
        for name in self.batchname:
            one_hot_encoded_list=self.creat_one_hot_label(after_process_rawdata[name])
            sequence = torch.argmax(torch.tensor(one_hot_encoded_list), dim=1)
            atom_level_label_dict[name]=sequence
        return atom_level_label_dict,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df
    

    
    def creatresiduallevellabel(self,split: Literal['setup1','setup2','setup3','setup4','setup5'],subset: Literal['train', 'val', 'test'] = 'train'):
        train_df,val_df,test_df = self.df_dataset(split)
        if subset == 'train':       
            real_node_level_label_dict={}
            for name in self.batchname:
                filtered_df= train_df[train_df['uniprot_id_low'] == name]
                rawlabel_total_list =[letter for sublist in filtered_df['raw_label'] for item in sublist for letter in item]
                num_total_list = [label_dict[char] for char in rawlabel_total_list]

                real_node_level_label_dict[name] = num_total_list

        if subset == 'val':       
            real_node_level_label_dict={}
            for name in self.batchname:
                filtered_df= val_df[val_df['uniprot_id_low'] == name]
                rawlabel_total_list =[letter for sublist in filtered_df['raw_label'] for item in sublist for letter in item]
                num_total_list = [label_dict[char] for char in rawlabel_total_list]

                real_node_level_label_dict[name] = num_total_list
        if subset == 'test':       
            real_node_level_label_dict={}
            for name in self.batchname:
                filtered_df= test_df[test_df['uniprot_id_low'] == name]
                rawlabel_total_list =[letter for sublist in filtered_df['raw_label'] for item in sublist for letter in item]
                num_total_list = [label_dict[char] for char in rawlabel_total_list]

                real_node_level_label_dict[name] = num_total_list

        return real_node_level_label_dict
    
    def labeldispatcher(self,split: Literal['setup1','setup2','setup3','setup4','setup5'],subset: Literal['train', 'val', 'test'] = 'train'):

        atom_level_label_dict,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df=self.createatomlevellable(split,subset)
        real_node_level_label_dict = self.creatresiduallevellabel(split,subset)
        
        return atom_level_label_dict,real_node_level_label_dict,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df







# class CreateLable(ProcessRawData):
#     def __init__(self,batchname,data_batch,root,raw_data_name):
#         super().__init__(root,raw_data_name)
#         self.batchname=batchname
#         self.data_batch = data_batch
    

#     def creat_one_hot_label(self, batch_map):
#             one_hot_encoded_list = []
#             for _, _, label in batch_map:
#                 one_hot_label = one_hot_encoding[label]
#                 one_hot_encoded_list.append(one_hot_label)

#             return one_hot_encoded_list
    
#     def amend_dataset(self,split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']):
#         '''
#         amend the dataset assembling the methods to creat a big tabel including the all of data information
#         The sequence,label and atom break down like (7,M,I) in order to match the predict label and make the real lable
#         '''

        
#         df = super().parse_dataset(split)

#         data_dict_model = {}


#         for num in range(len(self.data_batch)):
#             data =Batch.from_data_list([Data(pos=self.data_batch[num]['data']['pos'], x=self.data_batch[num]['data']['num'])])
#             data.edge_index = radius_graph(data.pos, r=8, max_num_neighbors=32,batch=data.batch)
#             row, col = data.edge_index
#             data.edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
#             name = self.data_batch[num]['name']
#             data_dict_model[name]= data 

#         precossor = DismatchIndexPadRawData(self.batchname,data_dict_model,df)
#         after_process_rawdata,dismatch_index_pred,dismatch_index_type= precossor.match_real_geometric()

#         return after_process_rawdata,dismatch_index_pred,dismatch_index_type,df
    
#     def createatomlevellable(self,split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']):
        
    
        
#         after_process_rawdata,dismatch_index_pred,dismatch_index_type,df = self.amend_dataset(split)
    

#         atom_level_label_dict = {}
#         for name in self.batchname:
#             one_hot_encoded_list=self.creat_one_hot_label(after_process_rawdata[name])
#             sequence = torch.argmax(torch.tensor(one_hot_encoded_list), dim=1)
#             atom_level_label_dict[name]=sequence
#         return atom_level_label_dict,dismatch_index_pred,dismatch_index_type,df
    
#     def creatresiduallevellabel(self,split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']):
        
#         df = super().parse_dataset(split)
#         real_node_level_label_dict={}
#         for name in self.batchname:
#             filtered_df= df[df['uniprot_id_low'] == name]
#             rawlabel_total_list =[letter for sublist in filtered_df['raw_label'] for item in sublist for letter in item]
#             num_total_list = [label_dict[char] for char in rawlabel_total_list]

#             real_node_level_label_dict[name] = num_total_list
        
#         return real_node_level_label_dict
    
#     def labeldispatcher(self,split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']):
#         atom_level_label_dict,dismatch_index_pred,dismatch_index_type,df=self.createatomlevellable(split)
#         real_node_level_label_dict = self.creatresiduallevellabel(split)
#         return atom_level_label_dict,real_node_level_label_dict,dismatch_index_pred,dismatch_index_type,df




class MapAtomNode():
    def __init__(self, predicted,val_batchname,val_dismatch_index_pred,val_dismatch_index_type,df_val):
       
        self.predicted = predicted
        self.val_batchname = val_batchname
        self.val_dismatch_index_pred = val_dismatch_index_pred  
        self.val_dismatch_index_type = val_dismatch_index_type
        self.df_val = df_val
    

    def find_most_frequent_elements(self,lst):
        if not lst:
            return []

        count = Counter(lst)
        max_occurrence = max(count.values())
        most_frequent = [num for num, occ in count.items() if occ == max_occurrence]

        return most_frequent
    

    
    
    def prcoess_predit_label(self):
        '''
        get rid of the wrong index from the predicted label called 'predicted_list'
        and create the length respondding to the node label called 'consecutive_lengths'
        '''
        atom_info=AtomInfo()

        delete_index=0
        seq_total_list = []
        consecutive_lengths=[]
        predicted_list=list(np.array(self.predicted)) # 
        

        for name in self.val_batchname:
            if self.val_dismatch_index_type[name] == 'pred > real':
                index = self.val_dismatch_index_pred[name]
                delete_index +=index[0]
                if 0 <= delete_index < len(predicted_list):
                    del predicted_list[delete_index]
                else: 
                    raise ValueError('index out of range')
            seq_total_list.extend(self.df_val[self.df_val['uniprot_id_low'] == name]['seq'].iloc[0])
            
        for char in seq_total_list:
            consecutive_lengths.append(atom_info.atoms_label[char]['len'])
        
        return predicted_list,consecutive_lengths
    
    def map_atom_node(self):
        '''
        map the atom label to the node label
        '''
        pred_seq = []
        first_one = 0
        last_one = 0
        length_total = 0
        predicted_list,consecutive_lengths = self.prcoess_predit_label()
        for num in consecutive_lengths:
            last_one += num
            list_part = predicted_list[first_one:last_one]
            most_common_one =self.find_most_frequent_elements(list_part)
            pred_seq.append(most_common_one[0])
            first_one = last_one
        for name in self.val_batchname:
            length_total +=self.df_val[self.df_val['uniprot_id_low'] == name]['seq_length'].iloc[0]

        if len(pred_seq) != length_total:
            raise LengthMismatchError("The length of the batch is not equal to the length of the model outcome after processing")

        return pred_seq
    

class TMPDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        # 从DataBatch对象中获取数据
        data = self.data_dict[key]
        
        return {'name': key,
                'data':data}
        # return data, atom_level_label



       



class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=1):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)






























