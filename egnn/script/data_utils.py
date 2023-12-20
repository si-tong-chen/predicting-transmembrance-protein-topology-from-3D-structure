import os
import json
import requests
import pathlib
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

import numpy as np
import pandas as pd
from loguru import logger
from torch_geometric import transforms as T
import random
import  pickle
from base import ChangeFormatTMPDataset,ProcessBatchData,AtomInfo,GetAtomPosNum

class LengthMismatchError(Exception):
    pass

class DismatchIndexPadRawData():
    '''
    batch_size = 1 留着 
    '''
    def __init__(self,batch_name,data,df_val):
        self.batch_name = batch_name 
        self.data = data 
        self.df_val = df_val 

    
    def atom_aa_label_mapping(self,sequence, labels):
        '''
        Create a list where each element is a tuple of (atom, amino acid, position label)
        '''
        atom_info=AtomInfo()
        atom_aa_label_mapping = []
        for aa, label in zip(sequence, labels):
            for atom in atom_info.atoms_label[aa]['atoms']:
                atom_aa_label_mapping.append((atom, aa, label))
        
        return atom_aa_label_mapping

    def pad_del_realdata(self,dismatch_indices,dismatch_type,atom_aa_label_mapping,real_atom,pred_label):
        '''
        adding or delete elements in the real data in order to have the same size between predict and real label
        '''
        for i,index in enumerate(dismatch_indices):
            if dismatch_type[i] == 'pred > real':
                # 首先拿到index-1 和 indx+1 的值 如果index+1 没有的话就等于 = index-1的值
                #index+1 的值 是否为7 是7的话 插入（pred_label[index]，index上一个的氨基酸和标签）
                #。                 不是7的话。看pred_index 是否为7 是7 插入（pred_label[index]，inde下一个的氨基酸和标签）
                #                                               不是7 插入（pred_label[index]，inde上一个的氨基酸和标签）
                last_one = atom_aa_label_mapping[index-1]
                if index+1 <len(real_atom):
                    after_one = atom_aa_label_mapping[index+1]
                else:
                    after_one = last_one
        
                if after_one[0]==7:
                     
                    atom_aa_label_mapping.insert(index,(pred_label[index],last_one[1],last_one[2]))
                else:
                    if pred_label[index] == 7:
                        atom_aa_label_mapping.insert(index,(pred_label[index],after_one[1],after_one[2]))
                    else:
                        atom_aa_label_mapping.insert(index,(pred_label[index],last_one[1],last_one[2]))
        
            else:
                pass
    
            
        return atom_aa_label_mapping
        
    
    def match_real_geometric(self):   
        dismatch_index_pred ={}
        dismatch_index_type={}
        after_process_rawdata={}
        

        for name in self.data.keys():
            real_atom=[]
            dismatch_indices = []
            dismatch_type=[]
            pred_label = np.array(self.data[name].x)
    

            
            atom_label = self.df_val[self.df_val['uniprot_id_low'] == name]['atom_label'].item()
                
            real_atom.extend(atom_label)
                
            real_atom_label=np.array(real_atom)
                
            if len(pred_label) > len(real_atom_label): #输出的index是真实的index，真实的比预测的短，找到预测的index（其他的算法） 加入到真实的index位置
                n = 0  
                m = 0  
                while n < len(pred_label):
                    if m >= len(real_atom_label) or pred_label[n] != real_atom_label[m]:
                        
                        dismatch_indices.append(n)  
                        dismatch_type.append('pred > real') 
                        n += 1  
                    else: 
                       
                        n += 1
                        m += 1
                dismatch_index_pred[name] = dismatch_indices
                dismatch_index_type[name] = 'pred > real'
        
            if len(pred_label) < len(real_atom_label):#输出的真实的index，真实的比预测的长，按照这个index 减去对应的真实的位置
                n = 0  
                for m in range(len(real_atom_label)):
                    if n < len(pred_label) and pred_label[n] == real_atom_label[m]:
                        n += 1
                    else:
                        dismatch_indices.append(m)
                        dismatch_type.append('pred < real')
                dismatch_index_pred[name] = dismatch_indices
                dismatch_index_type[name] = 'pred < real'
        
            sequence=[]
            labels=[]
            
            sequence.extend(list(self.df_val[self.df_val['uniprot_id_low'] == name]['seq'])[0])
            labels.extend(list(self.df_val[self.df_val['uniprot_id_low'] == name]['raw_label'])[0][0])

            atom_aa_label_mapping_a = self.atom_aa_label_mapping(sequence,labels)
            atom_aa_label_mapping_b = self.pad_del_realdata(dismatch_indices,dismatch_type,atom_aa_label_mapping_a,real_atom,pred_label)
            
            if len(atom_aa_label_mapping_b) != len(pred_label):
                raise LengthMismatchError("Lengths of raw data and predict label do not match.")
            after_process_rawdata[name] = atom_aa_label_mapping_b
       

        return after_process_rawdata,dismatch_index_pred,dismatch_index_type
    




class ProcessRawData():
    ''''
    留着 并且以dataframe的格式作为基础
    
    '''
    def __init__(
        self,
        path: str,
        raw_data_name: str,
        batch_size: int = 1,
        dataset_name_after_process: str = "tmp", 
        in_memory: bool = False,
        pin_memory: bool = True,
        num_workers: int = 4,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False
    ) -> None:
        self.dataset_name_after_process = dataset_name_after_process
        self.raw_data_name = raw_data_name
        self.batch_size = batch_size
        self.all_data = {}
        self.root = pathlib.Path(path) /"dataset"/ self.dataset_name_after_process 
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        
        self.pdb_dir = pathlib.Path(path) / "dataset"/ "structures"
        if not os.path.exists(self.pdb_dir):
            os.makedirs(self.pdb_dir, exist_ok=True)
        self.raw_data_dir = pathlib.Path(path) / "dataset"/ self.raw_data_name
        self.partitions_dir = pathlib.Path(path) / "dataset"/ 'DeepTMHMM.partitions.json'
        self.parsedata_pd = pathlib.Path(path) /"dataset"/ 'parse raw data'
        if not os.path.exists(self.parsedata_pd):
            os.makedirs(self.parsedata_pd, exist_ok=True)


    def setup(self, stage: Optional[str] = None):
        for split in {'cv0','cv1','cv2','cv3','cv4'}:
            data = self.parse_dataset(split)
            self.all_data[split] = data
            logger.info("Preprocessing " + split +" data")

    def file_paths_process(self,split: Literal['cv0','cv1','cv2','cv3','cv4']):
        directory = self.pdb_dir/ f"{split}"
        pdb_names = os.listdir(directory)
        file_paths = [str(directory / name) for name in pdb_names]
        return file_paths

    def split_data(self):
        '''
        split the data into train, validation, test and store in the files
        '''
        processor= ChangeFormatTMPDataset(self.raw_data_dir)
        data_dict = processor.get_data()
        null_structure = ['Q5I6C7', 'Q05470', 'Q6KC79', 'Q96Q15', 'P36022', 'Q96T58', 'Q9VDW6', 'Q3KNY0', 'Q14315', 'Q7TMY8', 'Q9SMH5', 'Q9VC56', 'Q8WXX0', 'Q01484', 'Q5VT06', 'Q8IZQ1', 'Q9P2D1', 
                  'F8VPN2', 'Q9U943', 'O83276', 'P14217', 'Q868Z9', 'O83774', 'Q61001', 'P98161', 'Q9UKN1', 'P04875', 'P0DTC2', 'P29994', 'Q14789', 'P69332', 'Q9VKA4']


        f = open(self.partitions_dir)
        cv_data = json.load(f)
        name_list = ['cv0','cv1','cv2','cv3','cv4']


        for name in name_list:
      
            cv0 = cv_data[name]
            cv0_name_list = [cv0[i]['id'] for i in range(len(cv0))]
            cv_name_list = [item for item in cv0_name_list if item not in null_structure]

            cv0_data= {name: data_dict[name] for name in cv_name_list}
            output_path_train = self.root /f"{name}.json"
            with open(output_path_train, 'w') as json_file:
                json.dump(cv0_data, json_file, indent=4)


    def parse_dataset(self, split: Literal['cv0','cv1','cv2','cv3','cv4']) -> pd.DataFrame:
        """
        processing the raw data DeepTMHMM(sequence)
        """

        data = json.load(open(self.root / f"{split}.json", "r"))
        data = pd.DataFrame.from_records(data).T
        data["uniprot_id"] = data.index
        data.columns = [
            "seq",
            "raw_label",
            "protein_type",
            "atom_length",
            "atom_label",
            "uniprot_id",
        ]
        data["uniprot_id_low"] = data["uniprot_id"].str.lower()
        data["seq_length"] = data["seq"].apply(len)

        if (data['uniprot_id_low'] == 'q841a2').any():
            data = data[data['uniprot_id_low'] != 'q841a2']
        if (data['uniprot_id_low'] == 'd6r8x8').any():
            data = data[data['uniprot_id_low'] != 'd6r8x8']

        with open(self.parsedata_pd/f"{split}.pickle", 'wb') as file:
            pickle.dump(data, file)

    

    def get_alphafold_db_pdb(self, protein_id: str, out_path: str) -> bool:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        requestURL = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
        r = requests.get(requestURL)

        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
                return True
        else:
            return False

    def download(self, split: Literal['cv0','cv1','cv2','cv3','cv4']):
        """
        download the 3D protein structures from alphafold

        """

        with open(self.parsedata_pd / f"{split}.pickle", 'rb') as file:
            data = pickle.load(file)


        uniprot_ids = list(data["uniprot_id"].unique())

        to_download = [
            id
            for id in uniprot_ids
            if not os.path.exists(self.pdb_dir/split/ f"{id}.pdb")
        ]
 
        logger.info(f"Downloading {len(to_download)} PDBs...")
        for id in to_download:
            out_path = os.path.join(self.pdb_dir,split, f"{id.lower()}.pdb")
            success = self.get_alphafold_db_pdb(id, out_path)
            if success:
                logger.info(f"Downloaded PDB for {id}")
            else:
                logger.warning(f"Failed to download PDB for {id}")

    def run(self):
        name_list=['cv0','cv1','cv2','cv3','cv4']
        self.split_data()
        for name in name_list:
            self.parse_dataset(name) 
            self.download(name)



class ParseStructure():
    '''
    deal with the download pdb and store them
    '''
    def __init__(self,
                 path: str,):

        self.output_path_parse_structure_dataset = pathlib.Path(path) / "dataset"/"parse sturcture dataset"
        self.pdb_dir = pathlib.Path(path) / "dataset"/ "structures"
        if not os.path.exists(self.output_path_parse_structure_dataset):
            os.makedirs(self.output_path_parse_structure_dataset, exist_ok=True)


    def file_paths_process(self,split: Literal['cv0','cv1','cv2','cv3','cv4']):
        directory = self.pdb_dir/ f"{split}"
        pdb_names = os.listdir(directory)
        file_paths = [str(directory / name) for name in pdb_names]
        return file_paths
    
    def store_strcture_data_after_parse(self,split: Literal['cv0','cv1','cv2','cv3','cv4']):
        file_paths = self.file_paths_process(split)
        processor = GetAtomPosNum(file_paths)
        atomposnump = processor.get_all()

        with open(self.output_path_parse_structure_dataset/f"{split}.pickle", 'wb') as file:
            pickle.dump(atomposnump, file)

    def run(self):
        # this is store the structer infromation after parsing, run this can get all we need 
        name_list=['cv0','cv1','cv2','cv3','cv4']
        [self.store_strcture_data_after_parse(split) for split in name_list] 













    
      
  
