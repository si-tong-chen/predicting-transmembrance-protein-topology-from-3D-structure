from task import CreateDataBeforeBatch,TMPDataset,CreateLable,MapAtomNode,node_accuracy,GaussianSmoothing,batchdata
from data_utils import ProcessRawData,ParseStructure
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from egnnmodel import EGNNModel
import numpy as np
from scipy.ndimage import gaussian_filter1d
from metrics_utils import label_list_to_topology,is_topologies_equal



class TMPTest():
    def __init__(self,TM_test,file_name,path,batch_size,num,setup,modelpath):
        self.TM_test = TM_test
        self.file_name=file_name
        self.path = path
        self.batch_size = batch_size 
        self.num=num
        self.setup=setup
        self.modelpath=modelpath
        

    def testtogther(self):
        test_dataset = TMPDataset(self.TM_test)
        test_data_loader = DataLoader(test_dataset, batch_size=100, shuffle=True,collate_fn=lambda x: x,pin_memory=True)

        # put test label togther
        test_residual_level_label={}
        test_atom_levl_label = {}
        test_dismatch_index_pred ={}
        test_dismatch_index_type ={}
        for data_batch in test_data_loader:
            batchname=[data_batch[num]['name'] for num in range(len(data_batch))]

            labelprocessor=CreateLable(batchname,data_batch,self.path,self.file_name)
            atom_level_label_dict,redidual_level_label_dict,dismatch_index_pred,dismatch_index_type,_,_,df_test=labelprocessor.labeldispatcher(self.setup,subset='test')

            test_atom_levl_label.update(atom_level_label_dict) 
            test_residual_level_label.update(redidual_level_label_dict) 
            test_dismatch_index_pred.update(dismatch_index_pred)
            test_dismatch_index_type.update(dismatch_index_type)
        
        return test_atom_levl_label,test_residual_level_label,test_dismatch_index_pred,test_dismatch_index_type,df_test

    
    def testmodel(self,):
        test_atom_levl_label,test_residual_level_label,test_dismatch_index_pred,test_dismatch_index_type,df_test=self.testtogther()

        test_dataset = TMPDataset(self.TM_test)
        test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=lambda x: x,pin_memory=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        max_len= 20000


        model = EGNNModel(out_dim=6,max_len=max_len,num_layers=5,emb_dim=128,residual=True,dropout=0.1).to(device)
    

        model.load_state_dict(torch.load(self.modelpath))

        criterion = nn.CrossEntropyLoss()
        smoothing = GaussianSmoothing(6, 29, 5)

        test_predict_node_label_lis = []

        check_zero = []
        test_node_acc_list = []
        test_node_acc_binary_list = []

        test_atom_acc_list = []
        test_atom_correct = []
        test_atom_total = []

        baseline_atom_correct = []
        baseline_atom_acc_list = []
        baseline_node_acc_list = []
        setup_test_real_node_label=[]
        model.eval()
        with torch.no_grad():
            for data_batch in test_data_loader:
                batchname=[data_batch[num]['name'] for num in range(len(data_batch))]
                label_part = [value.unsqueeze(0) for name in batchname for value in test_atom_levl_label[name].to_dense()]
                atom_levl_label = torch.cat(label_part).to(device)
                residual_level_label = [value for name in batchname for value in test_residual_level_label[name]]
                data = batchdata(data_batch) 

                outputs = model(data.to(device)) 
                prediction = outputs["node_embedding"] 

                predicted = torch.reshape(prediction.to('cpu'), (1,prediction.shape[1], prediction.shape[0]))
                predicted = F.pad(predicted, (14, 14), mode='reflect')
                predicted = smoothing(predicted)
                prediction_Gauss = torch.reshape(predicted, (prediction.shape[0], prediction.shape[1]))

                loss = criterion(prediction_Gauss.to(device), atom_levl_label)

                _, predicted = torch.max(prediction_Gauss.to(device), 1) 
                correct = (predicted == atom_levl_label ).sum().item()
                total = atom_levl_label.size(0)
                atom_level_accuracy =  correct / total

                test_atom_acc_list.append(atom_level_accuracy)
                test_atom_correct.append(correct)
                test_atom_total.append(total)

                # Baseline atom accuracy
                baseline_atom = torch.zeros_like(atom_levl_label) # note that in this case the most frequent class is class 0
                baseline_correct = (baseline_atom == atom_levl_label).sum().item()
                baseline_atom_correct.append(baseline_correct)

                processor = MapAtomNode(predicted.cpu(),batchname,test_dismatch_index_pred,test_dismatch_index_type,df_test)
                test_predict_node_label = processor.map_atom_node() 
                test_predict_node_label_lis.append(test_predict_node_label)



                accuracy_list = [1 if x == y else 0 for x, y in zip(test_predict_node_label, residual_level_label)]
                test_node_acc_binary_list += accuracy_list
                residual_level_accuracy = node_accuracy(test_predict_node_label, residual_level_label)
                test_node_acc_list.append(residual_level_accuracy)

            # Baseline node accuracy
                baseline_node_label = np.zeros_like(np.array(residual_level_label)) # note that in this case the most frequent class is class 0
                baseline_accuracy_list = [1 if x == y else 0 for x, y in zip(baseline_node_label,residual_level_label)]
                baseline_node_acc_list += baseline_accuracy_list

                setup_test_real_node_label.append(residual_level_label)

        return test_node_acc_list,test_node_acc_list,test_node_acc_binary_list,test_node_acc_binary_list,test_atom_acc_list,test_atom_correct,test_atom_total,test_predict_node_label_lis,setup_test_real_node_label
    

    def printresult(self):
        
        test_node_acc_list,test_node_acc_list,test_node_acc_binary_list,test_node_acc_binary_list,test_atom_acc_list,test_atom_correct,test_atom_total,test_predict_node_label_lis,setup_test_real_node_label=self.testmodel()
        final_node_acc = sum(test_node_acc_list)/len(test_node_acc_list)
       
        
        
        print("Node acc:", final_node_acc)

        final_node_binary_acc = sum(test_node_acc_binary_list)/len(test_node_acc_binary_list)
        print("Node binary acc:", final_node_binary_acc)

        final_atom_acc = sum(test_atom_acc_list)/len(test_atom_acc_list)
        print("Avg atom acc:", final_atom_acc)

        total_atom_acc = sum(test_atom_correct)/len(test_atom_total)
        print("Total atom acc:", total_atom_acc)

        test_resul = []
        for i in range(0, len(test_predict_node_label_lis)):
        #for j in range(0, len(test_BETA_predict_node_label_lis[i])):
            topo_A = label_list_to_topology(test_predict_node_label_lis[i])
            topo_B = label_list_to_topology(setup_test_real_node_label[i])

            test_resul.append(is_topologies_equal(topo_A, topo_B, self.num))

        print("Correct topology without Gaussian smoothing:", sum(test_resul)/len(test_resul))




        resul = []
        acc_lis = []
        for i in range(0, len(test_predict_node_label_lis)):

            #for j in range(0, len(test_BETA_predict_node_label_lis[i])):
            topo_A = label_list_to_topology(test_predict_node_label_lis[i])
            topo_B = label_list_to_topology(setup_test_real_node_label[i])

            #print(topo_A)
            #print(topo_B)

            resul.append(is_topologies_equal(topo_A, topo_B, self.num))
            #print(SP_TM_resul.append(is_topologies_equal(topo_A, topo_B, 5)))
            node_level_accuracy = node_accuracy(test_predict_node_label_lis[i], setup_test_real_node_label[i])
            acc_lis.append(node_level_accuracy)

        final_node_acc = sum(acc_lis)/len(acc_lis)
        print("Node acc before smoothing:", final_node_acc)
        print("Before smoothing correct topology:", sum(resul)/len(resul), "\n")


        resul_2 = []
        acc_lis = []
        for i in range(0, len(test_predict_node_label_lis)):
            smoothened = gaussian_filter1d(test_predict_node_label_lis[i], sigma=3)  # using same Gaussian filter setup as TMBED article

            topo_A = label_list_to_topology(smoothened.tolist())
            topo_B = label_list_to_topology(setup_test_real_node_label[i])


            resul_2.append(is_topologies_equal(topo_A, topo_B, self.num))
            node_level_accuracy = node_accuracy(smoothened.tolist(), setup_test_real_node_label[i])
            acc_lis.append(node_level_accuracy)


        final_node_acc = sum(acc_lis)/len(acc_lis)
        print("Node acc after Gaussian smoothing:", final_node_acc)
        print("After Gaussian smoothing correct topology:", sum(resul_2)/len(resul_2), "\n")




                        

