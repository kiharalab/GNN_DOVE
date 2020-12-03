
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from multiprocessing import Pool
from model.layers import GAT_gate

N_atom_features = 28

class GNN_Model(nn.Module):
    def __init__(self, params):
        super(GNN_Model, self).__init__()
        n_graph_layer = params['n_graph_layer']
        d_graph_layer = params['d_graph_layer']
        n_FC_layer = params['n_FC_layer']
        d_FC_layer = params['d_FC_layer']
        self.dropout_rate = params['dropout_rate']


        self.layers1 = [d_graph_layer for i in range(n_graph_layer +1)]
        self.gconv1 = nn.ModuleList \
            ([GAT_gate(self.layers1[i], self.layers1[ i +1]) for i in range(len(self.layers1 ) -1)])

        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i== 0 else
                                 nn.Linear(d_FC_layer, 1) if i == n_FC_layer - 1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])

        self.mu = nn.Parameter(torch.Tensor([params['initial_mu']]).float())
        self.dev = nn.Parameter(torch.Tensor([params['initial_dev']]).float())
        self.embede = nn.Linear(2 * N_atom_features, d_graph_layer, bias=False)
        self.params=params



    def fully_connected(self, c_hs):
        regularization = torch.empty(len(self.FC) * 1 - 1, device=c_hs.device)

        for k in range(len(self.FC)):
            # c_hs = self.FC[k](c_hs)
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)

        c_hs = torch.sigmoid(c_hs)

        return c_hs
    def Formulate_Adj2(self,c_adjs2,c_valid,atom_list,device):
        study_distance = c_adjs2.clone().detach().to(device)  # only focused on where there exist atoms, ignore the area filled with 0
        study_distance = torch.exp(-torch.pow(study_distance - self.mu.expand_as(study_distance), 2) / self.dev)
        filled_value = torch.Tensor([0]).expand_as(study_distance).to(device)
        for batch_idx in range(len(c_adjs2)):
            num_atoms = int(atom_list[batch_idx])
            count_receptor = len(c_valid[batch_idx].nonzero())
            c_adjs2[batch_idx,:count_receptor,count_receptor:num_atoms]=torch.where(c_adjs2[batch_idx,:count_receptor,count_receptor:num_atoms]<=10,study_distance[batch_idx,:count_receptor,count_receptor:num_atoms],filled_value[batch_idx,:count_receptor,count_receptor:num_atoms])
            c_adjs2[batch_idx,count_receptor:num_atoms,:count_receptor]=c_adjs2[batch_idx,:count_receptor,count_receptor:num_atoms].t()
        return c_adjs2

    def get_attention_weight(self,data):
        c_hs, c_adjs1, c_adjs2 = data
        atten1,c_hs1 = self.gconv1[0](c_hs, c_adjs1,request_attention=True)  # filled 0 part will not effect other parts
        atten2,c_hs2 = self.gconv1[0](c_hs, c_adjs2,request_attention=True)
        return atten1,atten2
    def embede_graph(self, data):
        """

        :param data:
        :return: c_hs:batch_size*max_atoms
        """
        c_hs, c_adjs1, c_adjs2= data
        regularization = torch.empty(len(self.gconv1), device=c_hs.device)

        for k in range(len(self.gconv1)):
            c_hs1 = self.gconv1[k](c_hs, c_adjs1)#filled 0 part will not effect other parts
            c_hs2 = self.gconv1[k](c_hs, c_adjs2)
            c_hs = c_hs2 - c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
        #c_hs = c_hs.sum(1)
        return c_hs
    def Get_Prediction(self,c_hs,atom_list):
        prediction=[]
        for batch_idx in range(len(atom_list)):
            num_atoms = int(atom_list[batch_idx])
            tmp_pred=c_hs[batch_idx,:num_atoms]
            tmp_pred=tmp_pred.sum(0)#sum all the used atoms
            #if self.params['debug']:
            #    print("pred feature size",tmp_pred.size())
            prediction.append(tmp_pred)
        prediction = torch.stack(prediction, 0)
        return prediction
    def train_model(self,data,device):
        #get data
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2=self.Formulate_Adj2(c_adjs2,c_valid,num_atoms,device)
        #then do the gate
        c_hs=self.embede_graph((c_hs,c_adjs1,c_adjs2))
        #if self.params['debug']:
        #    print("embedding size",c_hs.size())
        #sum based on the atoms
        c_hs=self.Get_Prediction(c_hs,num_atoms)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs
    def test_model(self, data,device):
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, c_valid, num_atoms,device)
        # then do the gate
        c_hs = self.embede_graph((c_hs, c_adjs1, c_adjs2))
        # sum based on the atoms
        c_hs = self.Get_Prediction(c_hs, num_atoms)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs
    def test_model_final(self,data,device):
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, c_valid, num_atoms, device)
        attention1, attention2 = self.get_attention_weight((c_hs, c_adjs1, c_adjs2))
        # then do the gate
        c_hs = self.embede_graph((c_hs, c_adjs1, c_adjs2))
        # sum based on the atoms
        c_hs = self.Get_Prediction(c_hs, num_atoms)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs,attention1,attention2
    def eval_model_attention(self,data,device):
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, c_valid, num_atoms, device)
        attention1,attention2 = self.get_attention_weight((c_hs, c_adjs1, c_adjs2))
        return attention1,attention2
    def feature_extraction(self,c_hs):
        for k in range(len(self.FC)):
                # c_hs = self.FC[k](c_hs)
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=False)
                c_hs = F.relu(c_hs)

            return c_hs
    def model_gnn_feature(self, data,device):
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, c_valid, num_atoms,device)
        # then do the gate
        c_hs = self.embede_graph((c_hs, c_adjs1, c_adjs2))
        # sum based on the atoms
        c_hs = self.Get_Prediction(c_hs, num_atoms)
        #c_hs = self.fully_connected(c_hs)
        #c_hs = c_hs.view(-1)
        c_hs=self.feature_extraction(c_hs)
        return c_hs