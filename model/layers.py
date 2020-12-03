import torch
import torch.nn.functional as F
import torch.nn as nn


class GAT_gate(nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        # self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature * 2, 1)#default bias=True
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj,request_attention=False):
        h = self.W(x)#x'=W*x_in
        batch_size = h.size()[0]
        N = h.size()[1]#num_atoms
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h, self.A), h))#A is E in the paper,
        #This function provides a way of computing multilinear expressions (i.e. sums of products) using the Einstein summation convention.
        e = e + e.permute((0, 2, 1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, h)
        output_attention=attention
        attention = attention * adj#final attention a_ij
        h_prime = F.relu(torch.einsum('aij,ajk->aik', (attention, h)))#x'' in the paper

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))##calculate z_i
        retval = coeff * x + (1 - coeff) * h_prime#final output,linear combination
        if request_attention:
            return output_attention,retval
        else:
            return retval
    def forward_single(self, x, adj):
        h = self.W(x)#x'=W*x_in
        #batch_size = h.size()[0]
        #N = h.size()[1]#num_atoms
        e = torch.einsum('jl,kl->jk', (torch.matmul(h, self.A), h))#A is E in the paper,
        #This function provides a way of computing multilinear expressions (i.e. sums of products) using the Einstein summation convention.
        e = e + e.permute((0, 1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, h)
        attention = attention * adj#final attention a_ij
        h_prime = F.relu(torch.einsum('ij,jk->ik', (attention, h)))#x'' in the paper

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))##calculate z_i
        retval = coeff * x + (1 - coeff) * h_prime#final output,linear combination
        return retval