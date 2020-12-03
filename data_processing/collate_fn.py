import numpy as np
import torch

def collate_fn(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])

    H = np.zeros((len(batch), max_natoms, 56))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    V = np.zeros((len(batch), max_natoms))
    #keys = []
    Atoms_Number=[]
    for i in range(len(batch)):
        natom = len(batch[i]['H'])

        H[i, :natom] = batch[i]['H']
        A1[i, :natom, :natom] = batch[i]['A1']
        A2[i, :natom, :natom] = batch[i]['A2']
        V[i, :natom] = batch[i]['V']
        #keys.append(batch[i]['key'])
        Atoms_Number.append(natom)
    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    V = torch.from_numpy(V).float()
    Atoms_Number=torch.Tensor(Atoms_Number)

    return H, A1, A2, V,Atoms_Number #, keys