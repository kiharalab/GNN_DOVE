# Publication:  "Protein Docking Model Evaluation by Graph Neural Networks", Xiao Wang, Sean T Flannery and Daisuke Kihara,  (2020)

#GNN-Dove is a computational tool using graph neural network that can evaluate the quality of docking protein-complexes.

#Copyright (C) 2020 Xiao Wang, Sean T Flannery, Daisuke Kihara, and Purdue University.

#License: GPL v3 for academic use. (For commercial use, please contact us for different licensing.)

#Contact: Daisuke Kihara (dkihara@purdue.edu)

#

# This program is free software: you can redistribute it and/or modify

# it under the terms of the GNU General Public License as published by

# the Free Software Foundation, version 3.

#

# This program is distributed in the hope that it will be useful,

# but WITHOUT ANY WARRANTY; without even the implied warranty of

# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the

# GNU General Public License V3 for more details.

#

# You should have received a copy of the GNU v3.0 General Public License

# along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.en.html.

import os
from ops.Timer_Control import set_timeout,after_timeout
RESIDUE_Forbidden_SET={"FAD"}

def Extract_Interface(pdb_path):
    """
    specially for 2 docking models
    :param pdb_path:docking model path
    :rcount: receptor atom numbers
    :return:
    extract a receptor and ligand, meanwhile, write two files of the receptor interface part, ligand interface part
    """
    receptor_list=[]
    ligand_list=[]
    rlist=[]
    llist=[]
    count_r=0
    count_l=0
    with open(pdb_path,'r') as file:
        line = file.readline()               # call readline()
        while line[0:4]!='ATOM':
            line=file.readline()
        atomid = 0
        count = 1
        goon = False
        chain_id = line[21]
        residue_type = line[17:20]
        pre_residue_type = residue_type
        tmp_list = []
        pre_residue_id = 0
        pre_chain_id = line[21]
        first_change=True
        while line:

            dat_in = line[0:80].split()
            if len(dat_in) == 0:
                line = file.readline()
                continue

            if (dat_in[0] == 'ATOM'):
                chain_id = line[21]
                residue_id = int(line[23:26])

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residue_type = line[17:20]
                # First try CA distance of contact map
                atom_type = line[13:16].strip()
                if chain_id=="B":
                    goon=True
                if (goon):
                    if first_change:
                        rlist.append(tmp_list)
                        tmp_list = []
                        tmp_list.append([x, y, z, atom_type, count_l])
                        count_l += 1
                        ligand_list.append(line)
                        first_change=False
                    else:
                        ligand_list.append(line)  # used to prepare write interface region
                        if pre_residue_type == residue_type:
                            tmp_list.append([x, y, z, atom_type, count_l])
                        else:
                            llist.append(tmp_list)
                            tmp_list = []
                            tmp_list.append([x, y, z, atom_type, count_l])
                        count_l += 1
                else:
                    receptor_list.append(line)
                    if pre_residue_type == residue_type:
                        tmp_list.append([x, y, z, atom_type, count_r])
                    else:
                        rlist.append(tmp_list)
                        tmp_list = []
                        tmp_list.append([x, y, z, atom_type, count_r])
                    count_r += 1

                atomid = int(dat_in[1])
                chain_id = line[21]
                count = count + 1
                pre_residue_type = residue_type
                pre_residue_id = residue_id
                pre_chain_id = chain_id
            line = file.readline()
    print("Extracting %d/%d atoms for receptor, %d/%d atoms for ligand"%(len(receptor_list),count_r,len(ligand_list),count_l))
    final_receptor, final_ligand=Form_interface(rlist,llist,receptor_list,ligand_list)
    #write that into our path
    rpath=Write_Interface(final_receptor,pdb_path,".rinterface")
    lpath=Write_Interface(final_ligand, pdb_path, ".linterface")
    return rpath,lpath
@set_timeout(100000, after_timeout)
def Form_interface(rlist,llist,receptor_list,ligand_list,cut_off=10):

    cut_off=cut_off**2
    r_index=set()
    l_index=set()
    for rindex,item1 in enumerate(rlist):
        for lindex,item2 in enumerate(llist):
            min_distance=1000000
            residue1_len=len(item1)
            residue2_len=len(item2)
            for m in range(residue1_len):
                atom1=item1[m]
                for n in range(residue2_len):
                    atom2=item2[n]
                    distance=0
                    for k in range(3):
                        distance+=(atom1[k]-atom2[k])**2
                    #distance=np.linalg.norm(atom1[:3]-atom2[:3])
                    if distance<=min_distance:
                        min_distance=distance
            if min_distance<=cut_off:
                if rindex not in r_index:
                    r_index.add(rindex)
                if lindex not in l_index:
                    l_index.add(lindex)
    r_index=list(r_index)
    l_index=list(l_index)
    newrlist=[]
    for k in range(len(r_index)):
        newrlist.append(rlist[r_index[k]])
    newllist=[]
    for k in range(len(l_index)):
        newllist.append(llist[l_index[k]])
    print("After filtering the interface region, %d/%d residue in receptor, %d/%d residue in ligand" % (len(newrlist),len(rlist), len(newllist),len(llist)))
    #get the line to write new interface file
    final_receptor=[]
    final_ligand=[]
    for residue in newrlist:
        for tmp_atom in residue:
            our_index=tmp_atom[4]
            final_receptor.append(receptor_list[our_index])

    for residue in newllist:
        for tmp_atom in residue:
            our_index=tmp_atom[4]
            #print (our_index)
            final_ligand.append(ligand_list[our_index])
    print("After filtering the interface region, %d receptor, %d ligand"%(len(final_receptor),len(final_ligand)))

    return final_receptor,final_ligand

def Write_Interface(line_list,pdb_path,ext_file):
    new_path=pdb_path[:-4]+ext_file
    with open(new_path,'w') as file:
        for line in line_list:
            #check residue in the common residue or not. If not, no write for this residue
            residue_type = line[17:20]
            if residue_type in RESIDUE_Forbidden_SET:
                continue
            file.write(line)
    return new_path
