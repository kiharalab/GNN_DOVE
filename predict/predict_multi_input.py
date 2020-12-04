import os
from ops.os_operation import mkdir
import shutil
import  numpy as np
from data_processing.Prepare_Input import Prepare_Input
from model.GNN_Model import GNN_Model
import torch
from ops.train_utils import count_parameters,initialize_model
from data_processing.collate_fn import collate_fn
from data_processing.Single_Dataset import Single_Dataset
from torch.utils.data import DataLoader
from predict.predict_single_input import init_model,Get_Predictions



def predict_multi_input(input_path, params):
    save_path = os.path.join(os.getcwd(), "Predict_Result")
    mkdir(save_path)
    save_path = os.path.join(save_path, "Multi_Target")
    mkdir(save_path)
    save_path = os.path.join(save_path, "Fold_" + str(params['fold']) + "_Result")
    mkdir(save_path)
    input_path=os.path.abspath(input_path)
    folder_name=os.path.split(input_path)[1]
    save_path = os.path.join(save_path, folder_name)
    mkdir(save_path)

    fold_choice = params['fold']
    # loading the model
    if fold_choice != -1:
        model_path = os.path.join(os.getcwd(), "best_model")
        model_path = os.path.join(model_path, "fold" + str(fold_choice))
        model_path = os.path.join(model_path, "checkpoint.pth.tar")
        model, device = init_model(model_path, params)
    else:
        root_model_path = os.path.join(os.getcwd(), "best_model")
        model_list = []
        for k in range(1, 4):
            model_path = os.path.join(root_model_path, "fold" + str(k))
            model_path = os.path.join(model_path, "checkpoint.pth.tar")
            model, device = init_model(model_path, params)
            model_list.append(model)
        model = model_list

    listfiles=[x for x in os.listdir(input_path) if ".pdb" in x]
    listfiles.sort()
    Study_Name=[]
    Input_File_List=[]
    for item in listfiles:
        input_pdb_path=os.path.join(input_path,item)
        cur_root_path = os.path.join(save_path, item[:-4])
        Study_Name.append(item[:-4])
        mkdir(cur_root_path)
        structure_path=os.path.join(cur_root_path,"Input.pdb")
        shutil.copy(input_pdb_path, structure_path)
        input_file = Prepare_Input(structure_path)
        Input_File_List.append(input_file)
    list_npz = Input_File_List
    dataset = Single_Dataset(list_npz)
    dataloader = DataLoader(dataset, params['batch_size'], shuffle=False,
                            num_workers=params['num_workers'],
                            drop_last=False, collate_fn=collate_fn)

    # prediction
    if fold_choice != -1:
        Final_Pred = Get_Predictions(dataloader, device, model)
    else:
        Final_Pred = []
        for cur_model in model:
            tmp_pred = Get_Predictions(dataloader, device, cur_model)
            Final_Pred.append(tmp_pred)
        Final_Pred = np.mean(Final_Pred, axis=0)
    pred_path = os.path.join(save_path, 'Predict.txt')
    with open(pred_path, 'w') as file:
        file.write("Input\tScore\n")
        for k in range(len(Input_File_List)):
            file.write(Study_Name[k] + "\t%.4f\n" % Final_Pred[k])
    pred_sort_path=os.path.join(save_path,"Predict_sort.txt")
    os.system("sort -n -k 2 -r "+pred_path+" >"+pred_sort_path)











