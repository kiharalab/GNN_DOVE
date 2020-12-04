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

def init_model(model_path,params):
    model = GNN_Model(params)
    print('    Total params: %.10fM' % (count_parameters(model) / 1000000.0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model, device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    return model,device
def Get_Predictions(dataloader,device,model):
    Final_pred = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            H, A1, A2, V, Atom_count = sample
            batch_size = H.size(0)
            H, A1, A2, V = H.to(device), A1.to(device), A2.to(device), V.to(device)
            pred= model.test_model((H, A1, A2, V, Atom_count), device)
            pred1 = pred.detach().cpu().numpy()
            Final_pred += list(pred1)
    return Final_pred

def predict_single_input(input_path,params):
    #create saving path
    save_path=os.path.join(os.getcwd(),"Predict_Result")
    mkdir(save_path)
    save_path = os.path.join(save_path, "Single_Target")
    mkdir(save_path)
    save_path = os.path.join(save_path, "Fold_"+str(params['fold'])+"_Result")
    mkdir(save_path)
    input_path=os.path.abspath(input_path)
    split_name=os.path.split(input_path)[1]
    original_pdb_name=split_name
    if ".pdb" in split_name:
        split_name=split_name[:-4]
    save_path=os.path.join(save_path,split_name)
    mkdir(save_path)

    structure_path=os.path.join(save_path,"Input.pdb")
    shutil.copy(input_path,structure_path)
    input_file=Prepare_Input(structure_path)
    fold_choice = params['fold']
    #loading the model
    if fold_choice != -1:
        model_path = os.path.join(os.getcwd(), "best_model")
        model_path = os.path.join(model_path, "fold" + str(fold_choice))
        model_path=os.path.join(model_path,"checkpoint.pth.tar")
        model,device=init_model(model_path,params)
    else:
        root_model_path = os.path.join(os.getcwd(), "best_model")
        model_list=[]
        for k in range(1,4):
            model_path = os.path.join(root_model_path, "fold" + str(k))
            model_path = os.path.join(model_path,"checkpoint.pth.tar")
            model,device = init_model(model_path, params)
            model_list.append(model)
        model=model_list

    #loading data for predicition
    list_npz = [input_file]
    dataset = Single_Dataset(list_npz)
    dataloader = DataLoader(dataset, 1, shuffle=False,
                            num_workers=params['num_workers'],
                            drop_last=False, collate_fn=collate_fn)

    #prediction
    if fold_choice!=-1:
        Final_Pred=Get_Predictions(dataloader, device, model)
    else:
        Final_Pred=[]
        for cur_model in model:
            tmp_pred=Get_Predictions(dataloader, device, cur_model)
            Final_Pred.append(tmp_pred)
        Final_Pred=np.mean(Final_Pred,axis=0)
    #write the predictions
    pred_path=os.path.join(save_path,'Predict.txt')
    with open(pred_path,'w') as file:
        file.write("Input\tScore\n")
        file.write(original_pdb_name+"\t%.4f\n"%Final_Pred[0])



















