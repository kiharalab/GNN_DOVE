#
# Copyright (C) 2020 Xiao Wang
# Email:xiaowang20140001@gmail.com wang3702@purdue.edu
#

import parser
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-F',type=str, required=True,help='decoy example path')#File path for decoy dir
    parser.add_argument('--mode',type=int,required=True,help='0: predicting for single docking model 1: predicting and sorting for a list of docking models')
    parser.add_argument('--gpu',type=str,default='0',help='Choose gpu id, example: \'1,2\'(specify use gpu 1 and 2)')
    parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
    parser.add_argument("--num_workers", help="number of workers", type=int, default=4)
    parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default=4)
    parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default=140)
    parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default=4)
    parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default=128)
    parser.add_argument("--initial_mu", help="initial value of mu", type=float, default=0.0)
    parser.add_argument("--initial_dev", help="initial value of dev", type=float, default=1.0)
    parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default=0.3)
    parser.add_argument('--seed',type=int,default=888,help='random seed for shuffling')
    parser.add_argument('--fold',required=True,help='specify fold model for prediction',type=int,default=-1)
    args = parser.parse_args()
    params = vars(args)
    return params