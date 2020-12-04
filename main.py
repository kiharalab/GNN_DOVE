from ops.argparser import argparser
import os

if __name__ == "__main__":
    params = argparser()
    #print(params)
    if params['mode']==0:
        input_path = os.path.abspath(params['F'])  # one pdb file
        os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
        from predict.predict_single_input import predict_single_input
        predict_single_input(input_path,params)

    elif params['mode']==1:
        input_path=os.path.abspath(params['F'])
        os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
        from predict.predict_multi_input import predict_multi_input

        predict_multi_input(input_path, params)

    elif params['mode']==2:
        input_path = os.path.abspath(params['F'])  # one pdb file
        os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
        from predict.visualize_attention import visualize_attention
        visualize_attention(input_path, params)



