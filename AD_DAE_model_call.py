### AD-DAE model
import argparse
import os
import torch
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from choices import *
from templates import *

def AD_DAE_model_call_func(root_path='/home/projects/medimg/ayantika/Ayantika/Diff_AE_xstart_w_xbsln_disentangle_unsup/AD_DAE_files/',\
                          checkpoint_path = '/home/projects/medimg/ayantika/Ayantika/results/AD_DAE_new_train/ADNI_AD_CN_MCI/',\
                          epoch_no = 120):
    
    h5_data_path = root_path+'/data_store/'
    region_info_path = root_path+'/data_store/'
    take_from_tmp = True
    conf = AD_DAE_autoenc_130M()
    data_config_path = './config_file_ADNI.yaml'
    config = OmegaConf.load(data_config_path)
    conf.batch_size = 10
    conf.data_name = 'ADNI'
    conf.img_size = config.dataloader.img_height
    conf.model_conf.image_size = config.dataloader.img_height
    conf.model_conf.in_channels = 2
    conf.model_conf.out_channels = 1
    conf.base_dir = './'
    conf.name = 'ADNI_AD_CN_MCI'
    conf.sample_size = 10
    conf.batch_size_eval = 10
    conf.beatgans_loss_type = LossType.mse
    conf.beatgans_model_mean_type = ModelMeanType.start_x
    conf.num_workers = 1
    conf.data_config_path = './config_file_ADNI.yaml'
    conf.img_size_height = 160
    conf.img_size_width = 208
    


    ##########
#     conf.csv_path = './dataset/ADNI_Data_loader_csv/ADNI_info_train_final_subset.csv'
    conf.csv_path_test = './dataset/ADNI_Data_loader_csv/ADNI_info_test_final.csv'
#     conf.h5_save_path_train = h5_data_path+'/ADNI_cond_train_ventricle_mask'
    conf.h5_save_path_test = h5_data_path+'/ADNI_cond_test_ventricle_mask'


    #     conf.eval_num_images = 80
    conf.eval_num_images = 1000


#     conf.csv_file_name_train = './dataset/ADNI_Data_loader_csv/ADNI_train_pair_data_info.csv'
#     conf.csv_mask_name_train ='./dataset/ADNI_Data_loader_csv/ADNI_train_pair_mask_info.csv'

    conf.csv_file_name_test = './dataset/ADNI_Data_loader_csv/ADNI_test_pair_data_info.csv'
    conf.csv_mask_name_test= './dataset/ADNI_Data_loader_csv/ADNI_test_pair_mask_info.csv'
       
    conf.ventricle_mask_root_path= region_info_path+'/ADNI_ventricle_mask/'


    conf.mode_train = 'train'
    conf.mode_test= 'test'
    conf.eval_num_images = 1000
    # checkpoint_path = '/home/projects/medimg/ayantika/Ayantika/results/AD_DAE_new_train/ADNI_AD_CN_MCI/epoch=169-step=320110.ckpt'


    epoch_no = epoch_no
    checkpoint_path = glob.glob(checkpoint_path+'epoch='+str(epoch_no)+'-**')[0]

    device = 'cuda'
    model = LitModel(conf)
    state = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)
    
    return model
