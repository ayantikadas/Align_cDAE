from choices import *
from templates import *
import os

# region
# # !nvidia-smi
# endregion




root_path='/home/projects/medimg/ayantika/Ayantika/Data_final/'
checkpoint_path = '/home/projects/medimg/ayantika/Ayantika/results/AD_DAE_new_train/ADNI_AD_CN_MCI/'

h5_data_path = '/home/projects/medimg/ayantika/Ayantika/h5data_store/'
region_info_path = '/home/projects/medimg/ayantika/Ayantika/Data_final/'
gpus = [0]

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
conf.base_dir = '/home/projects/medimg/ayantika/Ayantika/results/Align_DiffAE'
if not os.path.exists(conf.base_dir):
    os.mkdir(conf.base_dir)
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
conf.csv_path = root_path+'/ADNI_Data_loader_csv/ADNI_info_train_final_subset.csv'
conf.csv_path_test = root_path+'/ADNI_Data_loader_csv/ADNI_info_test_final.csv'


conf.h5_save_path_train = '/home/projects/medimg/ADNI_data_loc/ADNI_cond_train_ventricle_mask/'
conf.h5_save_path_test = '/home/projects/medimg/ADNI_data_loc/ADNI_cond_test_ventricle_mask/'
# conf.h5_save_path_train = h5_data_path+'/ADNI_cond_train_ventricle_mask'
# conf.h5_save_path_test = h5_data_path+'/ADNI_cond_test_ventricle_mask'


conf.eval_num_images = 80
conf.eval_num_images = 100


conf.csv_file_name_train = root_path+'/ADNI_Data_loader_csv/ADNI_train_pair_data_info.csv'
conf.csv_mask_name_train =root_path+'/ADNI_Data_loader_csv/ADNI_train_pair_mask_info.csv'

conf.csv_file_name_test = root_path+'/ADNI_Data_loader_csv/ADNI_test_pair_data_info.csv'
conf.csv_mask_name_test= root_path+'/ADNI_Data_loader_csv/ADNI_test_pair_mask_info.csv'

conf.ventricle_mask_root_path= '/home/projects/medimg/ADNI_data_loc/ADNI_ventricle_mask/'
# region _info_path+'/ADNI_ventricle_mask/'


conf.mode_train = 'train'
conf.mode_test= 'test'
conf.eval_num_images = 500
# checkpoint_path = '/home/projects/medimg/ayantika/Ayantika/results/AD_DAE_new_train/ADNI_AD_CN_MCI/epoch=169-step=320110.ckpt'


# epoch_no = 120
# checkpoint_path = glob.glob('/home/projects/medimg/ayantika/Ayantika/results/AD_DAE_new_train/ADNI_AD_CN_MCI/epoch='+str(epoch_no)+'-**')[0]
# device = 'cuda'
# model = LitModel(conf)
# state = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))
# model.load_state_dict(state['state_dict'], strict=False)
# model.ema_model.eval()
# model.ema_model.to(device)

# return model

# -

train(conf, gpus=gpus)

# # +
# # !nvidia-smi
# -



# # +
# conf.logdir

# # +
# # !nvidia-smi
# -
# endregion
