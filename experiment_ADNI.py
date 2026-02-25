import copy
import json
import os
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.lib.function_base import flip
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
from torch.cuda import amp
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from torchvision.utils import make_grid, save_image

# from config import *
# import sys
# sys.path.insert(0, '/storage/Ayantika/Diffusion_AE_medical_cond')
from config_ADNI import *
from dataset import *
from dist_utils import *
from lmdb_writer import *
from metrics_cond import *
from renderer_cond import *
from tqdm import tqdm
from omegaconf import OmegaConf
import cv2
from skimage.segmentation import chan_vese


# +

from torch import nn
from torchvision.models import resnet18,vgg19
import numpy as np
import torch
from torchvision.models.resnet import ResNet18_Weights

def save_hook(module, input, output):
    setattr(module, 'output', output)   
##### Latentshift predictor changed
class LatentShiftPredictor(nn.Module):
    def __init__(self, dim, in_channel = 3,downsample=None):
        super(LatentShiftPredictor, self).__init__()
        self.features_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
#         self.features_extractor.conv1 = nn.Conv2d(
#             in_channel, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
#                                 mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample
        

#         half dimension as we expect the model to be symmetric
#         self.type_estimator = nn.Linear(512, np.product(dim))
        self.type_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),  # Non-linear activation
            nn.Linear(256, 64),
            nn.ReLU(),  # Non-linear activation
            nn.Linear(64, np.product(dim))  # Output layer
        )
#         self.shift_estimator = nn.Linear(512, 1)
        self.shift_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),  # Non-linear activation
            nn.Linear(256, 64),
            nn.ReLU(),  # Non-linear activation
            nn.Linear(64, 16),  # Output layer
            nn.ReLU(),  # Non-linear activation
            nn.Linear(16, 1)  # Output layer
        )

    def forward(self, x1):
        batch_size = x1.shape[0]
        self.features_extractor(x1)
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)
 
        return logits,shift.squeeze()

class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())
        #load weights for the pre-trained LatentShiftPredictor
        latent_shift_predictor = LatentShiftPredictor( dim=12, in_channel = 3,downsample=None)
        latent_shift_predictor.load_state_dict(torch.load('/home/projects/medimg/ayantika/Ayantika/results/pre_train_resnet_age_gap_prediction/latent_shift_predictor_32.pt'))
        latent_shift_predictor.cuda()
        latent_shift_predictor.eval()
        self.latent_shift_predictor = latent_shift_predictor
        self.conf = conf
        data_config = OmegaConf.load(self.conf.data_config_path)
        self.conf.img_size_height = data_config.dataloader.img_height
        self.conf.img_size_width =  data_config.dataloader.img_width
        

        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
#         print("Litmodel init csv path",self.csv_path)

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()
        

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf(
            ).make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf(
            ).make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # initial variables for consistent sampling
        self.register_buffer(
            'x_T',
            torch.randn(conf.sample_size, 1, conf.img_size_height, conf.img_size_width))

        if conf.pretrain is not None:
            print(f'loading pretrain ... {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print('step:', state['global_step'])
            self.load_state_dict(state['state_dict'], strict=False)

        if conf.latent_infer_path is not None:
            print('loading latent stats ...')
            state = torch.load(conf.latent_infer_path)
            self.conds = state['conds']
            self.register_buffer('conds_mean', state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

        noise = torch.randn(N,
                            1,
                            self.conf.img_size_height,
                            self.conf.img_size_width,
                            device=device)
        pred_img = render_uncondition(
            self.conf,
            self.ema_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.conds_mean,
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

#     def render(self, noise, cond=None, x_start_baseline=None,age_diff=None,health_state=None,T=None,cond_shift_weight=0):
#         if T is None:
#             sampler = self.eval_sampler
#         else:
#             sampler = self.conf._make_diffusion_conf(T).make_sampler()
#         sampler.cond_shift_weight= cond_shift_weight
#         if cond is not None:
#             pred_img = render_condition(self.conf,
#                                         self.ema_model,
#                                         noise,
#                                         x_start_baseline = x_start_baseline,
#                                         age_diff = age_diff,
#                                         health_state = health_state,
#                                         sampler = sampler,
#                                         cond = cond)
#         else:
#             pred_img = render_uncondition(self.conf,
#                                           self.ema_model,
#                                           noise,
#                                           sampler=sampler,
#                                           latent_sampler=None)
#         pred_img = (pred_img + 1) / 2
#         return pred_img
    
    def render(self, noise, cond=None, T=None,mask_mult=False,gt=None,gt_keep_mask=None,health_state = None):
        
#         ,\
#            mask_mult=True,\
#          gt=bsln.to(device),\
#          gt_keep_mask= ventricle_mask_batch)
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        ###########################
        sampler.mask_mult = mask_mult
        sampler.gt=gt
        sampler.gt_keep_mask=gt_keep_mask
        ###################
        
        if cond is not None:
            pred_img = render_condition(self.conf,
                                        self.ema_model,
                                        noise,
                                        sampler=sampler,
                                        cond=cond,
                                        health_state = health_state
                                       )
        else:
            pred_img = render_uncondition(self.conf,
                                          self.ema_model,
                                          noise,
                                          sampler=sampler,
                                          latent_sampler=None)
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x):
        # TODO:
        assert self.conf.model_type.has_autoenc()
        cond = self.ema_model.encoder.forward(x)
        return cond

    def encode_stochastic(self, x, cond, age_diff, health_state, x_start_baseline, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_model,
                                               x,
                                               age_diff=age_diff,
                                               health_state = health_state,
                                               x_start_baseline = x_start_baseline,
                                               model_kwargs={'cond': cond})
        return out['sample']

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            if ema_model:
                model = self.ema_model
            else:
                model = self.model
            gen = self.eval_sampler.sample(model=model,
                                           noise=noise,
                                           x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################
#         print(f"batch size {self.conf.batch_size}")
#         csv_path: str = ''
#         csv_path_test: str = ''
#         path: str = ''



#         make_dataset(self, path = ' ',csv_path = ' ', h5_save_path = '', csv_file_name = '', csv_mask_name = '' , ventricle_mask_root_path = '', mode_ = '' , **kwargs)



        self.train_data = self.conf.make_dataset(path = self.conf.data_config_path,\
                                                csv_path = self.conf.csv_path ,\
                                                h5_save_path = self.conf.h5_save_path_train,\
                                                csv_file_name = self.conf.csv_file_name_train, \
                                                csv_mask_name = self.conf.csv_mask_name_train , \
                                                ventricle_mask_root_path = self.conf.ventricle_mask_root_path,\
                                                mode_ = self.conf.mode_train)
#         self.train_data = SubsetDataset(self.train_data, size=81650)
#         self.train_data = SubsetDataset(self.train_data, size=860)
        
        print('train data:', len(self.train_data))
        
        self.val_data = self.conf.make_dataset(path = self.conf.data_config_path,\
                                                csv_path =self.conf.csv_path_test ,\
                                                h5_save_path = self.conf.h5_save_path_test,\
                                                csv_file_name = self.conf.csv_file_name_test, \
                                                csv_mask_name = self.conf.csv_mask_name_test , \
                                                ventricle_mask_root_path = self.conf.ventricle_mask_root_path,\
                                                mode_ = self.conf.mode_test)
#         self.val_data = SubsetDataset(self.val_data, size=1000)
        self.val_data = SubsetDataset(self.val_data, size=800)
        print('val data:', len(self.val_data))

    def _train_dataloader(self, drop_last=True):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataloader = conf.make_loader(self.train_data,
                                      shuffle=True,
                                      drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        if latent mode => return the inferred latent dataset
        """
        print('on train dataloader start ...')
        if self.conf.train_mode.require_dataset_infer():
            if self.conds is None:
                # usually we load self.conds from a file
                # so we do not need to do this again!
                self.conds = self.infer_whole_dataset()
                # need to use float32! unless the mean & std will be off!
                # (1, c)
                self.conds_mean.data = self.conds.float().mean(dim=0,
                                                               keepdim=True)
                self.conds_std.data = self.conds.float().std(dim=0,
                                                             keepdim=True)
            print('mean:', self.conds_mean.mean(), 'std:',
                  self.conds_std.mean())

            # return the dataset with pre-calculated conds
            conf = self.conf.clone()
            conf.batch_size = self.batch_size
            data = TensorDataset(self.conds)
            return conf.make_loader(data, shuffle=True)
        else:
            return self._train_dataloader()

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0
    
    
    def age_vector(self,age):
        age_ranges = [(55, 65), (65, 75), (75, 85), (85, float('inf'))]
        label_vectors = torch.eye(4)
        label = torch.zeros(4)
        for idx, (lower, upper) in enumerate(age_ranges):
            if lower <= age < upper:
                label = label_vectors[idx]
                break

        return label
    
    def slice_vector(self,slicenum):
        slice_ranges = [(60, 70), (70, 80), (80, 90), (90, 100), (100, 110)]
        label_vectors = torch.eye(5)
        label = torch.zeros(5)
        for idx, (lower, upper) in enumerate(slice_ranges):
            if lower <= slicenum < upper:
                label = label_vectors[idx]
                break

        return label
    
    ######
    def age_gap_vectors(self,age_gap):
        age_gap_ranges = [(0, 0.5), (0.5, 1),\
                        (1, 1.5), (1.5, 2), \
                        (2, 2.5), (2.5, 3),\
                       (3, 3.5), (3.5, 4),(4, 4.5)]
        label_vectors = torch.eye(9)
        label = torch.zeros(9)
        for idx, (lower, upper) in enumerate(age_gap_ranges):
            if lower < age_gap <= upper:
                label = label_vectors[idx]
                break

        return label
    ########
    
    def str_list_tensor(self, age_):
        if type(age_ == list):
            if type(age_[0]) == str:
                return torch.tensor([torch.tensor(float(age)) for age in age_]) 
            else:
                return print('list element not str')
        else:
            return print('list of str')
    
    def get_data_elements(self,batch,age_diff,target_indices):
        health_encoding ={'AD': [1, 0, 0], 'CN': [0, 1, 0], 'MCI': [0, 0, 1]}
#         batch_dict = {'DDIM_reverse':[], 'DiffAE_pred':[], 'cond':[]}
        if '_train_image' in batch.keys():
            mode_ = '_train_image'
        elif '_test_image' in batch.keys():
            mode_ = '_test_image'
        for ii in range(0,batch[mode_].shape[0]):
            
            ###
            if age_diff[ii]!=0:
                target_indices[ii][3] = torch.tensor(1)

#             target_indices[ii][0:4] = age_vector(age=batch['Age'][ii])
            target_indices[ii][0:3] = torch.tensor(health_encoding[batch['Health status'][ii]])
            target_indices[ii][3:3+9] = self.age_gap_vectors((self.str_list_tensor(batch['Age']) - self.str_list_tensor(batch['baseline Age']))[ii])

#             target_indices[ii][4:8] = self.age_vector(age=batch['Age'][ii])
#             target_indices[ii][4:9] = self.slice_vector(slicenum=batch['slicenum'][ii])


    #         print(dict_['cond'].shape)

#         target_indices.cuda()
        target_indices = target_indices.to(batch[mode_].dtype)
        target_indices = target_indices.to(batch[mode_].device)
        shifts = age_diff
#         shifts.cuda()
        basis_shift = target_indices.clone()
#         basis_shift[:,5] = shifts
#         basis_shift[:,3] = shifts
        
        return target_indices,shifts,basis_shift
    
    def get_ventricle(self,img_):
        cv = chan_vese(img_, mu=0.2, lambda1=1, lambda2=1, tol=1e-3,
                       max_num_iter=200, dt=0.5, init_level_set="checkerboard",
                       extended_output=True)

        img_mask = (img_>0)
        kernel = np.ones((11, 11), np.uint8) 
        eroded_image = cv2.erode(img_mask.astype(np.uint8), kernel, iterations=1)
        ventricle_mask = np.logical_not(cv[0]) * eroded_image
        return ventricle_mask,cv
    def normalise_(self,img):
        return ((img- img.min())/(img.max() - img.min()))

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
#         print("current Epoch",self.trainer.current_epoch )
#         print("current sample no",self.num_samples )
        with amp.autocast(False):
            if '_train_image' in batch.keys():
                mode_ = '_train'
            elif '_test_image' in batch.keys():
                mode_ = '_test'

            ventricle_mask_batch = batch['ventricle_mask']

            x_start = batch['_train_image']
            x_start_baseline = batch[mode_+'_baseline_image']
            idxs = batch['idx']
            age_diff = self.str_list_tensor(batch['Age']) - self.str_list_tensor(batch['baseline Age'])
            age_diff = age_diff.to(batch[mode_+'_image'].dtype)
            age_diff = age_diff.to(x_start.device)

            cond_vector = torch.zeros(batch[mode_+'_image'].shape[0],12)
                
            cond_vector,shifts,cond_vector_shift = self.get_data_elements(batch,age_diff,cond_vector)
#             print("cond_vector,shifts,cond_vector_shift",cond_vector.device,shifts.device,cond_vector_shift.device)
#             print("x_start",x_start.device)

            # Convert the list of tensors into a single tensor
#             health_state = torch.stack(encoded_tensors)
            health_state = cond_vector_shift
            cond_vector_shift = cond_vector_shift.to(age_diff.dtype)
            cond_vector_shift = cond_vector_shift.to(age_diff.device)  
            cond_vector = cond_vector.to(age_diff.dtype)
            cond_vector = cond_vector.to(age_diff.device)  
            
            
            x_start = batch[mode_+'_image'].cuda()
            if self.conf.train_mode == TrainMode.diffusion:
                """
                main training mode!!!
                """
                # with numpy seed we have the problem that the sample t's are related!
                t, weight = self.T_sampler.sample(len(x_start), x_start.device)
                ###################### In the training phase this weightage is activated and deactiveated for two types of output
#                 self.sampler.cond_shift_weight= 1
#                 ###################### removing the ventricular_mask until its ready
                self.sampler.ventricle_mask_batch = ventricle_mask_batch
#                 print("(self.trainer.current_epoch",self.trainer.current_epoch)
                self.sampler.age_shift  = shifts.to(batch[mode_+'_image'].dtype).to(batch[mode_+'_image'].device)
                if (self.trainer.current_epoch<=50) or ((self.trainer.current_epoch%10)==0):
                    self.sampler.cond_shift_weight= 0
                    x_start = x_start_baseline
                else:
                    self.sampler.cond_shift_weight= 1
                losses = self.sampler.training_losses(model=self.model,
                                                      x_start=x_start,
                                                      x_start_baseline=x_start_baseline,
                                                      age_diff=age_diff,
                                                      health_state = cond_vector_shift,
                                                      cond_vector = cond_vector,
                                                      latent_shift_predictor = self.latent_shift_predictor,
                                                      t=t)
#                 print('losses', losses)
#                 "epsilon","cond_pre""cond_post" pred_xstart
            elif self.conf.train_mode.is_latent_diffusion():
                """
                training the latent variables!
                """
                # diffusion on the latent
                t, weight = self.T_sampler.sample(len(cond), cond.device)
                latent_losses = self.latent_sampler.training_losses(
                    model=self.model.latent_net, x_start=cond, t=t)
                # train only do the latent diffusion
                losses = {
                    'latent': latent_losses['loss'],
                    'loss': latent_losses['loss']
                }
            else:
                raise NotImplementedError()

            loss = losses['loss'].mean()
            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt','latent_code_mse','mse','sparsity_cons',\
                        'loss_cross_attention_info_max','loss_cross_attention_alignment']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()
            
            

            if self.global_rank == 0 and (self.num_samples/self.conf.save_every_samples).is_integer():
                
                ############### log images 
#                 grid_batch = (make_grid(batch['pre_op_image'][0:self.conf.batch_size_eval,:,:,:]) + 1) / 2
                grid_batch = (make_grid(batch[mode_+'_image'][0:self.conf.batch_size_eval,:,:,:]) + 1) / 2
                self.logger.experiment.add_image(
                    f'input_pre', grid_batch,
                    self.trainer.current_epoch)
                
                grid_epsilon = (make_grid(losses['epsilon'][0:self.conf.batch_size_eval,:,:,:]) + 1) / 2
                self.logger.experiment.add_image(
                    f'epsilon', grid_epsilon,
                    self.trainer.current_epoch)
                
                grid_xstart = (make_grid(losses['pred_xstart'][0:self.conf.batch_size_eval,:,:,:]) + 1) / 2
                self.logger.experiment.add_image(
                    f'pred_xstart', grid_xstart,
                    self.trainer.current_epoch)
                #################
                

            if self.global_rank == 0:
                ################# Logging losses ##############   
                self.logger.experiment.add_scalar('loss', losses['loss'],
                                                  self.trainer.current_epoch)
                for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt','latent_code_mse',\
                            'mse','sparsity_cons','loss_cross_attention_info_max','loss_cross_attention_alignment']:
                    if key in losses:
#                         print("losses[key]",losses[key],key)
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.num_samples)

        return {'loss': loss}
    
    
    #     def on_train_batch_end(self, batch, batch_idx: int,
#                        dataloader_idx: int) -> None:
    def on_train_batch_end(self, outputs, batch, batch_idx: int
                          ) -> None:
        """
        after each training step ...
        """
#         print("batch_idx",batch_idx)
#         print("self.is_last_accum(batch_idx)", self.is_last_accum(batch_idx))
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(self.model.latent_net, self.ema_model.latent_net,
                    self.conf.ema_decay)
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

            # logging
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            else:
                ###### get ventricular mask #######
                ######### stil being generated when completed will replace
                ###### get ventricular mask #######
#                 ventricular_mask_ = []
#                 for iii in range(0,batch['_train_image'].shape[0]):
#                     sub = batch['Subject'][iii]
#                     age = batch['Age'][iii]
#                     slice_num = batch['slicenum'][iii]
#                     store_name = sub+'_'+str(round(float(age),2))+'_'+str(int(slice_num))
#                     ventricular_mask = torch.load('/storage/Ayantika/Data_final/ADNI_ventricle_mask/'+store_name+'.pt')
# #                     print(ventricular_mask.shape)
#                     ventricular_mask_.append(ventricular_mask)

#                 ventricular_mask_batch_ = torch.cat(ventricular_mask_)
                ventricle_mask_batch = batch['ventricle_mask']
#     ventricular_mask_batch_.to(torch.float32).to(batch['_train_image'].device)
                if '_train_image' in batch.keys():
                    mode_ = '_train'
                elif '_test_image' in batch.keys():
                    mode_ = '_test'
                ##################################
                x_start = batch[mode_+'_image']
                x_start_baseline = batch[mode_+'_baseline_image']
                idxs = batch['idx']
                age_diff = (self.str_list_tensor(batch['Age']) - self.str_list_tensor(batch['baseline Age']))
                age_diff = age_diff.to(x_start.device)
                ###### for health state vector
                labels = batch['Health status']

                label_encoding = {'AD': [1, 0, 0], 'CN': [0, 1, 0], 'MCI': [0, 0, 1]}

                # Create tensors with label encodings
                target_indices = torch.zeros(batch[mode_+'_image'].shape[0],12)
                
                target_indices,shifts,basis_shift = self.get_data_elements(batch,age_diff,target_indices)

                # Convert the list of tensors into a single tensor
                #             health_state = torch.stack(encoded_tensors)
                health_state = basis_shift
                health_state = health_state.to(age_diff.dtype)
                health_state = health_state.to(age_diff.device)
                
            self.log_sample(x_start=x_start,x_start_baseline=x_start_baseline,\
                            age_diff=age_diff,health_state=health_state,\
                           ventricle_mask = ventricle_mask_batch)
            self.evaluate_scores()

#     def on_before_optimizer_step(self, optimizer: Optimizer,
#                                  optimizer_idx: int) -> None:
#         # fix the fp16 + clip grad norm problem with pytorch lightinng
#         # this is the currently correct way to do it
#         if self.conf.grad_clip > 0:
#             # from trainer.params_grads import grads_norm, iter_opt_params
#             params = [
#                 p for group in optimizer.param_groups for p in group['params']
#             ]
#             # print('before:', grads_norm(iter_opt_params(optimizer)))
#             torch.nn.utils.clip_grad_norm_(params,
#                                            max_norm=self.conf.grad_clip)
#             # print('after:', grads_norm(iter_opt_params(optimizer)))

    def log_sample(self, x_start=None,x_start_baseline=None,age_diff=None,health_state=None,ventricle_mask=None):
        """
        put images to the tensorboard
        """
        def do(model,
               postfix,
               use_xstart,
               save_real=False,
               no_latent_diff=False,
               interpolate=False):
            model.eval()
            with torch.no_grad():
                
                all_x_T = self.split_tensor(self.x_T)
                batch_size = min(len(all_x_T), self.conf.batch_size_eval)
                # allow for superlarge models
#                 print("x_T shape",self.x_T.shape)
#                 print("self.conf.batch_size_eval",self.conf.batch_size_eval)
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen_followup = []
#                 Baseline = []
#                 Org_followup = []
                
                for x_T in loader:
                    if use_xstart:
                        _xstart = x_start[:len(x_T)]
                        _xstart_baseline = x_start_baseline[:len(x_T)]
                        _age_diff = age_diff[:len(x_T)]
                        _health_state = health_state[:len(x_T)]
                        if ventricle_mask is not None:
                            _ventricle_mask = ventricle_mask[:len(x_T)]
                    else:
                        _xstart = None

                    if self.conf.train_mode.is_latent_diffusion(
                    ) and not use_xstart:
                        # diffusion of the latent first
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.eval_sampler,
                            latent_sampler=self.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std)
                        
                    else:
#                         print("not inside self.conf.train_mode.is_latent_diffusion", self.conf.train_mode.is_latent_diffusion())
#                         if not use_xstart and self.conf.model_type.has_noise_to_cond(
#                         ):
#                             model: BeatGANsAutoencModel
#                             # special case, it may not be stochastic, yet can sample
#                             cond = torch.randn(len(x_T),
#                                                self.conf.style_ch,
#                                                device=self.device)
#                             cond = model.noise_to_cond(cond)
#                         else:
# #                             if interpolate:
# #                                 with amp.autocast(self.conf.fp16):
# #                                     cond = model.encoder(_xstart)
# #                                     i = torch.randperm(len(cond))
# #                                     cond = (cond + cond[i]) / 2
# #                             else:
                        cond = model.encoder(_xstart_baseline)
#                         baseline, org followup, gen followup, org followup - gen followup,  baseline - gen followup
#                         if (self.trainer.current_epoch<=10) or ((self.trainer.current_epoch%10)==0):
#                             self.eval_sampler.cond_shift_weight= 0
#                             _xstart = _xstart_baseline
#                         else:
                        if (self.trainer.current_epoch<=50) or ((self.trainer.current_epoch%10)==0):
                            self.eval_sampler.cond_shift_weight= 0
                            _xstart = _xstart_baseline
                        else:
                            self.eval_sampler.cond_shift_weight= 1
                        if ventricle_mask is not None:
                            self.eval_sampler.mask_mult = False
#                             self.eval_sampler.gt_keep_mask = _ventricle_mask
#                             self.eval_sampler.gt = _xstart_baseline

                        gen = self.eval_sampler.sample(model=model,
                                                       noise=x_T,
                                                       cond=cond,
                                                       x_start_baseline = _xstart_baseline,
                                                       age_diff = _age_diff,
                                                       health_state = _health_state,
                                                       x_start = _xstart)
#                         print("gen shape inside logger for ",gen.shape)
#                         print("_xstart_baseline shape inside logger for ",_xstart_baseline.shape)
#                         print("_xstart shape inside logger for ",_xstart.shape)
                        
                        
                    Gen_followup.append(gen)
                    

                gen = torch.cat(Gen_followup)
#                 print("gen shape inside logger",gen.shape)
                
                gen = self.all_gather(gen)
                if gen.dim() == 5:
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart:
                    # save the original images to the tensorboard
                    org_followup = self.all_gather(_xstart)
                    if org_followup.dim() == 5:
                        org_followup = org_followup.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_org_followup = (make_grid(org_followup)+ 1) / 2
                        self.logger.experiment.add_image(
                            f'sample{postfix}/org_followup', grid_org_followup,
                            self.trainer.current_epoch)
                        
                    baseline = self.all_gather(_xstart_baseline)
                    if baseline.dim() == 5:
                        baseline = baseline.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_baseline = (make_grid(baseline)+ 1) / 2
                        self.logger.experiment.add_image(
                            f'sample{postfix}/baseline', grid_baseline,
                            self.trainer.current_epoch)

                if self.global_rank == 0:
                    # save samples to the tensorboard
                    grid = (make_grid(gen) + 1) / 2

                    grid_res_gen_org_followup = (make_grid(gen-org_followup) + 1) / 2
                    grid_res_gen_baseline = (make_grid(gen-baseline) + 1) / 2
                    
                    
                    sample_dir = os.path.join(self.conf.logdir,
                                              f'sample{postfix}_gen_followup')
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    path = os.path.join(sample_dir,
                                        '%d.png' % self.trainer.current_epoch)
                    save_image(grid, path)
                    self.logger.experiment.add_image(f'sample{postfix}/gen_followup', grid,
                                                     self.trainer.current_epoch)
                    self.logger.experiment.add_image(f'sample{postfix}/res_gen_org_followup', grid_res_gen_org_followup,
                                                     self.trainer.current_epoch)
                    self.logger.experiment.add_image(f'sample{postfix}/res_gen_baseline', grid_res_gen_baseline,
                                                     self.trainer.current_epoch)
            model.train()

        if self.conf.sample_every_samples > 0 and is_time(
                self.num_samples, self.conf.sample_every_samples,
                self.conf.batch_size_effective):

            if self.conf.train_mode.require_dataset_infer():
                do(self.model, '', use_xstart=False)
                do(self.ema_model, '_ema', use_xstart=False)
            else:
                
                if self.conf.model_type.has_autoenc(
                ) and self.conf.model_type.can_sample():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                    
                elif self.conf.train_mode.use_latent_net():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.model,
                       '_enc_nodiff',
                       use_xstart=True,
                       save_real=True,
                       no_latent_diff=True)
                    do(self.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                else:
                    do(self.model, '', use_xstart=True, save_real=True)
                    do(self.ema_model, '_ema', use_xstart=True, save_real=True)

    def evaluate_scores(self):
        """
        evaluate FID and other scores during training (put to the tensorboard)
        For, FID. It is a fast version with 5k images (gold standard is 50k).
        Don't use its results in the paper!
        """
        def fid(model, postfix):
            score = evaluate_fid(self.eval_sampler,
                                 model,
                                 self.conf,
                                 device=self.device,
                                 train_data=self.train_data,
                                 val_data=self.val_data,
                                 latent_sampler=self.eval_latent_sampler,
                                 conds_mean=self.conds_mean,
                                 conds_std=self.conds_std,
                                epoch = self.trainer.current_epoch,
                                )
            if self.global_rank == 0:
                self.logger.experiment.add_scalar(f'FID{postfix}', score,
                                                  self.num_samples)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, 'eval.txt'),
                          'a') as f:
                    metrics = {
                        f'FID{postfix}': score,
                        'num_samples': self.num_samples,
                    }
                    f.write(json.dumps(metrics) + "\n")

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc(
            ) and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(self.eval_sampler,
                                       model,
                                       self.conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=self.eval_latent_sampler,
                                       epoch = self.trainer.current_epoch,
                                )

                if self.global_rank == 0:
                    for key, val in score.items():
                        self.logger.experiment.add_scalar(
                            f'{key}{postfix}', val, self.num_samples)

#         if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(
#                 self.num_samples, self.conf.eval_every_samples,
#                 self.conf.batch_size_effective):
        if self.conf.sample_every_samples > 0 and is_time(
                self.num_samples, self.conf.sample_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid @ {self.num_samples}')
            lpips(self.model, '')
#             fid(self.model, '')

#         if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(
#                 self.num_samples, self.conf.eval_ema_every_samples,
#                 self.conf.batch_size_effective):
#         if self.conf.sample_every_samples > 0 and is_time(
#                 self.num_samples, self.conf.sample_every_samples,
#                 self.conf.batch_size_effective):
#             print(f'eval fid ema @ {self.num_samples}')
#             fid(self.ema_model, '_ema')
            # it's too slow
            # lpips(self.ema_model, '_ema')

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out['optimizer'] = optim
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                      lr_lambda=WarmupLR(
                                                          self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        for the "eval" mode. 
        We first select what to do according to the "conf.eval_programs". 
        test_step will only run for "one iteration" (it's a hack!).
        
        We just want the multi-gpu support. 
        """
        # make sure you seed each worker differently!
        self.setup()

        # it will run only one step!
        print('global step:', self.global_step)
        """
        "infer" = predict the latent variables using the encoder on the whole dataset
        """
        if 'infer' in self.conf.eval_programs:
            if 'infer' in self.conf.eval_programs:
                print('infer ...')
                conds = self.infer_whole_dataset().float()
                # NOTE: always use this path for the latent.pkl files
                save_path = f'checkpoints/{self.conf.name}/latent.pkl'
            else:
                raise NotImplementedError()

            if self.global_rank == 0:
                conds_mean = conds.mean(dim=0)
                conds_std = conds.std(dim=0)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(
                    {
                        'conds': conds,
                        'conds_mean': conds_mean,
                        'conds_std': conds_std,
                    }, save_path)
        """
        "infer+render" = predict the latent variables using the encoder on the whole dataset
        THIS ALSO GENERATE CORRESPONDING IMAGES
        """
        # infer + reconstruction quality of the input
        for each in self.conf.eval_programs:
            if each.startswith('infer+render'):
                m = re.match(r'infer\+render([0-9]+)', each)
                if m is not None:
                    T = int(m[1])
                    self.setup()
                    print(f'infer + reconstruction T{T} ...')
                    conds = self.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=
                        f'latent_infer_render{T}/{self.conf.name}.lmdb',
                    )
                    save_path = f'latent_infer_render{T}/{self.conf.name}.pkl'
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    torch.save(
                        {
                            'conds': conds,
                            'conds_mean': conds_mean,
                            'conds_std': conds_std,
                        }, save_path)

        # evals those "fidXX"
        """
        "fid<T>" = unconditional generation (conf.train_mode = diffusion).
            Note:   Diff. autoenc will still receive real images in this mode.
        "fid<T>,<T_latent>" = unconditional generation for latent models (conf.train_mode = latent_diffusion).
            Note:   Diff. autoenc will still NOT receive real images in this made.
                    but you need to make sure that the train_mode is latent_diffusion.
        """
        for each in self.conf.eval_programs:
            if each.startswith('fid'):
                m = re.match(r'fid\(([0-9]+),([0-9]+)\)', each)
                clip_latent_noise = False
                if m is not None:
                    # eval(T1,T2)
                    T = int(m[1])
                    T_latent = int(m[2])
                    print(f'evaluating FID T = {T}... latent T = {T_latent}')
                else:
                    m = re.match(r'fidclip\(([0-9]+),([0-9]+)\)', each)
                    if m is not None:
                        # fidclip(T1,T2)
                        T = int(m[1])
                        T_latent = int(m[2])
                        clip_latent_noise = True
                        print(
                            f'evaluating FID (clip latent noise) T = {T}... latent T = {T_latent}'
                        )
                    else:
                        # evalT
                        _, T = each.split('fid')
                        T = int(T)
                        T_latent = None
                        print(f'evaluating FID T = {T}...')

                self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                if T_latent is not None:
                    latent_sampler = self.conf._make_latent_diffusion_conf(
                        T=T_latent).make_sampler()
                else:
                    latent_sampler = None

                conf = self.conf.clone()
                conf.eval_num_images = 500
                print(conf.img_size)
                score = evaluate_fid(T,
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    latent_sampler=latent_sampler,
                    conds_mean=self.conds_mean,
                    conds_std=self.conds_std,
                    remove_cache=False,
                    clip_latent_noise=clip_latent_noise,
                )
                if T_latent is None:
                    self.log(f'fid_ema_T{T}', score)
                else:
                    name = 'fid'
                    if clip_latent_noise:
                        name += '_clip'
                    name += f'_ema_T{T}_Tlatent{T_latent}'
                    self.log(name, score)
        """
        "recon<T>" = reconstruction & autoencoding (without noise inversion)
        """
        for each in self.conf.eval_programs:
            if each.startswith('recon'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('recon')
                T = int(T)
                print(f'evaluating reconstruction T = {T}...')

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=None)
                for k, v in score.items():
                    self.log(f'{k}_ema_T{T}', v)
        """
        "inv<T>" = reconstruction with noise inversion
        """
        for each in self.conf.eval_programs:
            if each.startswith('inv'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('inv')
                T = int(T)
                print(
                    f'evaluating reconstruction with noise inversion T = {T}...'
                )

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=None,
                                       use_inverted_noise=True)
                for k, v in score.items():
                    self.log(f'{k}_inv_ema_T{T}', v)
# -



def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
    print('conf:', conf.name)
    # assert not (conf.fp16 and conf.grad_clip > 0
    #             ), 'pytorch lightning has bug with amp + gradient clipping'
    model = LitModel(conf)
    
    #############################
    # saving the checkpoints and sample images after one epoch is completed #
    model.setup()
    conf.save_every_samples = len(model.train_data)
    conf.sample_every_samples = conf.save_every_samples
    #############################

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=-1)
#                                  every_n_epochs = 5)
#                                  every_n_train_steps = conf.save_every_samples)
#     //
#                                  conf.batch_size_effective)
    checkpoint_path = f'{conf.logdir}/last.ckpt'
#     checkpoint_path = '/storage/Ayantika/results/Diff_AE_xstart_w_xbsln_disentangle_unsup/ADNI_AD_CN_MCI/last.ckpt'
    print('ckpt path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print('resume!')
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    # from pytorch_lightning.

    plugins = []
    if len(gpus) == 1 and nodes == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin

        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))
#     accelerator='gpu', devices=[0]

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
#         resume_from_checkpoint=resume,
        devices=gpus,
        num_nodes=nodes,
        accelerator='gpu',
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        # clip in the model instead
        # gradient_clip_val=conf.grad_clip,
        use_distributed_sampler =True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )
    
    if mode == 'train':
#         ,ckpt_path=resume
#         checkpoint_path = '/storage/Ayantika/results/Diff_AE_xstart_w_xbsln_disentangle_unsup/ADNI_AD_CN_MCI/last.ckpt'
#         state_dict = torch.load(checkpoint_path)
#         model.load_state_dict(state_dict['state_dict'], strict=False)
        trainer.fit(model,ckpt_path=resume)
    elif mode == 'eval':
        # load the latest checkpoint
        # perform lpips
        # dummy loader to allow calling "test_step"
        dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
                           batch_size=conf.batch_size)
        eval_path = conf.eval_path or checkpoint_path
        # conf.eval_num_images = 50
        print('loading from:', eval_path)
        state = torch.load(eval_path, map_location='cpu')
        print('step:', state['global_step'])
        model.load_state_dict(state['state_dict'])
        # trainer.fit(model)
        out = trainer.test(model, dataloaders=dummy)
        # first (and only) loader
        out = out[0]
        print(out)

        if get_rank() == 0:
            # save to tensorboard
            for k, v in out.items():
                tb_logger.experiment.add_scalar(
                    k, v, state['global_step'] * conf.batch_size_effective)

            # # save to file
            # # make it a dict of list
            # for k, v in out.items():
            #     out[k] = [v]
            tgt = f'evals/{conf.name}.txt'
            dirname = os.path.dirname(tgt)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(tgt, 'a') as f:
                f.write(json.dumps(out) + "\n")
            # pd.DataFrame(out).to_csv(tgt)
    else:
        raise NotImplementedError()










