from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .unet import *
from choices import *


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import numpy as np


def save_hook(module, input, output):
    setattr(module, 'output', output)

# ################ Decoding the conditioner ####################
# class Latent_decode(nn.Module):
#     def __init__(self,input_dim=50, out_dim=12, inner_dim=25,
#                  random_init=False, bias=True):
#         super(Latent_decode, self).__init__()
#         self.input_dim = input_dim 
#         self.out_dim = out_dim 

#         self.fc1 = nn.Linear(self.input_dim, inner_dim)
#         self.bn1 = nn.BatchNorm1d(inner_dim)
#         self.act1 = nn.ELU()

#         self.fc2 = nn.Linear(inner_dim, inner_dim)
#         self.bn2 = nn.BatchNorm1d(inner_dim)
#         self.act2 = nn.ELU()

#         self.fc3 = nn.Linear(inner_dim, inner_dim)
#         self.bn3 = nn.BatchNorm1d(inner_dim)
#         self.act3 = nn.ELU()

#         self.fc4 = nn.Linear(inner_dim, self.out_dim)

#     def forward(self, input):
        
#         input = input.view([-1, self.input_dim])
        
#         x1 = self.fc1(input)
#         x = self.act1(self.bn1(x1))
#         x2 = self.fc2(x)
#         x = self.act2(self.bn2(x2 + x1))

#         x3 = self.fc3(x)
#         x = self.act3(self.bn3(x3 + x2 + x1))
#         out = self.fc4(x)
#         return out

# ###############################

class LatentDeformator(nn.Module):
    def __init__(self, shift_dim=512, input_dim=12, out_dim=50, inner_dim=25,
                 random_init=False, bias=True):
        super(LatentDeformator, self).__init__()
        self.shift_dim = shift_dim
        self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
        self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

        self.fc1 = nn.Linear(self.input_dim, inner_dim)
        self.bn1 = nn.BatchNorm1d(inner_dim)
        self.act1 = nn.ELU()

        self.fc2 = nn.Linear(inner_dim, inner_dim)
        self.bn2 = nn.BatchNorm1d(inner_dim)
        self.act2 = nn.ELU()

        self.fc3 = nn.Linear(inner_dim, inner_dim)
        self.bn3 = nn.BatchNorm1d(inner_dim)
        self.act3 = nn.ELU()

        self.fc4 = nn.Linear(inner_dim, self.out_dim)

        

    def forward(self, input):
#         print("inside latent decode forward", input.shape)

        input = input.view([-1, self.input_dim])
        
        x1 = self.fc1(input)
        x = self.act1(self.bn1(x1))

        x2 = self.fc2(x)
        x = self.act2(self.bn2(x2 + x1))

        x3 = self.fc3(x)
        x = self.act3(self.bn3(x3 + x2 + x1))
#             print("inside latent deformator self.fc4(x)",self.fc4(x).shape)
#             print("inside latent deformator input",input.shape)

#             out = self.fc4(x) + input
        #input addition removed
        out = self.fc4(x)
        

        flat_shift_dim = np.product(self.shift_dim)
        if out.shape[1] < flat_shift_dim:
            padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[1]], device=out.device)
            out = torch.cat([out, padding], dim=1)
        elif out.shape[1] > flat_shift_dim:
            out = out[:, :flat_shift_dim]

        # handle spatial shifts
        try:
            out = out.view([-1] + self.shift_dim)
        except Exception:
            pass

        return out
# +
class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
            age_channels= 32,
            health_channel=3,
        )
        
        self.latent_shift_predictor = LatentDeformator(shift_dim=512, input_dim=12, out_dim=50, inner_dim=25,
                 random_init=False, bias=True)
#         self.latent_decode = Latent_decode()
#         self.image_regress = LatentShiftPredictor()

        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=1,##### changed conf.in_channels
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()
        

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):
        cond = self.encoder.forward(x)
        return {'cond': cond}

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S

    def forward(self,
                x,
                t,
                x_start=None,
                y=None,
                x_start_baseline = None,
                age_diff = None,
                health_state = None,
                cond = None,
                style = None,
                noise = None,
                t_cond = None,
                cond_shift_weight = 0,
                pred_with_cond = None,
                ventricular_mask=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
#         print("age_diff inside model forward call ",age_diff)
        self.pred_with_cond = pred_with_cond
        self.cond_shift_weight = cond_shift_weight
        self.cond = cond
        cond_post = None
        
        if t_cond is None:
            t_cond = t

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)
            
###### 
        if self.cond is None:
#             print("x shape,x_start shape", x.shape,x_start.shape)
            if x is not None:
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'
            

            tmp = self.encode(x_start_baseline)
            self.cond = tmp['cond']
#         print("After cond in model health_state", health_state,health_state.shape,health_state.dtype,health_state.device)
#         print("After cond in model age_diff", age_diff,age_diff.shape,age_diff.dtype,age_diff.device)
        
                
        ###################################### age and health state conditioning 
#         if age_diff is not None:            
#             _age_emb = timestep_embedding(age_diff, 32, max_period=10)
#             print("age_diff dtype",_age_emb.dtype)
        if health_state is not None:
            _health_emb = health_state
            _health_emb = _health_emb.to(torch.float32)
#             print("_health_emb dtype",_health_emb.dtype)
#             print("health_state",health_state.device)
#             _health_emb = timestep_embedding(health_state,32, max_period=10)
            
        ######################################
        self.latent_shift_predictor.to(_health_emb.device)
#         print("_health_emb",_health_emb.device)
        cond_shift = self.latent_shift_predictor(_health_emb)
#         print("After cond in model cond_shift", cond_shift,cond_shift.shape,cond_shift.dtype,cond_shift.device)
#         print("(self.cond_shift_weight)",self.cond_shift_weight)
        self.cond = self.cond + (self.cond_shift_weight)*cond_shift
#         if self.cond_shift_weight:
#             latent_decode_cond = self.latent_decode(cond_shift[:,0:50])
            
        
#         print("After cond in model cond_shift_weight = self.cond_shift_weight,",self.cond_shift_weight)
        ########################################
        
        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None
            
            
        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=self.cond,
                time_cond_emb=_t_cond_emb,
                cond_age = None,
                cond_health = None
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
            cond_age = None
            cond_health = None
#             cond_age = res.cond_age
#             cond_health = res.cond_health
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None
            cond_age = None
            cond_health = None

        # override the style if given
        style = style or res.style

#         assert (y is not None) == (
#             self.conf.num_classes is not None
#         ), "must specify y if and only if the model is class-conditional"

#         if self.conf.num_classes is not None:
#             raise NotImplementedError()
#             # assert y.shape == (x.shape[0], )
#             # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb
        
        
        # where in the model to supply age conditions
#         enc_cond_age_emb = cond_age
#         mid_cond_age_emb = cond_age
#         dec_cond_age_emb = cond_age
        enc_cond_age_emb = None
        mid_cond_age_emb = None
        dec_cond_age_emb = None
        
         # where in the model to supply health conditions
        enc_cond_health_emb = None
        mid_cond_health_emb = None
        dec_cond_health_emb = None


        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            ########################################## input block #############################################
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb,
                                            cond_age=enc_cond_age_emb,
                                            cond_health = enc_cond_health_emb)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)
            ########################################## middle block #############################################
            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb, cond_age=mid_cond_age_emb,                                 cond_health = mid_cond_health_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]
        ########################################## output block #############################################
        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          cond_age=dec_cond_age_emb,
                                          cond_health = dec_cond_health_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)
        
#         if self.cond_shift_weight and (self.pred_with_cond is not None):
#             if ventricular_mask is None:
#                 regress_age = self.image_regress((self.pred_with_cond-pred))
#             else:
#                 regress_age = self.image_regress((self.pred_with_cond-pred)*ventricular_mask)
            
#             return AutoencReturn(pred=pred, cond=cond, cond_age = cond_age,cond_health = cond_health, regress_age = regress_age)
        
#         else:
        if self.cond_shift_weight:
            return AutoencReturn(pred=pred, cond=self.cond, cond_age = cond_age,cond_health = cond_health)
#         ,latent_decode_cond = latent_decode_cond)
        else:
            return AutoencReturn(pred=pred, cond=self.cond, cond_age = cond_age,cond_health = cond_health)
# -

#         print("_t_emb",_t_emb.device)
#         print("cond",cond.device)
#         print("_t_cond_emb",_t_cond_emb.device)



class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None
    cond_age: Tensor = None
    cond_health: Tensor = None
    regress_age: Tensor = None
    latent_decode_cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None
    #age conditioning
    cond_age: Tensor = None
    #health conditioning
    cond_health: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels,age_channels=None,health_channel=None):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.age_embed = nn.Sequential(
            linear(age_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.health_embed = nn.Sequential(
            linear(health_channel, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, cond_age=None, cond_health = None, **kwargs):
#         print("time_emb",time_emb.device)
        ##### time
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        ##### age
        if cond_age is None:
            # happens with autoenc training mode
            cond_age = None
        else:
#             print("cond_age dtype",cond_age.dtype)
            cond_age = self.age_embed(cond_age)
        ##### health   
        if cond_health is None:
            # happens with autoenc training mode
            cond_health = None
        else:
            
            cond_health = self.health_embed(cond_health)
            
            
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style, cond_age = cond_age,cond_health =cond_health)
